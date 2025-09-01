#include <benchmark/benchmark.h>
#include "dsp/ola/OLAAccumulator.h"
#include "dsp/ola/kernels.h"
#include "dsp/window/WindowLUT.h"
#include <vector>
#include <random>
#include <chrono>

using namespace dsp;

// 벤치마크용 테스트 데이터 생성
class OLABenchmarkFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // 설정
        config_.sample_rate = 48000;
        config_.frame_size = 1024;
        config_.hop_size = 256;
        config_.channels = 1;
        config_.eps = 1e-8f;
        config_.apply_window_inside = false;

        // 윈도우 설정
        WindowLUT& lut = WindowLUT::getInstance();
        window_ = lut.GetWindow(WindowType::HANN, config_.frame_size);

        // 테스트 프레임 생성
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        test_frame_.resize(config_.frame_size);
        for (auto& sample : test_frame_) {
            sample = dist(gen);
        }

        output_buffer_.resize(config_.hop_size);
    }

protected:
    OLAConfig config_;
    const float* window_;
    std::vector<float> test_frame_;
    std::vector<float> output_buffer_;
};

// 파라미터화된 벤치마크용 픽스처
class OLAParameterizedFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // 파라미터 추출: frame_size, hop_ratio, channels, gain, apply_window
        const auto frame_size = static_cast<size_t>(state.range(0));
        const auto hop_ratio = state.range(1); // 0: N/4, 1: N/2
        const auto channels = static_cast<size_t>(state.range(2));
        const auto gain_idx = state.range(3); // 0: 1.0f, 1: 0.5f
        const auto window_idx = state.range(4); // 0: false, 1: true

        // 설정
        config_.sample_rate = 48000;
        config_.frame_size = frame_size;
        config_.hop_size = (hop_ratio == 0) ? frame_size / 4 : frame_size / 2;
        config_.channels = channels;
        config_.eps = 1e-8f;
        config_.apply_window_inside = (window_idx == 1);

        gain_ = (gain_idx == 0) ? 1.0f : 0.5f;

        // 윈도우 설정
        WindowLUT& lut = WindowLUT::getInstance();
        window_ = lut.GetWindow(WindowType::HANN, config_.frame_size);

        // 테스트 프레임 생성
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        test_frame_.resize(config_.frame_size);
        for (auto& sample : test_frame_) {
            sample = dist(gen);
        }

        output_buffer_.resize(config_.hop_size);

        // 다채널 프레임 생성
        multichannel_frames_.resize(config_.channels);
        for (auto& frame : multichannel_frames_) {
            frame.resize(config_.frame_size);
            for (auto& sample : frame) {
                sample = dist(gen);
            }
        }

        // AoS 프레임 생성
        aos_frame_.resize(config_.frame_size * config_.channels);
        for (size_t i = 0; i < config_.frame_size; ++i) {
            for (size_t ch = 0; ch < config_.channels; ++ch) {
                aos_frame_[i * config_.channels + ch] = multichannel_frames_[ch][i];
            }
        }

        // 출력 버퍼들
        multichannel_output_.resize(config_.channels);
        for (auto& buf : multichannel_output_) {
            buf.resize(config_.hop_size);
        }
    }

protected:
    OLAConfig config_;
    const float* window_;
    float gain_;
    std::vector<float> test_frame_;
    std::vector<float> output_buffer_;
    std::vector<std::vector<float>> multichannel_frames_;
    std::vector<float> aos_frame_;
    std::vector<std::vector<float>> multichannel_output_;
};

// add_frame_SoA 성능 테스트
BENCHMARK_DEFINE_F(OLABenchmarkFixture, AddFrameSoA)(benchmark::State& state) {
    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    const float* ch_frames[1] = {test_frame_.data()};

    for (auto _ : state) {
        ola.add_frame_SoA(ch_frames, window_, 0, 0, config_.frame_size, 1.0f);
        benchmark::DoNotOptimize(ola);
    }

    // 처리량 계산 (frames per second)
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * config_.frame_size * sizeof(float));
}

// produce 성능 테스트
BENCHMARK_DEFINE_F(OLABenchmarkFixture, Produce)(benchmark::State& state) {
    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    // 충분한 데이터 미리 생성
    const float* ch_frames[1] = {test_frame_.data()};
    for (int i = 0; i < 10; ++i) {
        ola.add_frame_SoA(ch_frames, window_, i * config_.hop_size, 0, config_.frame_size, 1.0f);
    }

    float* ch_out[1] = {output_buffer_.data()};

    for (auto _ : state) {
        size_t samples = ola.produce(ch_out, config_.hop_size);
        benchmark::DoNotOptimize(samples);

        // 데이터 보충
        if (samples > 0) {
            static int frame_counter = 10;
            ola.add_frame_SoA(ch_frames, window_, frame_counter++ * config_.hop_size, 0, config_.frame_size, 1.0f);
        }
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * config_.hop_size * sizeof(float));
}

// 전체 파이프라인 성능 테스트
BENCHMARK_DEFINE_F(OLABenchmarkFixture, FullPipeline)(benchmark::State& state) {
    for (auto _ : state) {
        OLAAccumulator ola(config_);
        ola.set_window(window_, config_.frame_size);

        // 여러 프레임 처리
        const float* ch_frames[1] = {test_frame_.data()};
        for (int i = 0; i < 5; ++i) {
            ola.add_frame_SoA(ch_frames, window_, i * config_.hop_size, 0, config_.frame_size, 1.0f);
        }

        // 출력
        std::vector<float> output(config_.hop_size * 5);
        float* ch_out[1] = {output.data()};
        size_t total_samples = 0;
        while (total_samples < output.size()) {
            size_t samples = ola.produce(ch_out, output.size() - total_samples);
            if (samples == 0) break;
            total_samples += samples;
        }

        benchmark::DoNotOptimize(total_samples);
    }

    state.SetItemsProcessed(state.iterations() * 5); // 5 frames per iteration
}

// 다채널 성능 테스트
BENCHMARK_DEFINE_F(OLABenchmarkFixture, Multichannel)(benchmark::State& state) {
    config_.channels = 2;
    std::vector<float> ch0_frame(config_.frame_size);
    std::vector<float> ch1_frame(config_.frame_size);

    // 스테레오 테스트 데이터
    for (int i = 0; i < config_.frame_size; ++i) {
        ch0_frame[i] = test_frame_[i];  // L
        ch1_frame[i] = test_frame_[i];  // R
    }

    const float* ch_frames[2] = {ch0_frame.data(), ch1_frame.data()};

    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    std::vector<float> ch0_out(config_.hop_size);
    std::vector<float> ch1_out(config_.hop_size);
    float* ch_out[2] = {ch0_out.data(), ch1_out.data()};

    for (auto _ : state) {
        ola.add_frame_SoA(ch_frames, window_, 0, 0, config_.frame_size, 1.0f);
        size_t samples = ola.produce(ch_out, config_.hop_size);
        benchmark::DoNotOptimize(samples);
    }

    state.SetItemsProcessed(state.iterations());
}

// push_frame_AoS 성능 테스트
BENCHMARK_DEFINE_F(OLABenchmarkFixture, AddFrameAoS)(benchmark::State& state) {
    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    // AoS 입력 생성
    std::vector<float> aos_frame(config_.frame_size * config_.channels);
    for (size_t i = 0; i < config_.frame_size; ++i) {
        aos_frame[i * config_.channels] = test_frame_[i];
    }

    for (auto _ : state) {
        ola.push_frame_AoS(aos_frame.data(), window_, 0, 0, config_.frame_size, 1.0f);
        benchmark::DoNotOptimize(ola);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * config_.frame_size * sizeof(float));
}

// AoS vs SoA 성능 비교
BENCHMARK_DEFINE_F(OLABenchmarkFixture, AoSvsSoAComparison)(benchmark::State& state) {
    // SoA 설정
    OLAAccumulator ola_soa(config_);
    ola_soa.set_window(window_, config_.frame_size);
    const float* ch_frames[1] = {test_frame_.data()};

    // AoS 설정
    OLAAccumulator ola_aos(config_);
    ola_aos.set_window(window_, config_.frame_size);
    std::vector<float> aos_frame(config_.frame_size * config_.channels);
    for (size_t i = 0; i < config_.frame_size; ++i) {
        aos_frame[i * config_.channels] = test_frame_[i];
    }

    double soa_time = 0.0;
    double aos_time = 0.0;

    for (auto _ : state) {
        // SoA 측정
        auto start = std::chrono::high_resolution_clock::now();
        ola_soa.add_frame_SoA(ch_frames, window_, 0, 0, config_.frame_size, 1.0f);
        auto end = std::chrono::high_resolution_clock::now();
        soa_time += std::chrono::duration<double, std::micro>(end - start).count();

        // AoS 측정
        start = std::chrono::high_resolution_clock::now();
        ola_aos.push_frame_AoS(aos_frame.data(), window_, 0, 0, config_.frame_size, 1.0f);
        end = std::chrono::high_resolution_clock::now();
        aos_time += std::chrono::duration<double, std::micro>(end - start).count();
    }

    // 결과 보고
    state.counters["SoA_time_us"] = benchmark::Counter(soa_time / state.iterations(),
                                                       benchmark::Counter::kAvgThreads);
    state.counters["AoS_time_us"] = benchmark::Counter(aos_time / state.iterations(),
                                                       benchmark::Counter::kAvgThreads);
    state.counters["AoS_overhead_us"] = benchmark::Counter((aos_time - soa_time) / state.iterations(),
                                                           benchmark::Counter::kAvgThreads);

    state.SetItemsProcessed(state.iterations());
}

// === 파라미터화된 벤치마크들 ===

// 파라미터화된 AddFrameSoA 성능 테스트
BENCHMARK_DEFINE_F(OLAParameterizedFixture, ParamAddFrameSoA)(benchmark::State& state) {
    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    std::vector<const float*> ch_frames(config_.channels);
    for (size_t ch = 0; ch < config_.channels; ++ch) {
        ch_frames[ch] = multichannel_frames_[ch].data();
    }

    for (auto _ : state) {
        ola.add_frame_SoA(ch_frames.data(), window_, 0, 0, config_.frame_size, gain_);
        benchmark::DoNotOptimize(ola);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * config_.frame_size * config_.channels * sizeof(float));
}

// 파라미터화된 Produce 성능 테스트
BENCHMARK_DEFINE_F(OLAParameterizedFixture, ParamProduce)(benchmark::State& state) {
    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    // 충분한 데이터 미리 생성
    std::vector<const float*> ch_frames(config_.channels);
    for (size_t ch = 0; ch < config_.channels; ++ch) {
        ch_frames[ch] = multichannel_frames_[ch].data();
    }
    for (int i = 0; i < 10; ++i) {
        ola.add_frame_SoA(ch_frames.data(), window_, i * config_.hop_size, 0, config_.frame_size, gain_);
    }

    std::vector<float*> ch_out(config_.channels);
    for (size_t ch = 0; ch < config_.channels; ++ch) {
        ch_out[ch] = multichannel_output_[ch].data();
    }

    for (auto _ : state) {
        size_t samples = ola.produce(ch_out.data(), config_.hop_size);
        benchmark::DoNotOptimize(samples);

        // 데이터 보충
        if (samples > 0) {
            static int frame_counter = 10;
            ola.add_frame_SoA(ch_frames.data(), window_, frame_counter++ * config_.hop_size, 0, config_.frame_size, gain_);
        }
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * config_.hop_size * config_.channels * sizeof(float));
}

// 파라미터화된 전체 파이프라인 성능 테스트
BENCHMARK_DEFINE_F(OLAParameterizedFixture, ParamFullPipeline)(benchmark::State& state) {
    for (auto _ : state) {
        OLAAccumulator ola(config_);
        ola.set_window(window_, config_.frame_size);

        // 여러 프레임 처리
        std::vector<const float*> ch_frames(config_.channels);
        for (size_t ch = 0; ch < config_.channels; ++ch) {
            ch_frames[ch] = multichannel_frames_[ch].data();
        }
        for (int i = 0; i < 5; ++i) {
            ola.add_frame_SoA(ch_frames.data(), window_, i * config_.hop_size, 0, config_.frame_size, gain_);
        }

        // 출력
        std::vector<std::vector<float>> output(config_.channels, std::vector<float>(config_.hop_size * 5));
        std::vector<float*> ch_out(config_.channels);
        for (size_t ch = 0; ch < config_.channels; ++ch) {
            ch_out[ch] = output[ch].data();
        }
        size_t total_samples = 0;
        while (total_samples < output[0].size()) {
            size_t samples = ola.produce(ch_out.data(), output[0].size() - total_samples);
            if (samples == 0) break;
            total_samples += samples;
            for (size_t ch = 0; ch < config_.channels; ++ch) {
                ch_out[ch] += samples;
            }
        }

        benchmark::DoNotOptimize(total_samples);
    }

    state.SetItemsProcessed(state.iterations() * 5); // 5 frames per iteration
}

// 파라미터화된 AoS vs SoA 성능 비교
BENCHMARK_DEFINE_F(OLAParameterizedFixture, ParamAoSvsSoAComparison)(benchmark::State& state) {
    // SoA 설정
    OLAAccumulator ola_soa(config_);
    ola_soa.set_window(window_, config_.frame_size);
    std::vector<const float*> ch_frames(config_.channels);
    for (size_t ch = 0; ch < config_.channels; ++ch) {
        ch_frames[ch] = multichannel_frames_[ch].data();
    }

    // AoS 설정
    OLAAccumulator ola_aos(config_);
    ola_aos.set_window(window_, config_.frame_size);

    double soa_time = 0.0;
    double aos_time = 0.0;

    for (auto _ : state) {
        // SoA 측정
        auto start = std::chrono::high_resolution_clock::now();
        ola_soa.add_frame_SoA(ch_frames.data(), window_, 0, 0, config_.frame_size, gain_);
        auto end = std::chrono::high_resolution_clock::now();
        soa_time += std::chrono::duration<double, std::micro>(end - start).count();

        // AoS 측정
        start = std::chrono::high_resolution_clock::now();
        ola_aos.push_frame_AoS(aos_frame_.data(), window_, 0, 0, config_.frame_size, gain_);
        end = std::chrono::high_resolution_clock::now();
        aos_time += std::chrono::duration<double, std::micro>(end - start).count();
    }

    // 결과 보고
    state.counters["SoA_time_us"] = benchmark::Counter(soa_time / state.iterations(),
                                                       benchmark::Counter::kAvgThreads);
    state.counters["AoS_time_us"] = benchmark::Counter(aos_time / state.iterations(),
                                                       benchmark::Counter::kAvgThreads);
    state.counters["AoS_overhead_us"] = benchmark::Counter((aos_time - soa_time) / state.iterations(),
                                                           benchmark::Counter::kAvgThreads);

    state.SetItemsProcessed(state.iterations());
}

// axpy_windowed 커널 성능 테스트
BENCHMARK_DEFINE_F(OLAParameterizedFixture, AxpyWindowedKernel)(benchmark::State& state) {
    // 테스트 데이터 준비
    std::vector<float> dst(config_.frame_size, 0.0f);
    std::vector<float> src = test_frame_;
    std::vector<float> win(config_.frame_size);
    for (size_t i = 0; i < config_.frame_size; ++i) {
        win[i] = static_cast<float>(i) / config_.frame_size; // 간단한 윈도우
    }

    for (auto _ : state) {
        axpy_windowed(dst.data(), src.data(), win.data(), gain_, config_.frame_size);
        benchmark::DoNotOptimize(dst);
    }

    state.SetItemsProcessed(state.iterations() * config_.frame_size);
    state.SetBytesProcessed(state.iterations() * config_.frame_size * 3 * sizeof(float)); // dst, src, win
}

// 벤치마크 등록
BENCHMARK_REGISTER_F(OLABenchmarkFixture, AddFrameSoA)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000);

BENCHMARK_REGISTER_F(OLABenchmarkFixture, Produce)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000);

BENCHMARK_REGISTER_F(OLABenchmarkFixture, FullPipeline)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000);

BENCHMARK_REGISTER_F(OLABenchmarkFixture, Multichannel)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(5000);

BENCHMARK_REGISTER_F(OLABenchmarkFixture, AddFrameAoS)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000);

BENCHMARK_REGISTER_F(OLABenchmarkFixture, AoSvsSoAComparison)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000);

// 파라미터화된 벤치마크 등록
// Args: frame_size, hop_ratio(0=N/4,1=N/2), channels, gain_idx(0=1.0,1=0.5), window_idx(0=false,1=true)
BENCHMARK_REGISTER_F(OLAParameterizedFixture, ParamAddFrameSoA)
    ->ArgsProduct({
        {1024, 2048, 4096},      // frame_size
        {0, 1},                  // hop_ratio
        {1, 2, 4},              // channels
        {0, 1},                  // gain
        {0, 1}                   // apply_window_inside
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000);

BENCHMARK_REGISTER_F(OLAParameterizedFixture, ParamProduce)
    ->ArgsProduct({
        {1024, 2048, 4096},      // frame_size
        {0, 1},                  // hop_ratio
        {1, 2, 4},              // channels
        {0, 1},                  // gain
        {0, 1}                   // apply_window_inside
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000);

BENCHMARK_REGISTER_F(OLAParameterizedFixture, ParamFullPipeline)
    ->ArgsProduct({
        {1024, 2048, 4096},      // frame_size
        {0, 1},                  // hop_ratio
        {1, 2, 4},              // channels
        {0, 1},                  // gain
        {0, 1}                   // apply_window_inside
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500);

BENCHMARK_REGISTER_F(OLAParameterizedFixture, ParamAoSvsSoAComparison)
    ->ArgsProduct({
        {1024, 2048, 4096},      // frame_size
        {0, 1},                  // hop_ratio
        {1, 2, 4},              // channels
        {0, 1},                  // gain
        {0, 1}                   // apply_window_inside
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500);

BENCHMARK_REGISTER_F(OLAParameterizedFixture, AxpyWindowedKernel)
    ->ArgsProduct({
        {1024, 2048, 4096},      // frame_size
        {0, 1},                  // hop_ratio (not used but needed for fixture)
        {1, 2, 4},              // channels (not used but needed for fixture)
        {0, 1},                  // gain
        {0, 1}                   // apply_window_inside (not used but needed for fixture)
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000);

BENCHMARK_MAIN();