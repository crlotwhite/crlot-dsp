#include <benchmark/benchmark.h>
#include "dsp/ola/OLAAccumulator.h"
#include "dsp/window/WindowLUT.h"
#include <vector>
#include <random>

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
        config_.center = false;
        config_.apply_window_inside = false;
        config_.gain = 1.0f;

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

// push_frame 성능 테스트
BENCHMARK_DEFINE_F(OLABenchmarkFixture, PushFrame)(benchmark::State& state) {
    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    int64_t frame_index = 0;

    for (auto _ : state) {
        ola.push_frame(frame_index++, test_frame_.data());
        benchmark::DoNotOptimize(ola);
    }

    // 처리량 계산 (frames per second)
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * config_.frame_size * sizeof(float));
}

// pull 성능 테스트
BENCHMARK_DEFINE_F(OLABenchmarkFixture, Pull)(benchmark::State& state) {
    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    // 충분한 데이터 미리 생성
    for (int i = 0; i < 10; ++i) {
        ola.push_frame(i, test_frame_.data());
    }

    for (auto _ : state) {
        int samples = ola.pull(output_buffer_.data(), config_.hop_size);
        benchmark::DoNotOptimize(samples);

        // 데이터 보충
        if (samples > 0) {
            static int frame_counter = 10;
            ola.push_frame(frame_counter++, test_frame_.data());
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
        for (int i = 0; i < 5; ++i) {
            ola.push_frame(i, test_frame_.data());
        }

        // 출력
        std::vector<float> output(config_.hop_size * 5);
        int total_samples = 0;
        while (total_samples < static_cast<int>(output.size())) {
            int samples = ola.pull(output.data() + total_samples,
                                  output.size() - total_samples);
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
    std::vector<float> stereo_frame(config_.frame_size * 2);

    // 스테레오 테스트 데이터
    for (int i = 0; i < config_.frame_size; ++i) {
        stereo_frame[i * 2] = test_frame_[i];      // L
        stereo_frame[i * 2 + 1] = test_frame_[i];  // R
    }

    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    std::vector<float> stereo_output(config_.hop_size * 2);

    for (auto _ : state) {
        ola.push_frame(0, stereo_frame.data());
        int samples = ola.pull(stereo_output.data(), config_.hop_size);
        benchmark::DoNotOptimize(samples);
    }

    state.SetItemsProcessed(state.iterations());
}

// 벤치마크 등록
BENCHMARK_REGISTER_F(OLABenchmarkFixture, PushFrame)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000);

BENCHMARK_REGISTER_F(OLABenchmarkFixture, Pull)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000);

BENCHMARK_REGISTER_F(OLABenchmarkFixture, FullPipeline)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000);

BENCHMARK_REGISTER_F(OLABenchmarkFixture, Multichannel)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(5000);

BENCHMARK_MAIN();