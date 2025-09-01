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

BENCHMARK_MAIN();