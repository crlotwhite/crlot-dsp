#include <benchmark/benchmark.h>
#include <vector>
#include <iostream>
#include <iomanip>

#include "dsp/ola/OLAAccumulator.h"
#include "dsp/window/WindowLUT.h"
#include "dsp/frame/FrameQueue.h"
#include "dsp/fft/api/fft_api.h"

using namespace dsp;

class PerformanceBenchmarkFixture : public benchmark::Fixture {
protected:
    void SetUp(const ::benchmark::State& state) override {
        // 벤치마크용 데이터 생성
        generateTestData();
    }

    void generateTestData() {
        // 다양한 크기의 테스트 데이터 생성
        test_data_1k_.resize(1024);
        test_data_4k_.resize(4096);
        test_data_16k_.resize(16384);

        // 실제적인 오디오 신호 시뮬레이션 (복합 사인파)
        for (size_t i = 0; i < test_data_16k_.size(); ++i) {
            float t = static_cast<float>(i) / 48000.0f;
            test_data_16k_[i] = 0.3f * std::sin(2.0f * M_PI * 440.0f * t) +
                               0.2f * std::sin(2.0f * M_PI * 880.0f * t) +
                               0.1f * std::sin(2.0f * M_PI * 1760.0f * t);
        }

        // 작은 데이터는 큰 데이터에서 추출
        std::copy(test_data_16k_.begin(), test_data_16k_.begin() + 1024, test_data_1k_.begin());
        std::copy(test_data_16k_.begin(), test_data_16k_.begin() + 4096, test_data_4k_.begin());
    }

    std::vector<float> test_data_1k_;
    std::vector<float> test_data_4k_;
    std::vector<float> test_data_16k_;
};

// OLA Accumulator 성능 테스트
BENCHMARK_DEFINE_F(PerformanceBenchmarkFixture, OLAAccumulatorPerformance)(benchmark::State& state) {
    int frame_size = static_cast<int>(state.range(0));
    int hop_size = static_cast<int>(state.range(1));
    int channels = static_cast<int>(state.range(2));

    OLAConfig config;
    config.sample_rate = 48000;
    config.frame_size = frame_size;
    config.hop_size = hop_size;
    config.channels = channels;
    config.center = true;
    config.apply_window_inside = true;

    OLAAccumulator ola(config);

    // 윈도우 설정
    WindowLUT& lut = WindowLUT::getInstance();
    auto window = lut.GetWindowSafe(WindowType::HANN, frame_size);
    ola.set_window(window.get(), frame_size);

    // 테스트 프레임 준비
    std::vector<float> test_frame(frame_size * channels);
    for (int i = 0; i < frame_size * channels; ++i) {
        test_frame[i] = test_data_1k_[i % test_data_1k_.size()];
    }

    std::vector<float> output(hop_size * channels);

    for (auto _ : state) {
        // push_frame
        ola.push_frame(0, test_frame.data());
        benchmark::DoNotOptimize(ola);

        // pull
        int samples = ola.pull(output.data(), hop_size);
        benchmark::DoNotOptimize(samples);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * (frame_size + hop_size) * channels * sizeof(float));
}

// WindowLUT 성능 테스트
BENCHMARK_DEFINE_F(PerformanceBenchmarkFixture, WindowLUTPerformance)(benchmark::State& state) {
    WindowType window_type = static_cast<WindowType>(state.range(0));
    size_t window_size = static_cast<size_t>(state.range(1));

    WindowLUT& lut = WindowLUT::getInstance();

    // 캐시 초기화
    lut.clearCache(true);

    for (auto _ : state) {
        auto window = lut.GetWindowSafe(window_type, window_size);
        benchmark::DoNotOptimize(window.get());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * window_size * sizeof(float));
}

// FFT 성능 테스트
BENCHMARK_DEFINE_F(PerformanceBenchmarkFixture, FFTPerformance)(benchmark::State& state) {
    int nfft = static_cast<int>(state.range(0));
    int batch = static_cast<int>(state.range(1));

    using namespace dsp::fft;

    FftPlanDesc desc;
    desc.domain = FftDomain::Real;
    desc.nfft = nfft;
    desc.in_place = false;
    desc.batch = 1;  // 배치 1로 고정
    desc.stride_in = 1;
    desc.stride_out = 1;

    auto plan = MakeFftPlan(desc);

    // 테스트 데이터 준비
    std::vector<float> input(batch * nfft);
    std::vector<std::complex<float>> output(batch * (nfft / 2 + 1));
    std::vector<float> reconstructed(batch * nfft);

    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = test_data_16k_[i % test_data_16k_.size()];
    }

    for (auto _ : state) {
        // Forward FFT
        plan->forward(input.data(), output.data(), batch);
        benchmark::DoNotOptimize(output.data());

        // Inverse FFT
        plan->inverse(output.data(), reconstructed.data(), batch);
        benchmark::DoNotOptimize(reconstructed.data());
    }

    state.SetItemsProcessed(state.iterations() * batch);
    state.SetBytesProcessed(state.iterations() * batch * nfft * sizeof(float));
}

// FrameQueue 성능 테스트
BENCHMARK_DEFINE_F(PerformanceBenchmarkFixture, FrameQueuePerformance)(benchmark::State& state) {
    size_t input_size = static_cast<size_t>(state.range(0));
    size_t frame_size = static_cast<size_t>(state.range(1));
    size_t hop_size = static_cast<size_t>(state.range(2));
    bool center = static_cast<bool>(state.range(3));

    const std::vector<float>& input_data = (input_size <= 4096) ? test_data_4k_ : test_data_16k_;

    FrameQueue fq(input_data.data(), input_size, frame_size, hop_size, center);
    size_t num_frames = fq.getNumFrames();

    std::vector<float> frame_buffer(frame_size);

    for (auto _ : state) {
        for (size_t i = 0; i < num_frames; ++i) {
            fq.copyFrame(i, frame_buffer.data());
            benchmark::DoNotOptimize(frame_buffer.data());
        }
    }

    state.SetItemsProcessed(state.iterations() * num_frames);
    state.SetBytesProcessed(state.iterations() * num_frames * frame_size * sizeof(float));
}

// 통합 파이프라인 성능 테스트
BENCHMARK_DEFINE_F(PerformanceBenchmarkFixture, IntegratedPipelinePerformance)(benchmark::State& state) {
    const size_t input_length = 16384;
    const size_t frame_size = 1024;
    const size_t hop_size = 512;

    for (auto _ : state) {
        // 1. 프레임 분할
        FrameQueue frames(test_data_16k_.data(), input_length, frame_size, hop_size, true);

        // 2. 윈도우 준비
        WindowLUT& lut = WindowLUT::getInstance();
        auto window = lut.GetWindowSafe(WindowType::HANN, frame_size);

        // 3. OLA 합성기 설정
        OLAConfig config;
        config.sample_rate = 48000;
        config.frame_size = frame_size;
        config.hop_size = hop_size;
        config.channels = 1;
        config.center = true;
        config.apply_window_inside = true;

        OLAAccumulator ola(config);
        ola.set_window(window.get(), frame_size);

        // 4. FFT 플랜 준비
        using namespace dsp::fft;
        FftPlanDesc desc;
        desc.domain = FftDomain::Real;
        desc.nfft = frame_size;
        desc.in_place = false;
        desc.batch = 1;
        desc.stride_in = 1;
        desc.stride_out = 1;
        auto fft_plan = MakeFftPlan(desc);

        // 5. 프레임별 처리
        std::vector<std::complex<float>> spectrum(frame_size / 2 + 1);
        std::vector<float> processed_frame(frame_size);

        for (size_t i = 0; i < frames.getNumFrames(); ++i) {
            const float* frame = frames.getFrame(i);
            std::copy(frame, frame + frame_size, processed_frame.begin());

            // FFT 분석 (in-place)
            fft_plan->forward(processed_frame.data(), spectrum.data());

            // 간단한 처리 (여기서는 그대로 통과)

            // IFFT 합성 (in-place)
            fft_plan->inverse(spectrum.data(), processed_frame.data());

            // OLA 누적
            ola.push_frame(i, processed_frame.data());
        }

        // 6. 출력 생성
        std::vector<float> output(input_length);
        int total_output = 0;
        while (total_output < static_cast<int>(input_length)) {
            int samples = ola.pull(output.data() + total_output, hop_size);
            if (samples == 0) break;
            total_output += samples;
        }

        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * 16384 * sizeof(float));
}

// 벤치마크 등록
BENCHMARK_REGISTER_F(PerformanceBenchmarkFixture, OLAAccumulatorPerformance)
    ->Args({512, 256, 1})
    ->Args({1024, 512, 1})
    ->Args({2048, 1024, 1})
    ->Args({1024, 512, 2})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(PerformanceBenchmarkFixture, WindowLUTPerformance)
    ->Args({static_cast<int>(WindowType::HANN), 512})
    ->Args({static_cast<int>(WindowType::HANN), 1024})
    ->Args({static_cast<int>(WindowType::HANN), 2048})
    ->Args({static_cast<int>(WindowType::HANN), 4096})
    ->Args({static_cast<int>(WindowType::HAMMING), 1024})
    ->Args({static_cast<int>(WindowType::BLACKMAN), 1024})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(PerformanceBenchmarkFixture, FFTPerformance)
    ->Args({512, 1})
    ->Args({1024, 1})
    ->Args({2048, 1})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(PerformanceBenchmarkFixture, FrameQueuePerformance)
    ->Args({4096, 512, 256, 0})  // no-center
    ->Args({4096, 512, 256, 1})  // center
    ->Args({16384, 1024, 512, 1})
    ->Args({16384, 2048, 1024, 1})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(PerformanceBenchmarkFixture, IntegratedPipelinePerformance)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10);

BENCHMARK_MAIN();