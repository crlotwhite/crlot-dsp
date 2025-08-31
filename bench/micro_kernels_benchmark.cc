#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include "dsp/ola/OLAAccumulator.h"
#include "dsp/window/WindowLUT.h"

using namespace dsp;

// 커널 마이크로벤치마크
class KernelMicroBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // 테스트 데이터 생성
        generateTestData();
        setupOLA();
    }

    void generateTestData() {
        // 테스트 프레임 생성
        frame_size_ = 1024;
        hop_size_ = 256;
        num_frames_ = 100;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        test_frames_.resize(num_frames_ * frame_size_);
        for (auto& sample : test_frames_) {
            sample = dist(gen);
        }

        // 윈도우 생성
        WindowLUT& lut = WindowLUT::getInstance();
        auto safe_window = lut.GetWindowSafe(WindowType::HANN, frame_size_);
        window_ = safe_window.get();
    }

    void setupOLA() {
        OLAConfig config;
        config.sample_rate = 48000;
        config.frame_size = frame_size_;
        config.hop_size = hop_size_;
        config.channels = 1;
        config.center = false;
        config.apply_window_inside = true;
        config.gain = 1.0f;

        ola_ = std::make_unique<OLAAccumulator>(config);
        ola_->set_window(window_, frame_size_);
    }

    size_t frame_size_;
    size_t hop_size_;
    size_t num_frames_;
    std::vector<float> test_frames_;
    const float* window_;
    std::unique_ptr<OLAAccumulator> ola_;
};

// 윈도우 곱셈 커널 벤치마크
BENCHMARK_DEFINE_F(KernelMicroBenchmark, WindowMultiply)(benchmark::State& state) {
    std::vector<float> frame(frame_size_);
    std::vector<float> output(frame_size_);

    for (auto _ : state) {
        // 단일 프레임에 윈도우 곱셈
        for (size_t i = 0; i < frame_size_; ++i) {
            output[i] = test_frames_[i] * window_[i];
        }
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(state.iterations() * frame_size_);
    state.SetBytesProcessed(state.iterations() * frame_size_ * sizeof(float));
}

// OLA 누적 커널 벤치마크 (push_frame)
BENCHMARK_DEFINE_F(KernelMicroBenchmark, OLAAccumulatePush)(benchmark::State& state) {
    int64_t frame_index = 0;

    for (auto _ : state) {
        ola_->push_frame(frame_index++, test_frames_.data());
        benchmark::DoNotOptimize(ola_.get());
    }

    state.SetItemsProcessed(state.iterations() * frame_size_);
    state.SetBytesProcessed(state.iterations() * frame_size_ * sizeof(float));
}

// OLA 누적 커널 벤치마크 (pull)
BENCHMARK_DEFINE_F(KernelMicroBenchmark, OLAAccumulatePull)(benchmark::State& state) {
    // 충분한 데이터 미리 푸시
    for (int i = 0; i < 10; ++i) {
        ola_->push_frame(i, test_frames_.data());
    }

    std::vector<float> output(hop_size_);

    for (auto _ : state) {
        int samples = ola_->pull(output.data(), hop_size_);
        benchmark::DoNotOptimize(samples);

        // 데이터 보충
        if (samples > 0) {
            static int frame_counter = 10;
            ola_->push_frame(frame_counter++, test_frames_.data());
        }
    }

    state.SetItemsProcessed(state.iterations() * hop_size_);
    state.SetBytesProcessed(state.iterations() * hop_size_ * sizeof(float));
}

// 윈도우 + OLA 통합 커널 벤치마크
BENCHMARK_DEFINE_F(KernelMicroBenchmark, WindowOLAKernel)(benchmark::State& state) {
    std::vector<float> frame(frame_size_);
    std::vector<float> output(hop_size_);
    int64_t frame_index = 0;

    for (auto _ : state) {
        // 윈도우 적용
        for (size_t i = 0; i < frame_size_; ++i) {
            frame[i] = test_frames_[i] * window_[i];
        }

        // OLA 누적
        ola_->push_frame(frame_index++, frame.data());
        int samples = ola_->pull(output.data(), hop_size_);
        benchmark::DoNotOptimize(samples);
    }

    state.SetItemsProcessed(state.iterations() * frame_size_);
    state.SetBytesProcessed(state.iterations() * (frame_size_ + hop_size_) * sizeof(float));
}

// SIMD 최적화된 윈도우 곱셈 (수동 벡터화)
BENCHMARK_DEFINE_F(KernelMicroBenchmark, WindowMultiplySIMD)(benchmark::State& state) {
    std::vector<float> frame(frame_size_);
    std::vector<float> output(frame_size_);

    // SIMD 처리 가능한 크기 (8개 float 단위)
    const size_t vec_size = 8;
    const size_t vec_end = (frame_size_ / vec_size) * vec_size;

    for (auto _ : state) {
        // SIMD 벡터화된 루프
        for (size_t i = 0; i < vec_end; i += vec_size) {
            // 수동 SIMD 시뮬레이션 (실제로는 컴파일러가 벡터화)
            for (size_t j = 0; j < vec_size; ++j) {
                output[i + j] = test_frames_[i + j] * window_[i + j];
            }
        }

        // 나머지 처리
        for (size_t i = vec_end; i < frame_size_; ++i) {
            output[i] = test_frames_[i] * window_[i];
        }

        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(state.iterations() * frame_size_);
    state.SetBytesProcessed(state.iterations() * frame_size_ * sizeof(float));
}

// 벤치마크 등록
BENCHMARK_REGISTER_F(KernelMicroBenchmark, WindowMultiply)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(100000);

BENCHMARK_REGISTER_F(KernelMicroBenchmark, OLAAccumulatePush)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(10000);

BENCHMARK_REGISTER_F(KernelMicroBenchmark, OLAAccumulatePull)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(10000);

BENCHMARK_REGISTER_F(KernelMicroBenchmark, WindowOLAKernel)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(5000);

BENCHMARK_REGISTER_F(KernelMicroBenchmark, WindowMultiplySIMD)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(100000);

// 커스텀 리포터: 커널 성능 로깅 (ns/elem)
class KernelPerformanceReporter : public benchmark::ConsoleReporter {
public:
    KernelPerformanceReporter() = default;

    void ReportRuns(const std::vector<Run>& reports) override {
        ConsoleReporter::ReportRuns(reports);

        // 커널 성능 요약 (ns per element)
        std::cout << "\n=== Kernel Performance Summary (ns/elem) ===\n";
        std::cout << std::fixed << std::setprecision(3);

        for (const auto& report : reports) {
            if (report.benchmark_name().find("Kernel") != std::string::npos ||
                report.benchmark_name().find("Multiply") != std::string::npos ||
                report.benchmark_name().find("Accumulate") != std::string::npos) {

                double time_ns = report.GetAdjustedRealTime() * 1000.0; // us to ns

                std::cout << report.benchmark_name() << ": "
                         << time_ns << " ns total\n";
            }
        }
        std::cout << "===========================================\n";
    }
};

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    // 커스텀 리포터 설정
    KernelPerformanceReporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);

    return 0;
}
