#include <benchmark/benchmark.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "dsp/fft/api/fft_api.h"

using namespace dsp::fft;

// FFT 마이크로벤치마크
class FFTMicroBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // 테스트 데이터 생성
        generateTestData();
        // FFT 플랜 생성
        setupFFTPlans();
    }

    void generateTestData() {
        // 다양한 크기의 테스트 데이터
        test_data_512_.resize(512);
        test_data_1024_.resize(1024);
        test_data_2048_.resize(2048);

        // 실제적인 오디오 신호 시뮬레이션
        for (size_t i = 0; i < test_data_2048_.size(); ++i) {
            double t = static_cast<double>(i) / 48000.0;
            test_data_2048_[i] = 0.3f * std::sin(2.0 * M_PI * 440.0 * t) +
                               0.2f * std::sin(2.0 * M_PI * 880.0 * t) +
                               0.1f * std::sin(2.0 * M_PI * 1760.0 * t);
        }

        // 작은 데이터는 큰 데이터에서 추출
        std::copy(test_data_2048_.begin(), test_data_2048_.begin() + 512, test_data_512_.begin());
        std::copy(test_data_2048_.begin(), test_data_2048_.begin() + 1024, test_data_1024_.begin());
    }

    void setupFFTPlans() {
        // 단일 FFT 플랜들
        {
            FftPlanDesc desc;
            desc.domain = FftDomain::Real;
            desc.nfft = 512;
            desc.in_place = false;
            desc.batch = 1;
            desc.stride_in = 1;
            desc.stride_out = 1;
            plan_512_ = MakeFftPlan(desc);
        }
        {
            FftPlanDesc desc;
            desc.domain = FftDomain::Real;
            desc.nfft = 1024;
            desc.in_place = false;
            desc.batch = 1;
            desc.stride_in = 1;
            desc.stride_out = 1;
            plan_1024_ = MakeFftPlan(desc);
        }
        {
            FftPlanDesc desc;
            desc.domain = FftDomain::Real;
            desc.nfft = 2048;
            desc.in_place = false;
            desc.batch = 1;
            desc.stride_in = 1;
            desc.stride_out = 1;
            plan_2048_ = MakeFftPlan(desc);
        }

        // 배치 FFT 플랜들
        {
            FftPlanDesc desc;
            desc.domain = FftDomain::Real;
            desc.nfft = 512;
            desc.in_place = false;
            desc.batch = 4;
            desc.stride_in = 1;
            desc.stride_out = 1;
            plan_batch_512_ = MakeFftPlan(desc);
        }
        {
            FftPlanDesc desc;
            desc.domain = FftDomain::Real;
            desc.nfft = 1024;
            desc.in_place = false;
            desc.batch = 4;
            desc.stride_in = 1;
            desc.stride_out = 1;
            plan_batch_1024_ = MakeFftPlan(desc);
        }
        {
            FftPlanDesc desc;
            desc.domain = FftDomain::Real;
            desc.nfft = 2048;
            desc.in_place = false;
            desc.batch = 4;
            desc.stride_in = 1;
            desc.stride_out = 1;
            plan_batch_2048_ = MakeFftPlan(desc);
        }
    }

    std::vector<float> test_data_512_;
    std::vector<float> test_data_1024_;
    std::vector<float> test_data_2048_;

    std::unique_ptr<IFftPlan> plan_512_;
    std::unique_ptr<IFftPlan> plan_1024_;
    std::unique_ptr<IFftPlan> plan_2048_;
    std::unique_ptr<IFftPlan> plan_batch_512_;
    std::unique_ptr<IFftPlan> plan_batch_1024_;
    std::unique_ptr<IFftPlan> plan_batch_2048_;
};

// 단일 FFT 512
BENCHMARK_DEFINE_F(FFTMicroBenchmark, SingleFFT512)(benchmark::State& state) {
    std::vector<std::complex<float>> output(257); // 512/2 + 1

    for (auto _ : state) {
        plan_512_->forward(test_data_512_.data(), output.data());
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * 512 * sizeof(float));
}

// 단일 FFT 1024
BENCHMARK_DEFINE_F(FFTMicroBenchmark, SingleFFT1024)(benchmark::State& state) {
    std::vector<std::complex<float>> output(513); // 1024/2 + 1

    for (auto _ : state) {
        plan_1024_->forward(test_data_1024_.data(), output.data());
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * 1024 * sizeof(float));
}

// 단일 FFT 2048
BENCHMARK_DEFINE_F(FFTMicroBenchmark, SingleFFT2048)(benchmark::State& state) {
    std::vector<std::complex<float>> output(1025); // 2048/2 + 1

    for (auto _ : state) {
        plan_2048_->forward(test_data_2048_.data(), output.data());
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * 2048 * sizeof(float));
}

// 배치 FFT 512 (배치 크기 4)
BENCHMARK_DEFINE_F(FFTMicroBenchmark, BatchFFT512)(benchmark::State& state) {
    std::vector<float> batch_input(512 * 4);
    std::vector<std::complex<float>> batch_output(257 * 4);

    // 배치 입력 데이터 준비
    for (int b = 0; b < 4; ++b) {
        std::copy(test_data_512_.begin(), test_data_512_.end(),
                  batch_input.begin() + b * 512);
    }

    for (auto _ : state) {
        plan_batch_512_->forward(batch_input.data(), batch_output.data(), 4);
        benchmark::DoNotOptimize(batch_output.data());
    }

    state.SetItemsProcessed(state.iterations() * 4);
    state.SetBytesProcessed(state.iterations() * 4 * 512 * sizeof(float));
}

// 배치 FFT 1024 (배치 크기 4)
BENCHMARK_DEFINE_F(FFTMicroBenchmark, BatchFFT1024)(benchmark::State& state) {
    std::vector<float> batch_input(1024 * 4);
    std::vector<std::complex<float>> batch_output(513 * 4);

    // 배치 입력 데이터 준비
    for (int b = 0; b < 4; ++b) {
        std::copy(test_data_1024_.begin(), test_data_1024_.end(),
                  batch_input.begin() + b * 1024);
    }

    for (auto _ : state) {
        plan_batch_1024_->forward(batch_input.data(), batch_output.data(), 4);
        benchmark::DoNotOptimize(batch_output.data());
    }

    state.SetItemsProcessed(state.iterations() * 4);
    state.SetBytesProcessed(state.iterations() * 4 * 1024 * sizeof(float));
}

// 배치 FFT 2048 (배치 크기 4)
BENCHMARK_DEFINE_F(FFTMicroBenchmark, BatchFFT2048)(benchmark::State& state) {
    std::vector<float> batch_input(2048 * 4);
    std::vector<std::complex<float>> batch_output(1025 * 4);

    // 배치 입력 데이터 준비
    for (int b = 0; b < 4; ++b) {
        std::copy(test_data_2048_.begin(), test_data_2048_.end(),
                  batch_input.begin() + b * 2048);
    }

    for (auto _ : state) {
        plan_batch_2048_->forward(batch_input.data(), batch_output.data(), 4);
        benchmark::DoNotOptimize(batch_output.data());
    }

    state.SetItemsProcessed(state.iterations() * 4);
    state.SetBytesProcessed(state.iterations() * 4 * 2048 * sizeof(float));
}

// 벤치마크 등록
BENCHMARK_REGISTER_F(FFTMicroBenchmark, SingleFFT512)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000);

BENCHMARK_REGISTER_F(FFTMicroBenchmark, SingleFFT1024)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(5000);

BENCHMARK_REGISTER_F(FFTMicroBenchmark, SingleFFT2048)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(2000);

BENCHMARK_REGISTER_F(FFTMicroBenchmark, BatchFFT512)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(5000);

BENCHMARK_REGISTER_F(FFTMicroBenchmark, BatchFFT1024)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(2000);

BENCHMARK_REGISTER_F(FFTMicroBenchmark, BatchFFT2048)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000);

// 커스텀 리포터: FFT 성능 로깅
class FFTPerformanceReporter : public benchmark::ConsoleReporter {
public:
    FFTPerformanceReporter() = default;

    void ReportRuns(const std::vector<Run>& reports) override {
        ConsoleReporter::ReportRuns(reports);

        // FFT 성능 요약
        std::cout << "\n=== FFT Performance Summary ===\n";
        std::cout << std::fixed << std::setprecision(2);

        for (const auto& report : reports) {
            if (report.benchmark_name().find("FFT") != std::string::npos) {
                double time_us = report.GetAdjustedRealTime();

                std::cout << report.benchmark_name() << ": "
                         << time_us << " us\n";
            }
        }
        std::cout << "===============================\n";
    }
};

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    // 커스텀 리포터 설정
    FFTPerformanceReporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);

    return 0;
}
