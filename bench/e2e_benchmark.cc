#include <benchmark/benchmark.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include "dsp/ola/OLAAccumulator.h"
#include "dsp/window/WindowLUT.h"
#include "dsp/frame/framer.h"
#include "dsp/fft/api/fft_api.h"
#include "io/wav.h"

using namespace dsp;
using namespace dsp::fft;

// E2E 벤치마크
class E2EBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // 테스트 데이터 생성
        generateTestData();
        setupPipeline();
    }

    void generateTestData() {
        // 테스트 WAV 데이터 생성 (1초, 48kHz, 모노)
        sample_rate_ = 48000;
        duration_sec_ = 1.0f;
        total_samples_ = static_cast<size_t>(sample_rate_ * duration_sec_);

        // 실제적인 오디오 신호 생성
        test_input_.resize(total_samples_);
        for (size_t i = 0; i < total_samples_; ++i) {
            double t = static_cast<double>(i) / sample_rate_;
            test_input_[i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * t) +
                           0.3f * std::sin(2.0 * M_PI * 880.0 * t) +
                           0.2f * std::sin(2.0 * M_PI * 1320.0 * t);
        }
    }

    void setupPipeline() {
        // 파이프라인 파라미터
        frame_size_ = 1024;
        hop_size_ = 512;

        // 프레이머 설정
        framer_.set_params(frame_size_, hop_size_, 1, BoundaryMode::ZERO_PAD);

        // 윈도우 설정
        WindowLUT& lut = WindowLUT::getInstance();
        auto safe_window = lut.GetWindowSafe(WindowType::HANN, frame_size_);
        window_ = safe_window.get();

        // OLA 설정
        OLAConfig ola_config;
        ola_config.sample_rate = sample_rate_;
        ola_config.frame_size = frame_size_;
        ola_config.hop_size = hop_size_;
        ola_config.channels = 1;
        ola_config.center = true;
        ola_config.apply_window_inside = true;
        ola_config.gain = 1.0f;

        ola_ = std::make_unique<OLAAccumulator>(ola_config);
        ola_->set_window(window_, frame_size_);

        // FFT 설정
        FftPlanDesc fft_desc;
        fft_desc.domain = FftDomain::Real;
        fft_desc.nfft = frame_size_;
        fft_desc.in_place = false;
        fft_desc.batch = 1;
        fft_desc.stride_in = 1;
        fft_desc.stride_out = 1;

        fft_plan_ = MakeFftPlan(fft_desc);
    }

    // SNR 계산 함수
    double calculateSNR(const std::vector<float>& original, const std::vector<float>& processed) {
        if (original.size() != processed.size()) {
            return -std::numeric_limits<double>::infinity();
        }

        double signal_power = 0.0;
        double noise_power = 0.0;

        for (size_t i = 0; i < original.size(); ++i) {
            double diff = original[i] - processed[i];
            signal_power += original[i] * original[i];
            noise_power += diff * diff;
        }

        if (noise_power < 1e-12) {
            return std::numeric_limits<double>::infinity();
        }

        return 10.0 * std::log10(signal_power / noise_power);
    }

    // 지연 측정 함수 (크로스 코릴레이션 기반)
    double calculateDelay(const std::vector<float>& original, const std::vector<float>& processed) {
        const size_t max_delay = 1024; // 최대 1024 샘플 지연
        double max_corr = -1.0;
        size_t best_delay = 0;

        for (size_t delay = 0; delay < max_delay; ++delay) {
            double corr = 0.0;
            size_t overlap = std::min(original.size() - delay, processed.size());

            for (size_t i = 0; i < overlap; ++i) {
                corr += original[i] * processed[i + delay];
            }

            if (corr > max_corr) {
                max_corr = corr;
                best_delay = delay;
            }
        }

        return static_cast<double>(best_delay) / sample_rate_ * 1000.0; // ms
    }

    size_t sample_rate_;
    float duration_sec_;
    size_t total_samples_;
    size_t frame_size_;
    size_t hop_size_;

    std::vector<float> test_input_;
    const float* window_;

    Framer framer_;
    std::unique_ptr<OLAAccumulator> ola_;
    std::unique_ptr<IFftPlan> fft_plan_;
};

// E2E 파이프라인 벤치마크
BENCHMARK_DEFINE_F(E2EBenchmark, FullPipeline)(benchmark::State& state) {
    std::vector<float> output(total_samples_);
    size_t output_samples = 0;

    for (auto _ : state) {
        // 1. WAV → Frame
        framer_.push(test_input_.data(), total_samples_);

        // 2. Frame → Window → FFT → iFFT → OLA
        std::vector<float> frame_buffer(frame_size_);
        std::vector<std::complex<float>> spectrum(frame_size_ / 2 + 1);
        std::vector<float> processed_frame(frame_size_);

        size_t frame_count = 0;
        while (framer_.pop(frame_buffer.data())) {
            // 윈도우 적용 (외부)
            for (size_t i = 0; i < frame_size_; ++i) {
                processed_frame[i] = frame_buffer[i] * window_[i];
            }

            // FFT
            fft_plan_->forward(processed_frame.data(), spectrum.data());

            // 간단한 처리 (예: 게인 1.0)
            // 실제로는 필터링이나 다른 처리를 할 수 있음

            // iFFT
            fft_plan_->inverse(spectrum.data(), processed_frame.data());

            // OLA 누적
            ola_->push_frame(frame_count++, processed_frame.data());
        }

        // 3. OLA → WAV
        output_samples = 0;
        while (output_samples < total_samples_) {
            int samples = ola_->pull(output.data() + output_samples,
                                   total_samples_ - output_samples);
            if (samples == 0) break;
            output_samples += samples;
        }

        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(state.iterations() * total_samples_);
    state.SetBytesProcessed(state.iterations() * total_samples_ * sizeof(float));
}

// E2E 품질 측정 벤치마크
BENCHMARK_DEFINE_F(E2EBenchmark, QualityMetrics)(benchmark::State& state) {
    std::vector<double> snr_values;
    std::vector<double> delay_values;

    for (auto _ : state) {
        // 파이프라인 실행
        framer_.push(test_input_.data(), total_samples_);

        std::vector<float> frame_buffer(frame_size_);
        std::vector<std::complex<float>> spectrum(frame_size_ / 2 + 1);
        std::vector<float> processed_frame(frame_size_);
        std::vector<float> output(total_samples_);

        size_t frame_count = 0;
        size_t output_samples = 0;

        while (framer_.pop(frame_buffer.data())) {
            // 윈도우 적용
            for (size_t i = 0; i < frame_size_; ++i) {
                processed_frame[i] = frame_buffer[i] * window_[i];
            }

            // FFT
            fft_plan_->forward(processed_frame.data(), spectrum.data());

            // iFFT
            fft_plan_->inverse(spectrum.data(), processed_frame.data());

            // OLA
            ola_->push_frame(frame_count++, processed_frame.data());
        }

        // 출력 수집
        while (output_samples < total_samples_) {
            int samples = ola_->pull(output.data() + output_samples,
                                   total_samples_ - output_samples);
            if (samples == 0) break;
            output_samples += samples;
        }

        // 품질 측정
        double snr = calculateSNR(test_input_, output);
        double delay = calculateDelay(test_input_, output);

        snr_values.push_back(snr);
        delay_values.push_back(delay);

        benchmark::DoNotOptimize(snr);
        benchmark::DoNotOptimize(delay);
    }

    // 평균 SNR과 지연 계산
    double avg_snr = 0.0;
    double avg_delay = 0.0;
    for (size_t i = 0; i < snr_values.size(); ++i) {
        avg_snr += snr_values[i];
        avg_delay += delay_values[i];
    }
    avg_snr /= snr_values.size();
    avg_delay /= delay_values.size();

    state.SetItemsProcessed(state.iterations());
    state.counters["avg_snr_dB"] = avg_snr;
    state.counters["avg_delay_ms"] = avg_delay;
}

// FFT 성능 집중 벤치마크 (1024 포인트)
BENCHMARK_DEFINE_F(E2EBenchmark, FFT1024Performance)(benchmark::State& state) {
    std::vector<float> frame_buffer(frame_size_);
    std::vector<std::complex<float>> spectrum(frame_size_ / 2 + 1);
    std::vector<float> processed_frame(frame_size_);

    // 테스트 프레임 준비
    std::copy(test_input_.begin(), test_input_.begin() + frame_size_, frame_buffer.begin());
    for (size_t i = 0; i < frame_size_; ++i) {
        processed_frame[i] = frame_buffer[i] * window_[i];
    }

    for (auto _ : state) {
        // FFT
        fft_plan_->forward(processed_frame.data(), spectrum.data());

        // iFFT
        fft_plan_->inverse(spectrum.data(), processed_frame.data());

        benchmark::DoNotOptimize(processed_frame.data());
    }

    state.SetItemsProcessed(state.iterations() * frame_size_);
    state.SetBytesProcessed(state.iterations() * frame_size_ * sizeof(float));
}

// 벤치마크 등록
BENCHMARK_REGISTER_F(E2EBenchmark, FullPipeline)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

BENCHMARK_REGISTER_F(E2EBenchmark, QualityMetrics)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK_REGISTER_F(E2EBenchmark, FFT1024Performance)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000);

// 커스텀 리포터: E2E 성능 및 품질 로깅
class E2EPerformanceReporter : public benchmark::ConsoleReporter {
public:
    E2EPerformanceReporter() = default;

    void ReportRuns(const std::vector<Run>& reports) override {
        ConsoleReporter::ReportRuns(reports);

        // E2E 성능 요약
        std::cout << "\n=== E2E Performance Summary ===\n";
        std::cout << std::fixed << std::setprecision(2);

        for (const auto& report : reports) {
            if (report.benchmark_name().find("FullPipeline") != std::string::npos) {
                double time_ms = report.GetAdjustedRealTime();
                double throughput = (48000.0 / time_ms) * 1000.0; // real-time factor

                std::cout << "Full Pipeline: " << time_ms << " ms, "
                         << throughput << "x real-time\n";
            }

            if (report.benchmark_name().find("FFT1024Performance") != std::string::npos) {
                double time_us = report.GetAdjustedRealTime();
                std::cout << "1024-pt FFT: " << time_us << " us\n";

                // AVX2/NEON 목표 확인
                if (time_us < 5.0) {
                    std::cout << "  ✓ AVX2 target met (< 5µs)\n";
                } else if (time_us < 10.0) {
                    std::cout << "  ✓ NEON target met (< 10µs)\n";
                } else {
                    std::cout << "  ✗ Performance target not met\n";
                }
            }

            if (report.benchmark_name().find("QualityMetrics") != std::string::npos) {
                auto snr_it = report.counters.find("avg_snr_dB");
                auto delay_it = report.counters.find("avg_delay_ms");

                if (snr_it != report.counters.end()) {
                    double snr = snr_it->second;
                    std::cout << "Average SNR: " << snr << " dB";

                    if (snr > 60.0) {
                        std::cout << " ✓ SNR target met (> 60 dB)\n";
                    } else {
                        std::cout << " ✗ SNR target not met\n";
                    }
                }

                if (delay_it != report.counters.end()) {
                    double delay = delay_it->second;
                    std::cout << "Average Delay: " << delay << " ms\n";
                }
            }
        }
        std::cout << "===============================\n";
    }
};

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    // 커스텀 리포터 설정
    E2EPerformanceReporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);

    return 0;
}
