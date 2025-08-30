#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>

#include "dsp/ola/OLAAccumulator.h"
#include "dsp/window/WindowLUT.h"
#include "dsp/frame/FrameQueue.h"
#include "dsp/fft/api/fft_api.h"

using namespace dsp;
using namespace std::chrono;

class PerformanceBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
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

    // 벤치마크 헬퍼 함수
    template<typename Func>
    double measureTime(Func&& func, int iterations = 1000) {
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end - start);
        return static_cast<double>(duration.count()) / iterations;  // 평균 마이크로초
    }

    std::vector<float> test_data_1k_;
    std::vector<float> test_data_4k_;
    std::vector<float> test_data_16k_;
};

// 피드백 반영: 성능 벤치마크 및 검증
TEST_F(PerformanceBenchmark, OLAAccumulatorPerformance) {
    std::cout << "\n=== OLA Accumulator 성능 벤치마크 ===" << std::endl;

    struct TestCase {
        int frame_size;
        int hop_size;
        int channels;
        const char* description;
    };

    std::vector<TestCase> test_cases = {
        {512, 256, 1, "512/256 모노"},
        {1024, 512, 1, "1024/512 모노"},
        {2048, 1024, 1, "2048/1024 모노"},
        {1024, 512, 2, "1024/512 스테레오"},
    };

    for (const auto& tc : test_cases) {
        OLAConfig config;
        config.sample_rate = 48000;
        config.frame_size = tc.frame_size;
        config.hop_size = tc.hop_size;
        config.channels = tc.channels;
        config.center = true;
        config.apply_window_inside = true;

        OLAAccumulator ola(config);

        // 윈도우 설정
        WindowLUT& lut = WindowLUT::getInstance();
        auto window = lut.GetWindowSafe(WindowType::HANN, tc.frame_size);
        ola.set_window(window.get(), tc.frame_size);

        // 테스트 프레임 준비
        std::vector<float> test_frame(tc.frame_size * tc.channels);
        for (int i = 0; i < tc.frame_size * tc.channels; ++i) {
            test_frame[i] = test_data_1k_[i % test_data_1k_.size()];
        }

        // push_frame 성능 측정
        double push_time = measureTime([&]() {
            ola.push_frame(0, test_frame.data());
        }, 10000);

        // pull 성능 측정
        std::vector<float> output(tc.hop_size * tc.channels);
        double pull_time = measureTime([&]() {
            ola.pull(output.data(), tc.hop_size);
        }, 10000);

        // 처리량 계산 (frames per second)
        double push_fps = 1000000.0 / push_time;  // 마이크로초 -> fps
        double pull_fps = 1000000.0 / pull_time;

        std::cout << std::fixed << std::setprecision(1);
        std::cout << tc.description << ":\n";
        std::cout << "  Push: " << push_time << "μs (" << push_fps << " fps)\n";
        std::cout << "  Pull: " << pull_time << "μs (" << pull_fps << " fps)\n";

        // 성능 기준 검증 (피드백 목표)
        EXPECT_LT(push_time, 500.0) << "Push performance below target for " << tc.description;
        EXPECT_LT(pull_time, 700.0) << "Pull performance below target for " << tc.description;
        EXPECT_GT(push_fps, 2000.0) << "Push FPS below target for " << tc.description;
    }
}

TEST_F(PerformanceBenchmark, WindowLUTPerformance) {
    std::cout << "\n=== WindowLUT 성능 벤치마크 ===" << std::endl;

    WindowLUT& lut = WindowLUT::getInstance();

    struct TestCase {
        WindowType type;
        size_t size;
        const char* description;
    };

    std::vector<TestCase> test_cases = {
        {WindowType::HANN, 512, "Hann 512"},
        {WindowType::HANN, 1024, "Hann 1024"},
        {WindowType::HANN, 2048, "Hann 2048"},
        {WindowType::HANN, 4096, "Hann 4096"},
        {WindowType::HAMMING, 1024, "Hamming 1024"},
        {WindowType::BLACKMAN, 1024, "Blackman 1024"},
    };

    for (const auto& tc : test_cases) {
        // 첫 번째 호출 (생성 시간)
        lut.clearCache(true);  // 캐시 초기화
        double creation_time = measureTime([&]() {
            lut.GetWindow(tc.type, tc.size);
        }, 100);

        // 두 번째 호출 (캐시 히트)
        double cache_time = measureTime([&]() {
            lut.GetWindow(tc.type, tc.size);
        }, 10000);

        // 안전한 API 성능
        double safe_time = measureTime([&]() {
            lut.GetWindowSafe(tc.type, tc.size);
        }, 10000);

        std::cout << tc.description << ":\n";
        std::cout << "  생성: " << creation_time << "μs\n";
        std::cout << "  캐시: " << cache_time << "μs\n";
        std::cout << "  안전: " << safe_time << "μs\n";

        // 성능 기준 검증
        EXPECT_LT(cache_time, 0.5) << "Cache hit too slow for " << tc.description;
        EXPECT_LT(safe_time, 1.0) << "Safe API too slow for " << tc.description;

        // 처리량 계산
        double cache_fps = 1000000.0 / cache_time;
        EXPECT_GT(cache_fps, 5000.0) << "Cache FPS below target for " << tc.description;
    }
}

TEST_F(PerformanceBenchmark, FFTPerformance) {
    std::cout << "\n=== FFT 성능 벤치마크 ===" << std::endl;

    using namespace dsp::fft;

    struct TestCase {
        int nfft;
        int batch;
        const char* description;
    };

    std::vector<TestCase> test_cases = {
        {512, 1, "512 단일"},
        {1024, 1, "1024 단일"},
        {2048, 1, "2048 단일"},
        {1024, 4, "1024 배치x4"},
        {1024, 8, "1024 배치x8"},
    };

    for (const auto& tc : test_cases) {
        FftPlanDesc desc;
        desc.domain = FftDomain::Real;
        desc.nfft = tc.nfft;
        desc.in_place = false;  // 명시적으로 false 설정
        desc.batch = tc.batch;
        desc.stride_in = 1;
        desc.stride_out = 1;

        auto plan = MakeFftPlan(desc);

        // 테스트 데이터 준비
        std::vector<float> input(tc.batch * tc.nfft);
        std::vector<std::complex<float>> output(tc.batch * (tc.nfft / 2 + 1));
        std::vector<float> reconstructed(tc.batch * tc.nfft);

        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = test_data_16k_[i % test_data_16k_.size()];
        }

        // Forward FFT 성능
        double forward_time = measureTime([&]() {
            plan->forward(input.data(), output.data(), tc.batch);
        }, 1000);

        // Inverse FFT 성능
        double inverse_time = measureTime([&]() {
            plan->inverse(output.data(), reconstructed.data(), tc.batch);
        }, 1000);

        // 처리량 계산
        double forward_fps = 1000000.0 / forward_time;
        double inverse_fps = 1000000.0 / inverse_time;

        std::cout << tc.description << ":\n";
        std::cout << "  Forward: " << forward_time << "μs (" << forward_fps << " fps)\n";
        std::cout << "  Inverse: " << inverse_time << "μs (" << inverse_fps << " fps)\n";

        // 성능 기준 검증
        if (tc.nfft <= 1024) {
            EXPECT_GT(forward_fps, 1500.0) << "Forward FPS below target for " << tc.description;
            EXPECT_GT(inverse_fps, 1500.0) << "Inverse FPS below target for " << tc.description;
        }

        // 배치 효율성 검증
        if (tc.batch > 1) {
            std::cout << "  배치 효율성: " << (tc.batch * 1000000.0 / forward_time) << " total fps\n";
        }
    }
}

TEST_F(PerformanceBenchmark, FrameQueuePerformance) {
    std::cout << "\n=== FrameQueue 성능 벤치마크 ===" << std::endl;

    struct TestCase {
        size_t input_size;
        size_t frame_size;
        size_t hop_size;
        bool center;
        const char* description;
    };

    std::vector<TestCase> test_cases = {
        {4096, 512, 256, false, "4k→512/256 no-center"},
        {4096, 512, 256, true, "4k→512/256 center"},
        {16384, 1024, 512, true, "16k→1024/512 center"},
        {16384, 2048, 1024, true, "16k→2048/1024 center"},
    };

    for (const auto& tc : test_cases) {
        const std::vector<float>& input_data = (tc.input_size <= 4096) ? test_data_4k_ : test_data_16k_;

        // FrameQueue 생성 성능
        double creation_time = measureTime([&]() {
            FrameQueue fq(input_data.data(), tc.input_size, tc.frame_size, tc.hop_size, tc.center);
        }, 100);

        // 프레임 접근 성능
        FrameQueue fq(input_data.data(), tc.input_size, tc.frame_size, tc.hop_size, tc.center);
        size_t num_frames = fq.getNumFrames();

        double access_time = measureTime([&]() {
            for (size_t i = 0; i < num_frames; ++i) {
                volatile const float* frame = fq.getFrame(i);
                (void)frame;  // 최적화 방지
            }
        }, 1000);

        // 프레임 복사 성능
        std::vector<float> frame_buffer(tc.frame_size);
        double copy_time = measureTime([&]() {
            for (size_t i = 0; i < num_frames; ++i) {
                fq.copyFrame(i, frame_buffer.data());
            }
        }, 100);

        std::cout << tc.description << " (" << num_frames << " frames):\n";
        std::cout << "  생성: " << creation_time << "μs\n";
        std::cout << "  접근: " << access_time / num_frames << "μs/frame\n";
        std::cout << "  복사: " << copy_time / num_frames << "μs/frame\n";

        // 성능 기준 검증
        EXPECT_LT(access_time / num_frames, 1.0) << "Frame access too slow for " << tc.description;
        EXPECT_LT(copy_time / num_frames, 10.0) << "Frame copy too slow for " << tc.description;
    }
}

// 통합 파이프라인 성능 테스트
TEST_F(PerformanceBenchmark, IntegratedPipelinePerformance) {
    std::cout << "\n=== 통합 파이프라인 성능 ===" << std::endl;

    const size_t input_length = 16384;
    const size_t frame_size = 1024;
    const size_t hop_size = 512;

    // 전체 STFT 분석/합성 파이프라인 성능 측정
    double pipeline_time = measureTime([&]() {
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
        desc.batch = 1;
        auto fft_plan = MakeFftPlan(desc);

        // 5. 프레임별 처리
        std::vector<std::complex<float>> spectrum(frame_size / 2 + 1);
        std::vector<float> processed_frame(frame_size);

        for (size_t i = 0; i < frames.getNumFrames(); ++i) {
            const float* frame = frames.getFrame(i);

            // FFT 분석
            fft_plan->forward(frame, spectrum.data());

            // 간단한 처리 (여기서는 그대로 통과)

            // IFFT 합성
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
    }, 100);

    // 실시간 성능 계산
    double audio_duration = static_cast<double>(input_length) / 48000.0;  // 초
    double realtime_factor = (pipeline_time / 1000000.0) / audio_duration;

    std::cout << "통합 파이프라인 (" << input_length << " 샘플):\n";
    std::cout << "  처리 시간: " << pipeline_time << "μs\n";
    std::cout << "  오디오 길이: " << audio_duration * 1000 << "ms\n";
    std::cout << "  실시간 배수: " << realtime_factor << "x\n";

    // 실시간 처리 가능 여부 검증
    EXPECT_LT(realtime_factor, 0.1) << "Pipeline not suitable for real-time processing";

    if (realtime_factor < 0.01) {
        std::cout << "  ✓ 초고속 처리 가능 (100x 실시간)" << std::endl;
    } else if (realtime_factor < 0.1) {
        std::cout << "  ✓ 고속 처리 가능 (10x 실시간)" << std::endl;
    } else if (realtime_factor < 1.0) {
        std::cout << "  ✓ 실시간 처리 가능" << std::endl;
    } else {
        std::cout << "  ⚠ 실시간 처리 어려움" << std::endl;
    }
}