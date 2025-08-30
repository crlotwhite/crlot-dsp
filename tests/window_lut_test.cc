#include <gtest/gtest.h>
#include "dsp/window/WindowLUT.h"
#include <cmath>
#include <vector>
#include <thread>
#include <chrono>

using namespace dsp;

class WindowLUTTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 각 테스트 전에 캐시 초기화
        WindowLUT::getInstance().clearCache();
    }

    void TearDown() override {
        // 테스트 후 캐시 초기화
        WindowLUT::getInstance().clearCache();
    }

    // 참조 Hann 윈도우 생성 (검증용)
    std::vector<float> generateReferenceHann(size_t N) {
        std::vector<float> window(N);
        if (N == 1) {
            window[0] = 1.0f;
            return window;
        }

        const double pi = M_PI;
        const double factor = 2.0 * pi / static_cast<double>(N - 1);

        for (size_t i = 0; i < N; ++i) {
            double angle = factor * static_cast<double>(i);
            window[i] = static_cast<float>(0.5 * (1.0 - std::cos(angle)));
        }
        return window;
    }

    // 참조 Hamming 윈도우 생성 (검증용)
    std::vector<float> generateReferenceHamming(size_t N) {
        std::vector<float> window(N);
        if (N == 1) {
            window[0] = 1.0f;
            return window;
        }

        const double pi = M_PI;
        const double factor = 2.0 * pi / static_cast<double>(N - 1);
        const double alpha = 0.54;
        const double beta = 0.46;

        for (size_t i = 0; i < N; ++i) {
            double angle = factor * static_cast<double>(i);
            window[i] = static_cast<float>(alpha - beta * std::cos(angle));
        }
        return window;
    }

    // 참조 Blackman 윈도우 생성 (검증용)
    std::vector<float> generateReferenceBlackman(size_t N) {
        std::vector<float> window(N);
        if (N == 1) {
            window[0] = 1.0f;
            return window;
        }

        const double pi = M_PI;
        const double factor = 2.0 * pi / static_cast<double>(N - 1);
        const double a0 = 0.42;
        const double a1 = 0.5;
        const double a2 = 0.08;

        for (size_t i = 0; i < N; ++i) {
            double angle = factor * static_cast<double>(i);
            double cos1 = std::cos(angle);
            double cos2 = std::cos(2.0 * angle);
            window[i] = static_cast<float>(a0 - a1 * cos1 + a2 * cos2);
        }
        return window;
    }
};

// 기본 API 테스트
TEST_F(WindowLUTTest, BasicAPI) {
    auto& lut = WindowLUT::getInstance();

    // 유효한 윈도우 요청
    const float* hann = lut.GetWindow(WindowType::HANN, 512);
    ASSERT_NE(hann, nullptr);

    const float* hamming = lut.GetWindow(WindowType::HAMMING, 1024);
    ASSERT_NE(hamming, nullptr);

    const float* blackman = lut.GetWindow(WindowType::BLACKMAN, 256);
    ASSERT_NE(blackman, nullptr);

    // 캐시 크기 확인
    EXPECT_EQ(lut.getCacheSize(), 3);
}

// 잘못된 입력 테스트
TEST_F(WindowLUTTest, InvalidInput) {
    auto& lut = WindowLUT::getInstance();

    // 크기가 0인 경우
    EXPECT_THROW(lut.GetWindow(WindowType::HANN, 0), std::invalid_argument);

    // 구현되지 않은 윈도우 타입
    EXPECT_THROW(lut.GetWindow(WindowType::BLACKMAN_HARRIS, 512), std::invalid_argument);
}

// Hann 윈도우 정확성 테스트
TEST_F(WindowLUTTest, HannWindowAccuracy) {
    auto& lut = WindowLUT::getInstance();

    // 다양한 크기에 대해 테스트
    std::vector<size_t> sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1000, 1023};

    for (size_t N : sizes) {
        const float* window = lut.GetWindow(WindowType::HANN, N);
        auto reference = generateReferenceHann(N);

        // RMS 오차 계산
        double rms_error = WindowLUT::calculateRMSError(window, reference.data(), N);
        EXPECT_LT(rms_error, 1e-6) << "Hann window RMS error too large for N=" << N;

        // 첫 번째와 마지막 값 확인 (N > 1인 경우)
        if (N > 1) {
            EXPECT_NEAR(window[0], 0.0f, 1e-6f) << "Hann window first value incorrect for N=" << N;
            EXPECT_NEAR(window[N-1], 0.0f, 1e-6f) << "Hann window last value incorrect for N=" << N;
        }
    }
}

// Hamming 윈도우 정확성 테스트
TEST_F(WindowLUTTest, HammingWindowAccuracy) {
    auto& lut = WindowLUT::getInstance();

    std::vector<size_t> sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1000, 1023};

    for (size_t N : sizes) {
        const float* window = lut.GetWindow(WindowType::HAMMING, N);
        auto reference = generateReferenceHamming(N);

        double rms_error = WindowLUT::calculateRMSError(window, reference.data(), N);
        EXPECT_LT(rms_error, 1e-6) << "Hamming window RMS error too large for N=" << N;

        // Hamming 윈도우의 첫 번째와 마지막 값 확인
        if (N > 1) {
            EXPECT_NEAR(window[0], 0.08f, 1e-6f) << "Hamming window first value incorrect for N=" << N;
            EXPECT_NEAR(window[N-1], 0.08f, 1e-6f) << "Hamming window last value incorrect for N=" << N;
        }
    }
}

// Blackman 윈도우 정확성 테스트
TEST_F(WindowLUTTest, BlackmanWindowAccuracy) {
    auto& lut = WindowLUT::getInstance();

    std::vector<size_t> sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1000, 1023};

    for (size_t N : sizes) {
        const float* window = lut.GetWindow(WindowType::BLACKMAN, N);
        auto reference = generateReferenceBlackman(N);

        double rms_error = WindowLUT::calculateRMSError(window, reference.data(), N);
        EXPECT_LT(rms_error, 1e-6) << "Blackman window RMS error too large for N=" << N;

        // Blackman 윈도우의 첫 번째와 마지막 값 확인 (거의 0)
        if (N > 1) {
            EXPECT_NEAR(window[0], 0.0f, 1e-6f) << "Blackman window first value incorrect for N=" << N;
            EXPECT_NEAR(window[N-1], 0.0f, 1e-6f) << "Blackman window last value incorrect for N=" << N;
        }
    }
}

// 정규화 특성 테스트 (합과 제곱합)
TEST_F(WindowLUTTest, NormalizationProperties) {
    auto& lut = WindowLUT::getInstance();

    std::vector<size_t> sizes = {16, 32, 64, 128, 256, 512, 1024};

    for (size_t N : sizes) {
        // Hann 윈도우
        const float* hann = lut.GetWindow(WindowType::HANN, N);
        double hann_sum = WindowLUT::calculateSum(hann, N);
        double hann_sum_sq = WindowLUT::calculateSumOfSquares(hann, N);

        // Hann 윈도우의 합은 대략 N/2 (더 관대한 허용 오차)
        EXPECT_NEAR(hann_sum, N / 2.0, N * 0.05) << "Hann window sum incorrect for N=" << N;

        // Hamming 윈도우
        const float* hamming = lut.GetWindow(WindowType::HAMMING, N);
        double hamming_sum = WindowLUT::calculateSum(hamming, N);
        double hamming_sum_sq = WindowLUT::calculateSumOfSquares(hamming, N);

        // Hamming 윈도우의 합은 대략 0.54*N (더 관대한 허용 오차)
        EXPECT_NEAR(hamming_sum, 0.54 * N, N * 0.05) << "Hamming window sum incorrect for N=" << N;

        // Blackman 윈도우
        const float* blackman = lut.GetWindow(WindowType::BLACKMAN, N);
        double blackman_sum = WindowLUT::calculateSum(blackman, N);
        double blackman_sum_sq = WindowLUT::calculateSumOfSquares(blackman, N);

        // Blackman 윈도우의 합은 대략 0.42*N (더 관대한 허용 오차)
        EXPECT_NEAR(blackman_sum, 0.42 * N, N * 0.05) << "Blackman window sum incorrect for N=" << N;

        // 제곱합은 양수여야 함
        EXPECT_GT(hann_sum_sq, 0.0);
        EXPECT_GT(hamming_sum_sq, 0.0);
        EXPECT_GT(blackman_sum_sq, 0.0);
    }
}

// 캐시 일관성 테스트
TEST_F(WindowLUTTest, CacheConsistency) {
    auto& lut = WindowLUT::getInstance();

    // 같은 윈도우를 여러 번 요청
    const float* hann1 = lut.GetWindow(WindowType::HANN, 512);
    const float* hann2 = lut.GetWindow(WindowType::HANN, 512);
    const float* hann3 = lut.GetWindow(WindowType::HANN, 512);

    // 같은 포인터를 반환해야 함 (캐시됨)
    EXPECT_EQ(hann1, hann2);
    EXPECT_EQ(hann2, hann3);

    // 캐시 크기는 1이어야 함
    EXPECT_EQ(lut.getCacheSize(), 1);

    // 다른 크기 요청
    const float* hann_256 = lut.GetWindow(WindowType::HANN, 256);
    EXPECT_NE(hann1, hann_256);
    EXPECT_EQ(lut.getCacheSize(), 2);

    // 다른 타입 요청
    const float* hamming_512 = lut.GetWindow(WindowType::HAMMING, 512);
    EXPECT_NE(hann1, hamming_512);
    EXPECT_EQ(lut.getCacheSize(), 3);
}

// 스레드 안전성 테스트
TEST_F(WindowLUTTest, ThreadSafety) {
    auto& lut = WindowLUT::getInstance();
    const size_t num_threads = 8;
    const size_t num_requests = 100;

    std::vector<std::thread> threads;
    std::vector<std::vector<const float*>> results(num_threads);

    // 여러 스레드에서 동시에 윈도우 요청
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            results[t].reserve(num_requests);
            for (size_t i = 0; i < num_requests; ++i) {
                WindowType type = static_cast<WindowType>(i % 3); // HANN, HAMMING, BLACKMAN
                size_t N = 64 + (i % 10) * 64; // 64, 128, ..., 704
                const float* window = lut.GetWindow(type, N);
                results[t].push_back(window);

                // 짧은 지연
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }

    // 모든 스레드 완료 대기
    for (auto& thread : threads) {
        thread.join();
    }

    // 결과 검증: 같은 (type, N) 조합은 같은 포인터를 반환해야 함
    for (size_t i = 0; i < num_requests; ++i) {
        WindowType type = static_cast<WindowType>(i % 3);
        size_t N = 64 + (i % 10) * 64;

        const float* first_result = results[0][i];
        for (size_t t = 1; t < num_threads; ++t) {
            EXPECT_EQ(first_result, results[t][i])
                << "Thread safety violation for type=" << static_cast<int>(type) << ", N=" << N;
        }
    }
}

// 메모리 정렬 테스트 (평가 피드백 반영: 32B 정렬)
TEST_F(WindowLUTTest, MemoryAlignment) {
    auto& lut = WindowLUT::getInstance();

    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096};

    for (size_t N : sizes) {
        const float* window = lut.GetWindow(WindowType::HANN, N);

        // 32-byte 정렬 확인 (AVX2/NEON 벡터화 지원)
        uintptr_t addr = reinterpret_cast<uintptr_t>(window);
        EXPECT_EQ(addr % 32, 0) << "Window data not 32-byte aligned for N=" << N;

        // 성능 메트릭 로깅
        std::cout << "Alignment check - N=" << N
                  << ", addr=0x" << std::hex << addr << std::dec
                  << ", aligned=" << (addr % 32 == 0 ? "YES" : "NO") << std::endl;
    }
}

// 다양한 윈도우 타입의 정렬 보장 테스트
TEST_F(WindowLUTTest, AlignmentGuaranteeAllTypes) {
    auto& lut = WindowLUT::getInstance();

    std::vector<WindowType> types = {
        WindowType::HANN,
        WindowType::HAMMING,
        WindowType::BLACKMAN,
        WindowType::RECT
    };

    std::vector<size_t> sizes = {256, 512, 1024, 2048};

    for (auto type : types) {
        for (size_t N : sizes) {
            const float* window = lut.GetWindow(type, N);
            ASSERT_NE(window, nullptr) << "Window creation failed for type="
                                       << static_cast<int>(type) << ", N=" << N;

            // 32B 정렬 보장
            uintptr_t addr = reinterpret_cast<uintptr_t>(window);
            EXPECT_EQ(addr % 32, 0) << "32-byte alignment violation for type="
                                    << static_cast<int>(type) << ", N=" << N;
        }
    }
}

// 정렬 성능 및 반복 생성/파괴 테스트
TEST_F(WindowLUTTest, AlignmentStressTest) {
    auto& lut = WindowLUT::getInstance();

    // 반복적으로 윈도우 생성/파괴하여 메모리 누수 및 정렬 일관성 확인
    for (int iteration = 0; iteration < 10; ++iteration) {
        lut.clearCache();

        std::vector<size_t> test_sizes = {128, 256, 512, 1024, 2048};
        for (size_t N : test_sizes) {
            const float* window = lut.GetWindow(WindowType::HANN, N);

            // 정렬 확인
            uintptr_t addr = reinterpret_cast<uintptr_t>(window);
            EXPECT_EQ(addr % 32, 0) << "Alignment failure in iteration " << iteration
                                    << " for N=" << N;
        }

        // 캐시 크기 확인
        EXPECT_EQ(lut.getCacheSize(), test_sizes.size())
            << "Cache size mismatch in iteration " << iteration;
    }
}

// 대용량 윈도우 테스트
TEST_F(WindowLUTTest, LargeWindows) {
    auto& lut = WindowLUT::getInstance();

    std::vector<size_t> large_sizes = {8192, 16384, 32768, 65536};

    for (size_t N : large_sizes) {
        const float* window = lut.GetWindow(WindowType::HANN, N);
        ASSERT_NE(window, nullptr) << "Failed to create large window of size " << N;

        // 첫 번째와 마지막 값 확인
        EXPECT_NEAR(window[0], 0.0f, 1e-6f);
        EXPECT_NEAR(window[N-1], 0.0f, 1e-6f);

        // 중간 값이 0이 아님을 확인
        EXPECT_GT(window[N/2], 0.5f);
    }
}

// 캐시 지우기 테스트
TEST_F(WindowLUTTest, CacheClear) {
    auto& lut = WindowLUT::getInstance();

    // 여러 윈도우 생성
    lut.GetWindow(WindowType::HANN, 512);
    lut.GetWindow(WindowType::HAMMING, 1024);
    lut.GetWindow(WindowType::BLACKMAN, 256);

    EXPECT_EQ(lut.getCacheSize(), 3);

    // 캐시 지우기
    lut.clearCache();
    EXPECT_EQ(lut.getCacheSize(), 0);

    // 다시 요청하면 새로 생성됨
    const float* window = lut.GetWindow(WindowType::HANN, 512);
    ASSERT_NE(window, nullptr);
    EXPECT_EQ(lut.getCacheSize(), 1);
}

// 유틸리티 함수 테스트
TEST_F(WindowLUTTest, UtilityFunctions) {
    std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // 합 계산
    double sum = WindowLUT::calculateSum(test_data.data(), test_data.size());
    EXPECT_NEAR(sum, 15.0, 1e-6);

    // 제곱합 계산
    double sum_sq = WindowLUT::calculateSumOfSquares(test_data.data(), test_data.size());
    EXPECT_NEAR(sum_sq, 55.0, 1e-6); // 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55

    // RMS 오차 계산
    std::vector<float> test_data2 = {1.1f, 2.1f, 3.1f, 4.1f, 5.1f};
    double rms_error = WindowLUT::calculateRMSError(test_data.data(), test_data2.data(), test_data.size());
    EXPECT_NEAR(rms_error, 0.1, 1e-6);

    // null 포인터 처리
    EXPECT_EQ(WindowLUT::calculateSum(nullptr, 5), 0.0);
    EXPECT_EQ(WindowLUT::calculateSumOfSquares(nullptr, 5), 0.0);
    EXPECT_EQ(WindowLUT::calculateRMSError(nullptr, test_data.data(), 5), 0.0);
}

// 피드백 반영: 캐시 안전성 테스트 (shared_ptr 기반)
TEST_F(WindowLUTTest, CacheSafetyWithSharedPtr) {
    auto& lut = WindowLUT::getInstance();

    // 안전한 API 사용
    auto safe_window1 = lut.GetWindowSafe(WindowType::HANN, 512);
    auto safe_window2 = lut.GetWindowSafe(WindowType::HANN, 512);

    ASSERT_NE(safe_window1, nullptr);
    ASSERT_NE(safe_window2, nullptr);

    // 같은 윈도우는 같은 데이터를 가리켜야 함
    EXPECT_EQ(safe_window1.get(), safe_window2.get()) << "Same window should share data";

    // 참조 카운트 확인 (최소 2개: safe_window1, safe_window2)
    EXPECT_GE(safe_window1.use_count(), 2) << "Reference count should be at least 2";

    // 캐시 클리어 후에도 기존 참조는 유효해야 함
    lut.clearCache(false);  // Generation 기반 무효화

    // 기존 참조는 여전히 유효
    EXPECT_NE(safe_window1.get(), nullptr) << "Existing reference should remain valid";
    EXPECT_NE(safe_window2.get(), nullptr) << "Existing reference should remain valid";

    // 새로운 요청은 다른 데이터를 가져올 수 있음 (새 generation)
    auto safe_window3 = lut.GetWindowSafe(WindowType::HANN, 512);
    ASSERT_NE(safe_window3, nullptr);

    // 데이터 내용은 동일해야 함
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_NEAR(safe_window1.get()[i], safe_window3.get()[i], 1e-6f)
            << "Window data should be identical at index " << i;
    }

    std::cout << "캐시 안전성 테스트 - Generation: " << lut.getCurrentGeneration()
              << ", 참조 카운트: " << safe_window1.use_count() << std::endl;
}

// 멀티스레드 캐시 안전성 테스트
TEST_F(WindowLUTTest, MultithreadedCacheSafety) {
    auto& lut = WindowLUT::getInstance();
    const size_t num_threads = 8;
    const size_t num_requests = 50;

    std::vector<std::thread> threads;
    std::vector<std::vector<std::shared_ptr<const float>>> results(num_threads);
    std::atomic<bool> start_flag{false};
    std::atomic<int> ready_count{0};

    // 여러 스레드에서 동시에 안전한 윈도우 요청
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            results[t].reserve(num_requests);
            ready_count.fetch_add(1);

            // 모든 스레드가 준비될 때까지 대기
            while (!start_flag.load()) {
                std::this_thread::yield();
            }

            for (size_t i = 0; i < num_requests; ++i) {
                WindowType type = static_cast<WindowType>(i % 3);
                size_t N = 64 + (i % 10) * 64;

                auto safe_window = lut.GetWindowSafe(type, N);
                results[t].push_back(safe_window);

                // 중간에 캐시 클리어 시도 (일부 스레드에서만)
                if (t == 0 && i == num_requests / 2) {
                    lut.clearCache(false);  // Generation 기반 안전한 클리어
                }

                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });
    }

    // 모든 스레드가 준비되면 시작
    while (ready_count.load() < static_cast<int>(num_threads)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    start_flag.store(true);

    // 모든 스레드 완료 대기
    for (auto& thread : threads) {
        thread.join();
    }

    // 결과 검증: 모든 참조가 유효해야 함
    for (size_t t = 0; t < num_threads; ++t) {
        for (size_t i = 0; i < num_requests; ++i) {
            EXPECT_NE(results[t][i], nullptr)
                << "Thread " << t << " request " << i << " should be valid";
            EXPECT_GE(results[t][i].use_count(), 1)
                << "Reference count should be at least 1";
        }
    }

    std::cout << "멀티스레드 안전성 테스트 완료 - 최종 Generation: "
              << lut.getCurrentGeneration() << std::endl;
}

// Generation 기반 캐시 무효화 테스트
TEST_F(WindowLUTTest, GenerationBasedInvalidation) {
    auto& lut = WindowLUT::getInstance();

    uint64_t initial_gen = lut.getCurrentGeneration();

    // 첫 번째 윈도우 생성
    auto window1 = lut.GetWindowSafe(WindowType::HANN, 256);
    ASSERT_NE(window1, nullptr);

    // Generation 확인
    EXPECT_EQ(lut.getCurrentGeneration(), initial_gen);

    // 캐시 클리어 (Generation 증가)
    lut.clearCache(false);
    uint64_t new_gen = lut.getCurrentGeneration();
    EXPECT_GT(new_gen, initial_gen) << "Generation should increase after clearCache";

    // 기존 참조는 여전히 유효
    EXPECT_NE(window1.get(), nullptr);

    // 새로운 윈도우 요청 (새 Generation)
    auto window2 = lut.GetWindowSafe(WindowType::HANN, 256);
    ASSERT_NE(window2, nullptr);

    // 데이터는 다른 메모리 위치일 수 있지만 내용은 동일
    bool same_pointer = (window1.get() == window2.get());
    if (!same_pointer) {
        // 다른 포인터라면 내용은 동일해야 함
        for (size_t i = 0; i < 256; ++i) {
            EXPECT_NEAR(window1.get()[i], window2.get()[i], 1e-6f)
                << "Window content should be identical at index " << i;
        }
    }

    std::cout << "Generation 테스트 - 초기: " << initial_gen
              << ", 현재: " << new_gen
              << ", 포인터 동일: " << (same_pointer ? "예" : "아니오") << std::endl;
}

// 메모리 누수 방지 테스트
TEST_F(WindowLUTTest, MemoryLeakPrevention) {
    auto& lut = WindowLUT::getInstance();

    std::vector<std::shared_ptr<const float>> windows;

    // 많은 윈도우 생성
    for (int i = 0; i < 100; ++i) {
        auto window = lut.GetWindowSafe(WindowType::HANN, 512 + i);
        windows.push_back(window);

        // 중간중간 캐시 클리어
        if (i % 20 == 0) {
            lut.clearCache(false);
        }
    }

    // 모든 참조가 유효한지 확인
    for (size_t i = 0; i < windows.size(); ++i) {
        EXPECT_NE(windows[i], nullptr) << "Window " << i << " should be valid";
        EXPECT_GE(windows[i].use_count(), 1) << "Reference count should be positive";
    }

    // 참조 해제
    windows.clear();

    // 강제 캐시 클리어
    lut.clearCache(true);

    // 새로운 윈도우 생성이 정상 작동하는지 확인
    auto new_window = lut.GetWindowSafe(WindowType::HANN, 1024);
    EXPECT_NE(new_window, nullptr) << "New window creation should work after cleanup";

    std::cout << "메모리 누수 방지 테스트 완료" << std::endl;
}