#include <gtest/gtest.h>
#include "dsp/window/WindowLUT.h"
#include <cmath>
#include <vector>
#include <chrono>

using namespace dsp;

class WindowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 각 테스트 전에 캐시 초기화
        WindowLUT::getInstance().clearCache();
    }

    void TearDown() override {
        // 테스트 후 캐시 초기화
        WindowLUT::getInstance().clearCache();
    }

    // 참조 값들 (스폿체크용)
    struct ReferenceValues {
        float first;
        float middle;
        float last;
    };

    // Hann 윈도우 참조값 계산
    ReferenceValues calculateHannReference(size_t N, bool periodic = false) {
        ReferenceValues ref;
        if (N == 1) {
            ref.first = ref.middle = ref.last = 1.0f;
            return ref;
        }

        const double pi = M_PI;
        const double denominator = periodic ? static_cast<double>(N) : static_cast<double>(N - 1);
        const double factor = 2.0 * pi / denominator;

        ref.first = static_cast<float>(0.5 * (1.0 - std::cos(0.0)));
        ref.middle = static_cast<float>(0.5 * (1.0 - std::cos(factor * (N / 2))));
        ref.last = static_cast<float>(0.5 * (1.0 - std::cos(factor * (N - 1))));

        return ref;
    }

    // Hamming 윈도우 참조값 계산
    ReferenceValues calculateHammingReference(size_t N, bool periodic = false) {
        ReferenceValues ref;
        if (N == 1) {
            ref.first = ref.middle = ref.last = 1.0f;
            return ref;
        }

        const double pi = M_PI;
        const double denominator = periodic ? static_cast<double>(N) : static_cast<double>(N - 1);
        const double factor = 2.0 * pi / denominator;
        const double alpha = 0.54;
        const double beta = 0.46;

        ref.first = static_cast<float>(alpha - beta * std::cos(0.0));
        ref.middle = static_cast<float>(alpha - beta * std::cos(factor * (N / 2)));
        ref.last = static_cast<float>(alpha - beta * std::cos(factor * (N - 1)));

        return ref;
    }

    // Blackman 윈도우 참조값 계산
    ReferenceValues calculateBlackmanReference(size_t N, bool periodic = false) {
        ReferenceValues ref;
        if (N == 1) {
            ref.first = ref.middle = ref.last = 1.0f;
            return ref;
        }

        const double pi = M_PI;
        const double denominator = periodic ? static_cast<double>(N) : static_cast<double>(N - 1);
        const double factor = 2.0 * pi / denominator;
        const double a0 = 0.42;
        const double a1 = 0.5;
        const double a2 = 0.08;

        auto calc = [&](size_t i) {
            double angle = factor * static_cast<double>(i);
            double cos1 = std::cos(angle);
            double cos2 = std::cos(2.0 * angle);
            return static_cast<float>(a0 - a1 * cos1 + a2 * cos2);
        };

        ref.first = calc(0);
        ref.middle = calc(N / 2);
        ref.last = calc(N - 1);

        return ref;
    }
};

// PR2 점검표: Hann/Hamming/Blackman 값 스폿체크 (첫/중앙/끝 샘플 오차 < 1e-6)
TEST_F(WindowTest, HannSpotCheck) {
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};

    for (size_t N : sizes) {
        // 비주기적 윈도우
        WindowLUT lut(N, WindowType::HANN, false);
        const float* window = lut.data();
        auto ref = calculateHannReference(N, false);

        EXPECT_NEAR(window[0], ref.first, 1e-6f)
            << "Hann first value incorrect for N=" << N;
        EXPECT_NEAR(window[N/2], ref.middle, 1e-6f)
            << "Hann middle value incorrect for N=" << N;
        EXPECT_NEAR(window[N-1], ref.last, 1e-6f)
            << "Hann last value incorrect for N=" << N;

        // 주기적 윈도우 (FFT용)
        WindowLUT lut_periodic(N, WindowType::HANN, true);
        const float* window_periodic = lut_periodic.data();
        auto ref_periodic = calculateHannReference(N, true);

        EXPECT_NEAR(window_periodic[0], ref_periodic.first, 1e-6f)
            << "Hann periodic first value incorrect for N=" << N;
        EXPECT_NEAR(window_periodic[N/2], ref_periodic.middle, 1e-6f)
            << "Hann periodic middle value incorrect for N=" << N;
        EXPECT_NEAR(window_periodic[N-1], ref_periodic.last, 1e-6f)
            << "Hann periodic last value incorrect for N=" << N;
    }
}

TEST_F(WindowTest, HammingSpotCheck) {
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};

    for (size_t N : sizes) {
        WindowLUT lut(N, WindowType::HAMMING, false);
        const float* window = lut.data();
        auto ref = calculateHammingReference(N, false);

        EXPECT_NEAR(window[0], ref.first, 1e-6f)
            << "Hamming first value incorrect for N=" << N;
        EXPECT_NEAR(window[N/2], ref.middle, 1e-6f)
            << "Hamming middle value incorrect for N=" << N;
        EXPECT_NEAR(window[N-1], ref.last, 1e-6f)
            << "Hamming last value incorrect for N=" << N;
    }
}

TEST_F(WindowTest, BlackmanSpotCheck) {
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};

    for (size_t N : sizes) {
        WindowLUT lut(N, WindowType::BLACKMAN, false);
        const float* window = lut.data();
        auto ref = calculateBlackmanReference(N, false);

        EXPECT_NEAR(window[0], ref.first, 1e-6f)
            << "Blackman first value incorrect for N=" << N;
        EXPECT_NEAR(window[N/2], ref.middle, 1e-6f)
            << "Blackman middle value incorrect for N=" << N;
        EXPECT_NEAR(window[N-1], ref.last, 1e-6f)
            << "Blackman last value incorrect for N=" << N;
    }
}

// PR2 점검표: 정규화 옵션 동작 테스트
TEST_F(WindowTest, NormalizationL2) {
    size_t N = 512;
    WindowLUT lut(N, WindowType::HANN, false, NormalizationType::L2_NORM);
    const float* window = lut.data();

    double sum_sq = WindowLUT::calculateSumOfSquares(window, N);
    EXPECT_NEAR(sum_sq, 1.0, 1e-6) << "L2 normalization failed";
}

TEST_F(WindowTest, NormalizationSumToOne) {
    size_t N = 512;
    WindowLUT lut(N, WindowType::HANN, false, NormalizationType::SUM_TO_ONE);
    const float* window = lut.data();

    double sum = WindowLUT::calculateSum(window, N);
    EXPECT_NEAR(sum, 1.0, 1e-6) << "Sum=1 normalization failed";
}

TEST_F(WindowTest, NormalizationOLA) {
    size_t N = 512;

    WindowLUT lut(N, WindowType::HANN, false, NormalizationType::OLA_UNITY_GAIN);
    const float* window = lut.data();

    // OLA 정규화는 현재 L2 정규화로 구현됨
    double sum_sq = WindowLUT::calculateSumOfSquares(window, N);
    EXPECT_NEAR(sum_sq, 1.0, 1e-6) << "OLA normalized window should have L2 norm = 1";
}

// Rect 윈도우 테스트
TEST_F(WindowTest, RectWindow) {
    size_t N = 256;
    WindowLUT lut(N, WindowType::RECT, false);
    const float* window = lut.data();

    // 모든 값이 1.0이어야 함
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(window[i], 1.0f, 1e-6f) << "Rect window value incorrect at index " << i;
    }
}

// 성능 테스트: LUT 생성 시간 vs 재사용
TEST_F(WindowTest, PerformanceLUTCreationVsReuse) {
    size_t N = 2048;
    const int num_iterations = 1000;

    // 생성 시간 측정
    auto start_creation = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        WindowLUT lut(N, WindowType::HANN, false);
        volatile const float* window = lut.data();  // 최적화 방지
        (void)window;
    }
    auto end_creation = std::chrono::high_resolution_clock::now();
    auto creation_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_creation - start_creation).count();

    // 재사용 시간 측정 (캐시 사용)
    auto& lut_cache = WindowLUT::getInstance();
    auto start_reuse = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        volatile const float* window = lut_cache.GetWindow(WindowType::HANN, N);
        (void)window;
    }
    auto end_reuse = std::chrono::high_resolution_clock::now();
    auto reuse_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_reuse - start_reuse).count();

    // 재사용이 훨씬 빨라야 함
    EXPECT_LT(reuse_time * 10, creation_time)
        << "Cache reuse should be at least 10x faster than creation";

    std::cout << "Creation time: " << creation_time << " μs" << std::endl;
    std::cout << "Reuse time: " << reuse_time << " μs" << std::endl;
    std::cout << "Speedup: " << static_cast<double>(creation_time) / reuse_time << "x" << std::endl;
}

// 메모리 정렬 테스트 (16B 이상)
TEST_F(WindowTest, MemoryAlignment) {
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};

    for (size_t N : sizes) {
        WindowLUT lut(N, WindowType::HANN, false);
        const float* window = lut.data();

        // 16-byte 정렬 확인
        uintptr_t addr = reinterpret_cast<uintptr_t>(window);
        EXPECT_EQ(addr % 16, 0) << "Window data not 16-byte aligned for N=" << N;
    }
}

// 주기적 vs 비주기적 윈도우 차이 검증
TEST_F(WindowTest, PeriodicVsNonPeriodic) {
    size_t N = 64;

    WindowLUT lut_nonperiodic(N, WindowType::HANN, false);
    WindowLUT lut_periodic(N, WindowType::HANN, true);

    const float* win_nonperiodic = lut_nonperiodic.data();
    const float* win_periodic = lut_periodic.data();

    // 비주기적 윈도우는 양 끝이 0에 가까워야 함
    EXPECT_NEAR(win_nonperiodic[0], 0.0f, 1e-6f);
    EXPECT_NEAR(win_nonperiodic[N-1], 0.0f, 1e-6f);

    // 주기적 윈도우는 다른 값을 가져야 함 (중간값으로 비교)
    EXPECT_NE(win_periodic[N/4], win_nonperiodic[N/4]);

    // 또는 전체 RMS 차이로 검증
    double rms_diff = WindowLUT::calculateRMSError(win_periodic, win_nonperiodic, N);
    EXPECT_GT(rms_diff, 1e-6) << "Periodic and non-periodic windows should be different";
}