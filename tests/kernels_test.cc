#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstring>
#include "dsp/ola/kernels.h"

using namespace dsp;

/**
 * ULP(Units in the Last Place) 차이를 계산하는 유틸리티 함수
 * IEEE 754 표준에 따른 부동소수점 정확도 검증용
 */
uint32_t ulp_diff(float a, float b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b)) return UINT32_MAX;

    // 부호가 다르면 매우 큰 차이로 간주
    if ((a > 0) != (b > 0)) return UINT32_MAX;

    uint32_t ia, ib;
    std::memcpy(&ia, &a, sizeof(float));
    std::memcpy(&ib, &b, sizeof(float));

    // 음수의 경우 2의 보수 표현 조정
    if (a < 0) {
        ia = 0x80000000 - ia;
        ib = 0x80000000 - ib;
    }

    return (ia > ib) ? (ia - ib) : (ib - ia);
}

/**
 * ±1 ULP 오차 범위 내인지 확인
 */
bool is_within_1_ulp(float a, float b) {
    return ulp_diff(a, b) <= 1;
}

class KernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 테스트용 데이터 생성
        test_data_.resize(100);
        std::iota(test_data_.begin(), test_data_.end(), 0.0f);

        window_data_.resize(100);
        for (size_t i = 0; i < window_data_.size(); ++i) {
            window_data_[i] = 0.5f + 0.5f * std::cos(2.0 * M_PI * i / (window_data_.size() - 1));
        }

        norm_data_.resize(100);
        for (size_t i = 0; i < norm_data_.size(); ++i) {
            norm_data_[i] = 1.0f + 0.1f * i;  // 1.0에서 10.0까지
        }
    }

    std::vector<float> test_data_;
    std::vector<float> window_data_;
    std::vector<float> norm_data_;
};

// axpy 함수 테스트
TEST_F(KernelsTest, AxpyBasic) {
    std::vector<float> dst(10, 1.0f);
    std::vector<float> src(10, 2.0f);
    float g = 0.5f;

    axpy(dst.data(), src.data(), g, 10);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(dst[i], 1.0f + 2.0f * 0.5f, 1e-6f);
    }
}

TEST_F(KernelsTest, AxpyZeroGain) {
    std::vector<float> dst(10, 1.0f);
    std::vector<float> src(10, 2.0f);
    std::vector<float> original_dst = dst;

    axpy(dst.data(), src.data(), 0.0f, 10);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(dst[i], original_dst[i]);
    }
}

TEST_F(KernelsTest, AxpyLargeN) {
    size_t n = 1000;
    std::vector<float> dst(n, 1.0f);
    std::vector<float> src(n, 2.0f);
    float g = 0.5f;

    axpy(dst.data(), src.data(), g, n);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(dst[i], 1.0f + 2.0f * 0.5f, 1e-6f);
    }
}

TEST_F(KernelsTest, AxpyZeroN) {
    std::vector<float> dst(10, 1.0f);
    std::vector<float> src(10, 2.0f);
    std::vector<float> original_dst = dst;

    axpy(dst.data(), src.data(), 0.5f, 0);

    // n=0이므로 dst가 변경되지 않아야 함
    EXPECT_EQ(dst, original_dst);
}

// axpy_windowed 함수 테스트
TEST_F(KernelsTest, AxpyWindowedBasic) {
    std::vector<float> dst(10, 1.0f);
    std::vector<float> src(10, 2.0f);
    std::vector<float> win(10, 0.5f);
    float g = 1.0f;

    axpy_windowed(dst.data(), src.data(), win.data(), g, 10);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(dst[i], 1.0f + 2.0f * 0.5f * 1.0f, 1e-6f);
    }
}

TEST_F(KernelsTest, AxpyWindowedZeroN) {
    std::vector<float> dst(10, 1.0f);
    std::vector<float> src(10, 2.0f);
    std::vector<float> win(10, 0.5f);
    std::vector<float> original_dst = dst;

    axpy_windowed(dst.data(), src.data(), win.data(), 1.0f, 0);

    EXPECT_EQ(dst, original_dst);
}

TEST_F(KernelsTest, AxpyWindowedLargeN) {
    size_t n = 1000;
    std::vector<float> dst(n, 1.0f);
    std::vector<float> src(n, 2.0f);
    std::vector<float> win(n, 0.5f);
    float g = 1.0f;

    axpy_windowed(dst.data(), src.data(), win.data(), g, n);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(dst[i], 1.0f + 2.0f * 0.5f * 1.0f, 1e-6f);
    }
}

// normalize_and_clear 함수 테스트
TEST_F(KernelsTest, NormalizeAndClearBasic) {
    std::vector<float> out(10);
    std::vector<float> acc(10, 2.0f);
    std::vector<float> norm(10, 2.0f);
    float eps = 1e-6f;

    normalize_and_clear(out.data(), acc.data(), norm.data(), eps, 10);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(out[i], 1.0f, 1e-6f);  // 2.0 / 2.0 = 1.0
        EXPECT_EQ(acc[i], 0.0f);  // acc가 클리어되었는지 확인
    }
}

TEST_F(KernelsTest, NormalizeAndClearWithEps) {
    std::vector<float> out(10);
    std::vector<float> acc(10, 1.0f);
    std::vector<float> norm(10, 0.0f);  // 0으로 나누기 시도
    float eps = 0.1f;

    normalize_and_clear(out.data(), acc.data(), norm.data(), eps, 10);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(out[i], 1.0f / 0.1f, 1e-6f);  // 1.0 / (0.0 + 0.1) = 10.0
        EXPECT_EQ(acc[i], 0.0f);
    }
}

TEST_F(KernelsTest, NormalizeAndClearZeroN) {
    std::vector<float> out(10);
    std::vector<float> acc(10, 2.0f);
    std::vector<float> norm(10, 2.0f);
    std::vector<float> original_acc = acc;

    normalize_and_clear(out.data(), acc.data(), norm.data(), 1e-6f, 0);

    // n=0이므로 acc가 변경되지 않아야 함
    EXPECT_EQ(acc, original_acc);
}

TEST_F(KernelsTest, NormalizeAndClearLargeN) {
    size_t n = 1000;
    std::vector<float> out(n);
    std::vector<float> acc(n, 3.0f);
    std::vector<float> norm(n, 1.5f);
    float eps = 1e-6f;

    normalize_and_clear(out.data(), acc.data(), norm.data(), eps, n);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], 3.0f / 1.5f, 1e-5f);  // 2.0
        EXPECT_EQ(acc[i], 0.0f);
    }
}

//==============================================================================
// Highway vs 스칼라 구현 정확도 검증 테스트
//==============================================================================

class HighwayAccuracyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 랜덤 데이터 생성
        std::random_device rd;
        std::mt19937 gen(42); // 재현가능한 테스트를 위해 고정 시드
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        for (size_t size : test_sizes_) {
            random_data_[size] = std::vector<float>(size);
            window_data_[size] = std::vector<float>(size);
            norm_data_[size] = std::vector<float>(size);

            for (size_t i = 0; i < size; ++i) {
                random_data_[size][i] = dis(gen);
                window_data_[size][i] = 0.5f + 0.5f * std::cos(2.0f * M_PI * i / size);
                norm_data_[size][i] = 0.1f + dis(gen) * dis(gen); // 양수 보장
            }
        }
    }

    std::vector<size_t> test_sizes_ = {0, 1, 7, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 1024, 4096};
    std::map<size_t, std::vector<float>> random_data_;
    std::map<size_t, std::vector<float>> window_data_;
    std::map<size_t, std::vector<float>> norm_data_;
};

// AXPY 정확도 테스트
TEST_F(HighwayAccuracyTest, AxpyAccuracy) {
    for (size_t n : test_sizes_) {
        if (n == 0) continue; // n=0은 별도 테스트

        std::vector<float> dst_scalar(n, 1.0f);
        std::vector<float> dst_hwy(n, 1.0f);
        const auto& src = random_data_[n];
        float g = 0.75f;

        // 스칼라와 Highway 구현으로 동일한 연산 수행
        axpy_scalar(dst_scalar.data(), src.data(), g, n);
        axpy_hwy(dst_hwy.data(), src.data(), g, n);

        // 결과 비교 (±1 ULP 오차 허용)
        for (size_t i = 0; i < n; ++i) {
            EXPECT_TRUE(is_within_1_ulp(dst_scalar[i], dst_hwy[i]))
                << "Size " << n << ", index " << i
                << ": scalar=" << dst_scalar[i] << ", highway=" << dst_hwy[i]
                << ", ULP diff=" << ulp_diff(dst_scalar[i], dst_hwy[i]);
        }
    }
}

// AXPY Windowed 정확도 테스트
TEST_F(HighwayAccuracyTest, AxpyWindowedAccuracy) {
    for (size_t n : test_sizes_) {
        if (n == 0) continue;

        std::vector<float> dst_scalar(n, 0.5f);
        std::vector<float> dst_hwy(n, 0.5f);
        const auto& src = random_data_[n];
        const auto& win = window_data_[n];
        float g = 1.25f;

        axpy_windowed_scalar(dst_scalar.data(), src.data(), win.data(), g, n);
        axpy_windowed_hwy(dst_hwy.data(), src.data(), win.data(), g, n);

        for (size_t i = 0; i < n; ++i) {
            EXPECT_TRUE(is_within_1_ulp(dst_scalar[i], dst_hwy[i]))
                << "Size " << n << ", index " << i
                << ": scalar=" << dst_scalar[i] << ", highway=" << dst_hwy[i]
                << ", ULP diff=" << ulp_diff(dst_scalar[i], dst_hwy[i]);
        }
    }
}

// Normalize and Clear 정확도 테스트
TEST_F(HighwayAccuracyTest, NormalizeAndClearAccuracy) {
    for (size_t n : test_sizes_) {
        if (n == 0) continue;

        std::vector<float> out_scalar(n);
        std::vector<float> out_hwy(n);
        std::vector<float> acc_scalar = random_data_[n]; // 복사본
        std::vector<float> acc_hwy = random_data_[n];    // 복사본
        const auto& norm = norm_data_[n];
        float eps = 1e-8f;

        normalize_and_clear_scalar(out_scalar.data(), acc_scalar.data(), norm.data(), eps, n);
        normalize_and_clear_hwy(out_hwy.data(), acc_hwy.data(), norm.data(), eps, n);

        // 출력 비교
        for (size_t i = 0; i < n; ++i) {
            EXPECT_TRUE(is_within_1_ulp(out_scalar[i], out_hwy[i]))
                << "Size " << n << ", index " << i
                << ": scalar=" << out_scalar[i] << ", highway=" << out_hwy[i]
                << ", ULP diff=" << ulp_diff(out_scalar[i], out_hwy[i]);
        }

        // 누산기 클리어 확인
        for (size_t i = 0; i < n; ++i) {
            EXPECT_EQ(acc_scalar[i], acc_hwy[i]) << "Accumulator clear mismatch at " << i;
            EXPECT_EQ(acc_hwy[i], 0.0f) << "Highway accumulator not cleared at " << i;
        }
    }
}

//==============================================================================
// 경계 조건 및 특수 케이스 테스트
//==============================================================================

// n=0 경계 조건 테스트
TEST_F(HighwayAccuracyTest, EdgeCaseZeroN) {
    std::vector<float> dst(10, 1.0f);
    std::vector<float> src(10, 2.0f);
    std::vector<float> win(10, 0.5f);
    std::vector<float> acc(10, 3.0f);
    std::vector<float> out(10);
    std::vector<float> norm(10, 2.0f);

    // 원본 복사본
    auto dst_orig = dst, acc_orig = acc;

    // 모든 함수에서 n=0 테스트
    axpy_hwy(dst.data(), src.data(), 1.0f, 0);
    EXPECT_EQ(dst, dst_orig);

    axpy_windowed_hwy(dst.data(), src.data(), win.data(), 1.0f, 0);
    EXPECT_EQ(dst, dst_orig);

    normalize_and_clear_hwy(out.data(), acc.data(), norm.data(), 1e-8f, 0);
    EXPECT_EQ(acc, acc_orig);
}

// n=1 경계 조건 테스트
TEST_F(HighwayAccuracyTest, EdgeCaseSingleElement) {
    float dst_scalar = 1.0f, dst_hwy = 1.0f;
    float src = 2.5f, win = 0.8f, g = 1.5f;

    axpy_scalar(&dst_scalar, &src, g, 1);
    axpy_hwy(&dst_hwy, &src, g, 1);
    EXPECT_TRUE(is_within_1_ulp(dst_scalar, dst_hwy));

    // 윈도우 적용 테스트
    dst_scalar = dst_hwy = 0.5f;
    axpy_windowed_scalar(&dst_scalar, &src, &win, g, 1);
    axpy_windowed_hwy(&dst_hwy, &src, &win, g, 1);
    EXPECT_TRUE(is_within_1_ulp(dst_scalar, dst_hwy));

    // 정규화 테스트
    float out_scalar, out_hwy;
    float acc_scalar = 4.0f, acc_hwy = 4.0f;
    float norm = 2.0f, eps = 1e-8f;

    normalize_and_clear_scalar(&out_scalar, &acc_scalar, &norm, eps, 1);
    normalize_and_clear_hwy(&out_hwy, &acc_hwy, &norm, eps, 1);

    EXPECT_TRUE(is_within_1_ulp(out_scalar, out_hwy));
    EXPECT_EQ(acc_scalar, 0.0f);
    EXPECT_EQ(acc_hwy, 0.0f);
}

// SIMD lanes 경계 테스트
TEST_F(HighwayAccuracyTest, SIMDLanesBoundaryTest) {
    // SIMD lanes 근처의 크기들에 대해 상세 테스트
    std::vector<size_t> boundary_sizes = {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65};

    for (size_t n : boundary_sizes) {
        std::vector<float> dst_scalar(n), dst_hwy(n);
        std::vector<float> src(n), win(n), norm(n);

        // 특수한 패턴 생성 (SIMD 정렬 이슈 감지용)
        for (size_t i = 0; i < n; ++i) {
            src[i] = static_cast<float>(i + 1) * 0.1f;
            win[i] = 1.0f - static_cast<float>(i) / n;
            norm[i] = 1.0f + static_cast<float>(i) * 0.01f;
            dst_scalar[i] = dst_hwy[i] = static_cast<float>(i) * 0.05f;
        }

        float g = 2.0f;

        // AXPY 테스트
        axpy_scalar(dst_scalar.data(), src.data(), g, n);
        axpy_hwy(dst_hwy.data(), src.data(), g, n);

        for (size_t i = 0; i < n; ++i) {
            EXPECT_TRUE(is_within_1_ulp(dst_scalar[i], dst_hwy[i]))
                << "AXPY Size " << n << ", index " << i;
        }

        // AXPY Windowed 테스트
        std::fill(dst_scalar.begin(), dst_scalar.end(), 0.1f);
        std::fill(dst_hwy.begin(), dst_hwy.end(), 0.1f);

        axpy_windowed_scalar(dst_scalar.data(), src.data(), win.data(), g, n);
        axpy_windowed_hwy(dst_hwy.data(), src.data(), win.data(), g, n);

        for (size_t i = 0; i < n; ++i) {
            EXPECT_TRUE(is_within_1_ulp(dst_scalar[i], dst_hwy[i]))
                << "AXPY Windowed Size " << n << ", index " << i;
        }

        // Normalize and Clear 테스트
        std::vector<float> out_scalar(n), out_hwy(n);
        std::vector<float> acc_scalar = src; // 복사
        std::vector<float> acc_hwy = src;    // 복사

        normalize_and_clear_scalar(out_scalar.data(), acc_scalar.data(), norm.data(), 1e-8f, n);
        normalize_and_clear_hwy(out_hwy.data(), acc_hwy.data(), norm.data(), 1e-8f, n);

        for (size_t i = 0; i < n; ++i) {
            EXPECT_TRUE(is_within_1_ulp(out_scalar[i], out_hwy[i]))
                << "Normalize Size " << n << ", index " << i;
            EXPECT_EQ(acc_scalar[i], acc_hwy[i]) << "Accumulator Size " << n << ", index " << i;
        }
    }
}

// 통합 테스트: 실제 사용 시나리오
TEST_F(KernelsTest, IntegrationScenario) {
    size_t n = 100;
    std::vector<float> dst(n, 0.0f);
    std::vector<float> src = test_data_;
    std::vector<float> win = window_data_;
    std::vector<float> norm = norm_data_;
    std::vector<float> acc(n, 0.0f);
    std::vector<float> out(n);

    float g = 0.5f;
    float eps = 1e-8f;

    // 1. axpy로 누산
    axpy(acc.data(), src.data(), g, n);

    // 2. axpy_windowed로 윈도우 적용 누산
    axpy_windowed(acc.data(), src.data(), win.data(), g, n);

    // 3. 정규화 및 클리어
    normalize_and_clear(out.data(), acc.data(), norm.data(), eps, n);

    // 결과 검증
    for (size_t i = 0; i < n; ++i) {
        float expected = (src[i] * g + src[i] * win[i] * g) / norm[i];
        EXPECT_NEAR(out[i], expected, 1e-5f);
        EXPECT_EQ(acc[i], 0.0f);
    }
}