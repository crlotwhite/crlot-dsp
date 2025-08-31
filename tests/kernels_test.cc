#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>
#include "dsp/ola/kernels.h"

using namespace dsp;

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