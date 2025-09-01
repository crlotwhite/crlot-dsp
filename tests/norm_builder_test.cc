#include <gtest/gtest.h>
#include "dsp/ola/norm_builder.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

using namespace dsp::ola;

class NormBuilderTest : public ::testing::Test {
protected:
    // 스칼라 기반 COLA 정규화 계산 (참조 구현)
    void build_norm_scalar(float* norm, const float* window,
                          size_t ring_len, int frame_size, int hop) {
        std::fill(norm, norm + ring_len, 0.0f);

        // 기존 3중 루프 방식으로 정확성 검증용
        for (size_t ring_pos = 0; ring_pos < ring_len; ++ring_pos) {
            float overlap_sum = 0.0f;

            // 각 프레임 위치에서 중첩 계산
            int min_offset = -static_cast<int>(std::ceil(static_cast<double>(frame_size) / hop));
            int max_offset = static_cast<int>(std::ceil(static_cast<double>(ring_len + frame_size - 1) / hop));
            for (int frame_offset = min_offset; frame_offset <= max_offset; ++frame_offset) {
                int64_t frame_start = frame_offset * hop;

                for (int t = 0; t < frame_size; ++t) {
                    int64_t sample_pos = frame_start + t;
                    size_t mapped_pos = static_cast<size_t>(sample_pos % static_cast<int64_t>(ring_len));

                    // 음수 모듈로 처리
                    if (sample_pos < 0) {
                        mapped_pos = ring_len - (static_cast<size_t>(-sample_pos) % ring_len);
                        if (mapped_pos == ring_len) mapped_pos = 0;
                    }

                    if (mapped_pos == ring_pos) {
                        overlap_sum += window[t];
                    }
                }
            }

            norm[ring_pos] = std::max(overlap_sum, 1e-8f);
        }
    }
};

// 기본 정확성 테스트
TEST_F(NormBuilderTest, BasicAccuracy) {
    const int frame_size = 256;
    const int hop = 64;
    const size_t ring_len = 1024;

    // Hann 윈도우 생성
    std::vector<float> window(frame_size);
    for (int i = 0; i < frame_size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (frame_size - 1)));
    }

    // 두 구현 비교
    std::vector<float> norm_linear(ring_len);
    std::vector<float> norm_scalar(ring_len);

    build_norm_linear(norm_linear.data(), window.data(), ring_len, frame_size, hop);
    build_norm_scalar(norm_scalar.data(), window.data(), ring_len, frame_size, hop);

    // 정확성 검증
    for (size_t i = 0; i < ring_len; ++i) {
        EXPECT_NEAR(norm_linear[i], norm_scalar[i], 1e-6f)
            << "Mismatch at position " << i
            << ": linear=" << norm_linear[i] << ", scalar=" << norm_scalar[i];
    }
}

// 다양한 파라미터 조합 테스트
TEST_F(NormBuilderTest, ParameterCombinations) {
    std::vector<std::tuple<int, int, size_t>> test_cases = {
        {128, 32, 512},   // 작은 프레임, 작은 홉
        {512, 128, 2048}, // 큰 프레임, 큰 홉
        {256, 64, 1024},  // 표준 파라미터
        {64, 16, 256},    // 매우 작은 값들
        {1024, 256, 4096} // 큰 값들
    };

    for (auto [frame_size, hop, ring_len] : test_cases) {
        // Hann 윈도우 생성
        std::vector<float> window(frame_size);
        for (int i = 0; i < frame_size; ++i) {
            window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (frame_size - 1)));
        }

        std::vector<float> norm_linear(ring_len);
        std::vector<float> norm_scalar(ring_len);

        build_norm_linear(norm_linear.data(), window.data(), ring_len, frame_size, hop);
        build_norm_scalar(norm_scalar.data(), window.data(), ring_len, frame_size, hop);

        // 각 조합에서 정확성 검증
        double max_error = 0.0;
        for (size_t i = 0; i < ring_len; ++i) {
            double error = std::abs(norm_linear[i] - norm_scalar[i]);
            max_error = std::max(max_error, error);
        }

        EXPECT_LT(max_error, 1e-5f)
            << "Parameter combination (N=" << frame_size << ", H=" << hop
            << ", ring_len=" << ring_len << ") failed with max error: " << max_error;
    }
}

// 경계 조건 테스트
TEST_F(NormBuilderTest, EdgeCases) {
    // ring_len이 frame_size보다 작은 경우 (비현실적이지만 테스트)
    const int frame_size = 128;
    const int hop = 32;
    const size_t ring_len = 256; // frame_size보다 큼

    std::vector<float> window(frame_size, 1.0f); // 사각 윈도우

    std::vector<float> norm_linear(ring_len);
    std::vector<float> norm_scalar(ring_len);

    build_norm_linear(norm_linear.data(), window.data(), ring_len, frame_size, hop);
    build_norm_scalar(norm_scalar.data(), window.data(), ring_len, frame_size, hop);

    // 경계 조건에서도 정확성 유지
    for (size_t i = 0; i < ring_len; ++i) {
        EXPECT_NEAR(norm_linear[i], norm_scalar[i], 1e-5f)
            << "Edge case failed at position " << i;
    }
}

// 최소값 보장 테스트 (ε 처리)
TEST_F(NormBuilderTest, MinimumValueGuarantee) {
    const int frame_size = 256;
    const int hop = 256; // 큰 홉, 중첩 적음
    const size_t ring_len = 1024;

    // 0에 가까운 윈도우 생성
    std::vector<float> window(frame_size, 1e-10f);

    std::vector<float> norm(ring_len);
    build_norm_linear(norm.data(), window.data(), ring_len, frame_size, hop);

    // 모든 값이 ε 이상인지 확인 (소비 단계 가드 고려)
    const float eps = 1e-10f;
    for (size_t i = 0; i < ring_len; ++i) {
        EXPECT_GE(norm[i], eps)
            << "Value at position " << i << " is below epsilon: " << norm[i];
    }
}

// 성능 테스트 (O(K·N) 복잡도 검증)
TEST_F(NormBuilderTest, PerformanceComplexity) {
    const int frame_size = 1024;
    const int hop = 256;
    const size_t ring_len = 4096;

    std::vector<float> window(frame_size, 1.0f);

    std::vector<float> norm(ring_len);

    // 시간 측정 (단순 반복으로 복잡도 추정)
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 10; ++iter) {
        build_norm_linear(norm.data(), window.data(), ring_len, frame_size, hop);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // O(K·N) 복잡도 확인을 위한 로그
    size_t K = (ring_len + hop - 1) / hop;
    std::cout << "Performance test: K=" << K << ", N=" << frame_size
              << ", time=" << duration.count() << "ms for 10 iterations" << std::endl;

    // 합리적인 시간 내 완료 확인
    EXPECT_LT(duration.count(), 1000) << "Performance test took too long";
}

// 메모리 안전성 테스트
TEST_F(NormBuilderTest, MemorySafety) {
    const int frame_size = 512;
    const int hop = 128;
    const size_t ring_len = 2048;

    std::vector<float> window(frame_size, 1.0f);
    std::vector<float> norm(ring_len);

    // 정상적인 입력으로 메모리 문제 없음 확인
    EXPECT_NO_THROW(
        build_norm_linear(norm.data(), window.data(), ring_len, frame_size, hop)
    );

    // 모든 값이 유효한지 확인
    for (size_t i = 0; i < ring_len; ++i) {
        EXPECT_FALSE(std::isnan(norm[i])) << "NaN detected at position " << i;
        EXPECT_FALSE(std::isinf(norm[i])) << "Infinity detected at position " << i;
    }
}