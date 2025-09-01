#include <gtest/gtest.h>
#include "dsp/ola/kernels.h"
#include <cstdio>

using namespace dsp;

/**
 * 런타임 디스패치 디버깅 정보 출력 테스트
 */
TEST(DispatchDebugTest, PrintDispatchInfo) {
    printf("\n"); // 가독성을 위한 줄바꿈

    // 디스패치 정보 출력
    print_kernel_dispatch_info();

    // 개별 함수들도 테스트
    printf("\nIndividual function tests:\n");
    printf("get_supported_targets(): %s\n", get_supported_targets());
    printf("get_current_target(): %s\n", get_current_target());
    printf("get_simd_lanes(): %zu\n", get_simd_lanes());
}

/**
 * 런타임 디스패치 함수들의 기본 동작 검증
 */
TEST(DispatchDebugTest, DispatchFunctionsWork) {
    // 지원 타겟 문자열이 null이 아닌지 확인
    const char* supported = get_supported_targets();
    ASSERT_NE(supported, nullptr);
    ASSERT_GT(strlen(supported), 0);

    // 현재 타겟 문자열이 null이 아닌지 확인
    const char* current = get_current_target();
    ASSERT_NE(current, nullptr);
    ASSERT_GT(strlen(current), 0);

    // SIMD lanes가 합리적인 범위인지 확인
    size_t lanes = get_simd_lanes();
    ASSERT_GE(lanes, 1);
    ASSERT_LE(lanes, 64);  // 합리적인 상한선
}

/**
 * 실제 커널 함수들이 작동하는지 간단 검증
 */
TEST(DispatchDebugTest, KernelFunctionsWork) {
    const size_t n = 16;
    std::vector<float> dst(n, 1.0f);
    std::vector<float> src(n, 2.0f);
    std::vector<float> win(n, 0.5f);
    std::vector<float> norm(n, 2.0f);
    std::vector<float> acc(n, 4.0f);
    std::vector<float> out(n);

    printf("\nTesting kernel functions with debug info:\n");
    printf("Input size: %zu (SIMD lanes: %zu)\n", n, get_simd_lanes());

    // AXPY 테스트
    axpy(dst.data(), src.data(), 0.5f, n);
    printf("AXPY completed - first element: %.2f (expected: 2.0)\n", dst[0]);
    EXPECT_NEAR(dst[0], 2.0f, 1e-6f);

    // AXPY Windowed 테스트
    std::fill(dst.begin(), dst.end(), 1.0f);
    axpy_windowed(dst.data(), src.data(), win.data(), 1.0f, n);
    printf("AXPY Windowed completed - first element: %.2f (expected: 2.0)\n", dst[0]);
    EXPECT_NEAR(dst[0], 2.0f, 1e-6f);

    // Normalize and Clear 테스트
    normalize_and_clear(out.data(), acc.data(), norm.data(), 1e-8f, n);
    printf("Normalize and Clear completed - first element: %.2f (expected: 2.0)\n", out[0]);
    printf("Accumulator cleared - first element: %.2f (expected: 0.0)\n", acc[0]);
    EXPECT_NEAR(out[0], 2.0f, 1e-6f);
    EXPECT_EQ(acc[0], 0.0f);
}

/**
 * 성능 비교 테스트 (스칼라 vs Highway)
 */
TEST(DispatchDebugTest, PerformanceComparison) {
    const size_t n = 1024;
    std::vector<float> dst_scalar(n, 1.0f);
    std::vector<float> dst_hwy(n, 1.0f);
    std::vector<float> src(n, 2.0f);

    printf("\nPerformance comparison test (size: %zu):\n", n);
    printf("Current SIMD target: %s (%zu lanes)\n", get_current_target(), get_simd_lanes());

    // 간단한 타이밍 비교 (정확하지는 않지만 대략적인 차이 확인)
    clock_t start, end;

    // 스칼라 구현
    start = clock();
    for (int i = 0; i < 100; ++i) {
        axpy_scalar(dst_scalar.data(), src.data(), 0.5f, n);
    }
    end = clock();
    double scalar_time = double(end - start) / CLOCKS_PER_SEC;

    // Highway 구현
    start = clock();
    for (int i = 0; i < 100; ++i) {
        axpy_hwy(dst_hwy.data(), src.data(), 0.5f, n);
    }
    end = clock();
    double hwy_time = double(end - start) / CLOCKS_PER_SEC;

    printf("Scalar time: %.6f seconds\n", scalar_time);
    printf("Highway time: %.6f seconds\n", hwy_time);
    if (hwy_time > 0) {
        printf("Speedup: %.2fx\n", scalar_time / hwy_time);
    }

    // 결과 정확성 검증
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(dst_scalar[i], dst_hwy[i], 1e-6f) << "Mismatch at index " << i;
    }
}