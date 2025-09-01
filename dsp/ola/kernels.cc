#include "kernels.h"

#include <algorithm>
#include <cmath>  // for std::fma

namespace dsp {

// Highway 구현 함수들 (kernels_hwy.cc에서 정의됨)
extern void axpy_hwy(float* dst, const float* src, float g, size_t n) noexcept;
extern void axpy_windowed_hwy(float* dst, const float* src, const float* win, float g, size_t n) noexcept;
extern void normalize_and_clear_hwy(float* out, float* acc, const float* norm, float eps, size_t n) noexcept;

//==============================================================================
// 스칼라 참조 구현 함수들
//==============================================================================

void axpy_scalar(float* dst, const float* src, float g, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::fma(src[i], g, dst[i]);
    }
}

void axpy_windowed_scalar(float* dst, const float* src, const float* win, float g, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::fma(std::fma(src[i], win[i], 0.0f), g, dst[i]);
    }
}

void normalize_and_clear_scalar(float* out, float* acc, const float* norm, float eps, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        const float d = (norm[i] > eps) ? norm[i] : eps; // 무편향 가드
        out[i] = acc[i] / d;
        acc[i] = 0.0f;
    }
}

//==============================================================================
// 메인 API 함수들 - Highway 런타임 디스패치 사용
//==============================================================================

void axpy(float* dst, const float* src, float g, size_t n) noexcept {
    return axpy_hwy(dst, src, g, n);
}

void axpy_windowed(float* dst, const float* src, const float* win, float g, size_t n) noexcept {
    return axpy_windowed_hwy(dst, src, win, g, n);
}

void normalize_and_clear(float* out, float* acc, const float* norm, float eps, size_t n) noexcept {
    return normalize_and_clear_hwy(out, acc, norm, eps, n);
}

//==============================================================================
// 런타임 디스패치 디버깅 함수들
//==============================================================================

#include <cstdio>

const char* get_supported_targets() noexcept {
    // CPU 기능 탐지를 통한 지원 타겟 목록
    #if defined(__AVX512F__)
        return "AVX512,AVX2,SSE4,SCALAR";
    #elif defined(__AVX2__)
        return "AVX2,SSE4,SCALAR";
    #elif defined(__SSE4_1__)
        return "SSE4,SCALAR";
    #elif defined(__ARM_NEON)
        return "NEON,SCALAR";
    #else
        return "SCALAR";
    #endif
}

const char* get_current_target() noexcept {
    // 컴파일 시점에서 추정되는 최적 타겟
    #if defined(__AVX512F__)
        return "AVX512 (estimated)";
    #elif defined(__AVX2__)
        return "AVX2 (estimated)";
    #elif defined(__SSE4_1__)
        return "SSE4 (estimated)";
    #elif defined(__ARM_NEON)
        return "NEON (estimated)";
    #else
        return "SCALAR";
    #endif
}

size_t get_simd_lanes() noexcept {
    // CPU 기능에 따른 추정 lanes 수
    #if defined(__AVX512F__)
        return 16;  // AVX-512: 512bit / 32bit = 16 lanes
    #elif defined(__AVX2__)
        return 8;   // AVX2: 256bit / 32bit = 8 lanes
    #elif defined(__SSE4_1__)
        return 4;   // SSE4: 128bit / 32bit = 4 lanes
    #elif defined(__ARM_NEON)
        return 4;   // NEON: 128bit / 32bit = 4 lanes
    #else
        return 1;   // SCALAR: 1 lane
    #endif
}

void print_kernel_dispatch_info() noexcept {
    printf("=== Highway Runtime Dispatch Information ===\n");
    printf("Supported targets: %s\n", get_supported_targets());
    printf("Current target: %s\n", get_current_target());
    printf("SIMD lanes (float): %zu\n", get_simd_lanes());

    // 아키텍처 정보
    printf("Architecture info:\n");
    #if defined(__x86_64__) || defined(_M_X64)
        printf("  - x86-64 architecture\n");
    #elif defined(__i386__) || defined(_M_IX86)
        printf("  - x86-32 architecture\n");
    #elif defined(__aarch64__) || defined(_M_ARM64)
        printf("  - ARM64 architecture\n");
    #elif defined(__arm__) || defined(_M_ARM)
        printf("  - ARM32 architecture\n");
    #else
        printf("  - Unknown architecture\n");
    #endif

    // 컴파일러 최적화 정보
    printf("Compiler optimization flags:\n");
    #ifdef __OPTIMIZE__
        printf("  - Optimization enabled\n");
    #else
        printf("  - Debug build (optimization disabled)\n");
    #endif

    // 성능 예상
    printf("Expected performance characteristics:\n");
    const size_t lanes = get_simd_lanes();
    if (lanes >= 16) {
        printf("  - Excellent vectorization (16+ lanes)\n");
    } else if (lanes >= 8) {
        printf("  - High-performance vectorization (8 lanes)\n");
    } else if (lanes >= 4) {
        printf("  - Moderate vectorization (4 lanes)\n");
    } else {
        printf("  - Scalar execution (1 lane)\n");
    }

    printf("Note: This shows compile-time estimates.\n");
    printf("Highway's runtime dispatch may select different targets.\n");
    printf("=============================================\n");
}

} // namespace dsp