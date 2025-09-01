#include <algorithm>
#include <cmath>

// Highway 헤더들
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dsp/ola/kernels_hwy.cc"  // 이 파일 자체
#include "hwy/foreach_target.h"  // 모든 타겟에 대한 코드 생성
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace dsp {
namespace HWY_NAMESPACE {  // 각 타겟별로 다른 네임스페이스

namespace hn = hwy::HWY_NAMESPACE;

/**
 * AXPY 연산의 Highway SIMD 구현: dst[i] += src[i] * g
 */
void AxpyImpl(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src, float g, size_t n) {
    const hn::ScalableTag<float> d;
    const auto vg = hn::Set(d, g);

    const size_t N = hn::Lanes(d);
    size_t i = 0;

    // 메인 벡터화된 루프 - SIMD lanes 단위로 처리
    for (; i + N <= n; i += N) {
        const auto vsrc = hn::LoadU(d, src + i);
        const auto vdst = hn::LoadU(d, dst + i);
        const auto result = hn::MulAdd(vsrc, vg, vdst);  // dst + src * g
        hn::StoreU(result, d, dst + i);
    }

    // 나머지 요소들을 스칼라로 처리
    for (; i < n; ++i) {
        dst[i] = std::fma(src[i], g, dst[i]);
    }
}

/**
 * 윈도우 적용 AXPY 연산의 Highway SIMD 구현: dst[i] += src[i] * win[i] * g
 */
void AxpyWindowedImpl(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                      const float* HWY_RESTRICT win, float g, size_t n) {
    const hn::ScalableTag<float> d;
    const auto vg = hn::Set(d, g);

    const size_t N = hn::Lanes(d);
    size_t i = 0;

    // 메인 벡터화된 루프
    for (; i + N <= n; i += N) {
        const auto vsrc = hn::LoadU(d, src + i);
        const auto vwin = hn::LoadU(d, win + i);
        const auto vdst = hn::LoadU(d, dst + i);

        // src * win * g + dst 계산
        const auto src_win = hn::Mul(vsrc, vwin);
        const auto result = hn::MulAdd(src_win, vg, vdst);

        hn::StoreU(result, d, dst + i);
    }

    // 나머지 요소들을 스칼라로 처리
    for (; i < n; ++i) {
        dst[i] = std::fma(std::fma(src[i], win[i], 0.0f), g, dst[i]);
    }
}

/**
 * 정규화 및 클리어의 Highway SIMD 구현: out[i] = acc[i] / norm[i], acc[i] = 0
 */
void NormalizeAndClearImpl(float* HWY_RESTRICT out, float* HWY_RESTRICT acc,
                          const float* HWY_RESTRICT norm, float eps, size_t n) {
    const hn::ScalableTag<float> d;
    const auto veps = hn::Set(d, eps);
    const auto vzero = hn::Zero(d);

    const size_t N = hn::Lanes(d);
    size_t i = 0;

    // 메인 벡터화된 루프
    for (; i + N <= n; i += N) {
        const auto vacc = hn::LoadU(d, acc + i);
        const auto vnorm = hn::LoadU(d, norm + i);

        // norm[i] > eps ? norm[i] : eps (0으로 나누기 방지)
        const auto vdenom = hn::Max(vnorm, veps);

        // out[i] = acc[i] / denom
        const auto vout = hn::Div(vacc, vdenom);
        hn::StoreU(vout, d, out + i);

        // acc[i] = 0
        hn::StoreU(vzero, d, acc + i);
    }

    // 나머지 요소들을 스칼라로 처리
    for (; i < n; ++i) {
        const float d = (norm[i] > eps) ? norm[i] : eps;
        out[i] = acc[i] / d;
        acc[i] = 0.0f;
    }
}

// Highway 매크로로 함수들을 내보냄
}  // namespace HWY_NAMESPACE
}  // namespace dsp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dsp {

// 각 함수에 대한 HWY_EXPORT 선언
HWY_EXPORT(AxpyImpl);
HWY_EXPORT(AxpyWindowedImpl);
HWY_EXPORT(NormalizeAndClearImpl);

// 런타임 디스패치 래퍼 함수들
void axpy_hwy(float* dst, const float* src, float g, size_t n) noexcept {
    return HWY_DYNAMIC_DISPATCH(AxpyImpl)(dst, src, g, n);
}

void axpy_windowed_hwy(float* dst, const float* src, const float* win, float g, size_t n) noexcept {
    return HWY_DYNAMIC_DISPATCH(AxpyWindowedImpl)(dst, src, win, g, n);
}

void normalize_and_clear_hwy(float* out, float* acc, const float* norm, float eps, size_t n) noexcept {
    return HWY_DYNAMIC_DISPATCH(NormalizeAndClearImpl)(out, acc, norm, eps, n);
}

// 디버깅 함수들은 kernels.cc에서 구현됩니다

}  // namespace dsp

#endif  // HWY_ONCE