#include "kernels.h"

#include <algorithm>
#include <cmath>  // for std::fma

namespace dsp {

void axpy(float* dst, const float* src, float g, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::fma(src[i], g, dst[i]);
    }
}

void axpy_windowed(float* dst, const float* src, const float* win, float g, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::fma(std::fma(src[i], win[i], 0.0f), g, dst[i]);
    }
}

void normalize_and_clear(float* out, float* acc, const float* norm, float eps, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        const float d = (norm[i] > eps) ? norm[i] : eps; // 무편향 가드
        out[i] = acc[i] / d;
        acc[i] = 0.0f;
    }
}

} // namespace dsp