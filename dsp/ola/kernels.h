#pragma once

#include <cstddef>

namespace dsp {

/**
 * 스칼라 커널 함수들 - SIMD 최적화 전 참조 구현
 */

/**
 * AXPY 연산: dst[i] += src[i] * g for i in 0..n-1
 *
 * @param dst 목적지 배열
 * @param src 소스 배열
 * @param g 게인 계수
 * @param n 요소 수
 */
void axpy(float* dst, const float* src, float g, size_t n) noexcept;

/**
 * 윈도우 적용 AXPY 연산: dst[i] += src[i] * win[i] * g for i in 0..n-1
 *
 * @param dst 목적지 배열
 * @param src 소스 배열
 * @param win 윈도우 배열
 * @param g 게인 계수
 * @param n 요소 수
 */
void axpy_windowed(float* dst, const float* src, const float* win, float g, size_t n) noexcept;

/**
 * 정규화 및 클리어: out[i] = acc[i] / norm[i], acc[i] = 0 for i in 0..n-1
 * eps로 0으로 나누기 방지
 *
 * @param out 출력 배열
 * @param acc 누산 배열 (수정됨)
 * @param norm 정규화 계수 배열
 * @param eps 최소값 (0으로 나누기 방지)
 * @param n 요소 수
 */
void normalize_and_clear(float* out, float* acc, const float* norm, float eps, size_t n) noexcept;

} // namespace dsp