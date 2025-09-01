#pragma once

#include <cstddef>

namespace dsp {

/**
 * 최대 프레임 크기 상수 (Shadow Ring 구현용)
 * 일반적인 오디오 처리에서 사용되는 최대 프레임 크기를 정의합니다.
 */
constexpr size_t kMaxFrameSize = 16384;


/**
 * 커널 함수들 - Highway SIMD 최적화 포함
 * 런타임에 CPU가 지원하는 최적의 SIMD 구현을 자동 선택합니다.
 */

/**
 * AXPY 연산: dst[i] += src[i] * g for i in 0..n-1
 * Highway SIMD 최적화 적용 (런타임 디스패치)
 *
 * @param dst 목적지 배열
 * @param src 소스 배열
 * @param g 게인 계수
 * @param n 요소 수
 */
void axpy(float* dst, const float* src, float g, size_t n) noexcept;

/**
 * 윈도우 적용 AXPY 연산: dst[i] += src[i] * win[i] * g for i in 0..n-1
 * Highway SIMD 최적화 적용 (런타임 디스패치)
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
 * Highway SIMD 최적화 적용 (런타임 디스패치)
 * eps로 0으로 나누기 방지
 *
 * @param out 출력 배열
 * @param acc 누산 배열 (수정됨)
 * @param norm 정규화 계수 배열
 * @param eps 최소값 (0으로 나누기 방지)
 * @param n 요소 수
 */
void normalize_and_clear(float* out, float* acc, const float* norm, float eps, size_t n) noexcept;

/**
 * Highway SIMD 구현 함수들 (직접 호출 가능)
 * 테스트 및 벤치마크 목적으로 노출됩니다.
 */
void axpy_hwy(float* dst, const float* src, float g, size_t n) noexcept;
void axpy_windowed_hwy(float* dst, const float* src, const float* win, float g, size_t n) noexcept;
void normalize_and_clear_hwy(float* out, float* acc, const float* norm, float eps, size_t n) noexcept;

/**
 * 스칼라 참조 구현 함수들 (직접 호출 가능)
 * 테스트 및 벤치마크 목적으로 노출됩니다.
 */
void axpy_scalar(float* dst, const float* src, float g, size_t n) noexcept;
void axpy_windowed_scalar(float* dst, const float* src, const float* win, float g, size_t n) noexcept;
void normalize_and_clear_scalar(float* out, float* acc, const float* norm, float eps, size_t n) noexcept;

/**
 * 런타임 디스패치 디버깅 및 검증 함수들
 */

/**
 * 현재 시스템에서 사용 가능한 SIMD 타겟들을 문자열로 반환
 * @return 지원되는 SIMD 명령어 세트 목록 (예: "AVX3,AVX2,SSE4")
 */
const char* get_supported_targets() noexcept;

/**
 * 현재 Highway가 선택한 최적 타겟을 반환
 * @return 현재 사용 중인 SIMD 명령어 세트 (예: "AVX2")
 */
const char* get_current_target() noexcept;

/**
 * 각 커널 함수가 사용하는 SIMD 타겟 정보 출력 (디버그 모드에서만)
 * 표준 출력으로 현재 선택된 SIMD 구현 정보를 출력합니다.
 */
void print_kernel_dispatch_info() noexcept;

/**
 * SIMD lanes 수 정보를 반환 (현재 타겟 기준)
 * @return float 벡터의 lanes 수 (예: AVX2=8, AVX-512=16)
 */
size_t get_simd_lanes() noexcept;

} // namespace dsp