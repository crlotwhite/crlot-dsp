#pragma once

#include <cstddef>

namespace dsp {
namespace ola {

/**
 * COLA 정규화 계수 선형 누적 빌더
 *
 * 기존의 비효율적인 3중 루프 구조를 제거하고,
 * 산술 split 계산을 활용하여 O(K·N) 시간 복잡도로
 * 정확한 COLA 정규화를 프리컴퓨트합니다.
 *
 * K = ceil((ring_len + frame_size - 1)/hop) + 1
 * 이렇게 하면 ring 전체를 덮는 프레임 시작점들로 누적하여
 * 정확한 COLA 조건을 만족합니다.
 *
 * 빌더는 물리적 합만 계산하며, ε 가드는 소비 단계에서 수행.
 *
 * @param norm 출력 정규화 계수 버퍼 [ring_len]
 * @param window 입력 윈도우 함수 [frame_size]
 * @param ring_len 링 버퍼 크기
 * @param frame_size 프레임 크기 N
 * @param hop 홉 크기 H
 */
void build_norm_linear(float* norm, const float* window,
                      size_t ring_len, size_t frame_size, size_t hop);

}  // namespace ola
}  // namespace dsp