#pragma once

#include <cstddef>

namespace dsp {
namespace ola {

/**
 * 인터리브된 AoS 데이터를 SoA 형식으로 디인터리브하여 scratch 버퍼에 저장
 *
 * @param interleaved 인터리브된 입력 데이터 [channels * n]
 *                   형식: [ch0_s0, ch1_s0, ..., ch0_s1, ch1_s1, ...]
 * @param n 샘플 수 (각 채널당)
 * @param channels 채널 수
 * @param scratch 출력 scratch 버퍼 [channels * n]
 *               형식: [ch0_s0, ch0_s1, ..., ch1_s0, ch1_s1, ...]
 */
void deinterleave_to_scratch(const float* interleaved, size_t n, size_t channels, float* scratch);

} // namespace ola
} // namespace dsp