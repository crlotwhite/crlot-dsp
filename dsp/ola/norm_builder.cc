#include "norm_builder.h"
#include <algorithm>
#include <utility>

namespace dsp {
namespace ola {

void build_norm_linear(float* norm, const float* window,
                       size_t ring_len, size_t frame_size, size_t hop) {
    // 정규화 버퍼 초기화
    std::fill(norm, norm + ring_len, 0.0f);

    // 전제 체크
    if (ring_len == 0 || frame_size == 0 || hop == 0) return;

    // 안전 전제(옵션): frame_size <= ring_len 가정
    // if (frame_size > ring_len) frame_size = ring_len; // 또는 assert

    // 연속 두 구간 분할을 위한 보조 함수 (음수 처리 추가)
    auto split_span = [&](int64_t start, size_t len) -> std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>> {
       // 음수 모듈로 처리
       if (start < 0) {
           start = static_cast<int64_t>(ring_len) + (start % static_cast<int64_t>(ring_len));
           if (start < 0) start += ring_len;
       }
       size_t s = start % ring_len;
       const size_t first  = std::min(len, ring_len - s);
       const size_t second = len - first;
       return std::make_pair(std::make_pair(s, first),
                             std::make_pair(size_t{0}, second));
    };

    // 전 범위를 덮도록 음수 방향도 포함 (스칼라 버전과 일치하도록)
    const size_t N = frame_size, H = hop;
    const auto floor_div = [](int64_t a, size_t b){ return a < 0 ? (a - static_cast<int64_t>(b) + 1) / static_cast<int64_t>(b) : a / static_cast<int64_t>(b); };
    const int64_t K_start = floor_div(-static_cast<int64_t>(N), H);
    const int64_t K_end = static_cast<int64_t>((static_cast<int64_t>(ring_len) + N - 1 + H - 1) / H);

    for (int64_t k = K_start; k <= K_end; ++k) {
       const int64_t s = k * static_cast<int64_t>(H);
       auto [A, B] = split_span(s, N);

       // A 구간 누적: norm[s ... s+A.second) += window[0 ... A.second)
       for (size_t i = 0; i < A.second; ++i)
       norm[A.first + i] += window[i];

       // B 구간 누적: norm[0 ... B.second) += window[A.second ... N)
       for (size_t i = 0; i < B.second; ++i)
       norm[i] += window[A.second + i];
    }
    // eps 가드는 소비 단계(normalize_and_clear)에서 max(norm, eps)로 처리
   }

}  // namespace ola
}  // namespace dsp