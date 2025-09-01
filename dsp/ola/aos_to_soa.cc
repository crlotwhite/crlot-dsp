#include "aos_to_soa.h"
#include <cstring>

namespace dsp {
namespace ola {

void deinterleave_to_scratch(const float* interleaved, size_t n, size_t channels, float* scratch) {
    if (interleaved == nullptr || scratch == nullptr || n == 0 || channels == 0) {
        return;
    }

    // 최적화된 디인터리브: 채널별로 연속 복사
    for (size_t ch = 0; ch < channels; ++ch) {
        for (size_t i = 0; i < n; ++i) {
            scratch[ch * n + i] = interleaved[i * channels + ch];
        }
    }
}

} // namespace ola
} // namespace dsp