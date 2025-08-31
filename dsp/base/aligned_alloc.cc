#include "dsp/base/aligned_alloc.h"

#include <cstddef>
#include <cstdlib>
#include <new>

namespace dsp {
namespace base {

void DeallocateAligned(void* ptr) {
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

}  // namespace base
}  // namespace dsp