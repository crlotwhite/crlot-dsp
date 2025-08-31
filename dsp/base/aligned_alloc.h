#ifndef DSP_BASE_ALIGNED_ALLOC_H_
#define DSP_BASE_ALIGNED_ALLOC_H_

#include <cstddef>
#include <cstdlib>
#include <new>
#include <limits>

namespace dsp {
namespace base {

// Allocates memory for n elements of type T, aligned to 64 bytes.
// Throws std::bad_alloc on failure.
// Use DeallocateAligned to free the memory.
template <typename T>
T* AllocateAligned(size_t n) {
  if (n == 0) return nullptr;

  // Overflow check
  if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::bad_alloc();
  }

  size_t size = n * sizeof(T);
  void* ptr = nullptr;

#if defined(_MSC_VER)
  ptr = _aligned_malloc(size, 64);
  if (!ptr) throw std::bad_alloc();
#elif defined(_POSIX_VERSION)
  if (posix_memalign(&ptr, 64, size) != 0) {
    throw std::bad_alloc();
  }
#else
  // C++17 aligned_alloc: size must be multiple of alignment
  size_t size_aligned = ((size + 63) / 64) * 64;
  ptr = std::aligned_alloc(64, size_aligned);
  if (!ptr) throw std::bad_alloc();
#endif

  return static_cast<T*>(ptr);
}

// Deallocates memory allocated by AllocateAligned.
// ptr must be nullptr or a pointer returned by AllocateAligned.
void DeallocateAligned(void* ptr);

}  // namespace base
}  // namespace dsp

#endif  // DSP_BASE_ALIGNED_ALLOC_H_