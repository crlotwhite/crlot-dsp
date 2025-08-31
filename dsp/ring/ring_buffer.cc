#include "dsp/ring/ring_buffer.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include "dsp/base/aligned_alloc.h"

namespace dsp {
namespace ring {

template <typename T>
RingBuffer<T>::RingBuffer(size_t capacity, bool shadow)
    : capacity_(capacity), write_pos_(0), shadow_(shadow) {
  // Precondition: capacity must be > 0 to avoid division by zero
  if (capacity == 0) {
    throw std::invalid_argument("RingBuffer capacity must be > 0");
  }

  // Allocate 64-byte aligned memory
  buffer_ = base::AllocateAligned<T>(capacity_);
}

template <typename T>
RingBuffer<T>::~RingBuffer() {
  if (buffer_) {
    base::DeallocateAligned(buffer_);
    buffer_ = nullptr;
  }
}

template <typename T>
std::pair<base::Span<T>, base::Span<T>> RingBuffer<T>::split(size_t start, size_t len) noexcept {
  // Handle empty request
  if (len == 0) {
    return {base::Span<T>(), base::Span<T>()};
  }

  // Overflow protection
  if (len > std::numeric_limits<size_t>::max() - start) {
    len = std::numeric_limits<size_t>::max() - start;
  }

  // Clamp len to capacity for safety
  if (len > capacity_) {
    len = capacity_;
  }

  // Normalize start position to be within bounds
  start = start % capacity_;

#ifndef NDEBUG
  // Debug boundary checks
  assert(start < capacity_);
  assert(len <= capacity_);
#endif

  // Calculate the end position
  size_t end = start + len;

  // Case 1: No wrap-around - the entire range fits in one contiguous block
  if (end <= capacity_) {
    return {base::Span<T>(buffer_ + start, len), base::Span<T>()};
  }

  // Case 2: Wrap-around - split into two spans
  size_t first_len = capacity_ - start;
  size_t second_len = len - first_len;

  base::Span<T> first_span(buffer_ + start, first_len);
  base::Span<T> second_span(buffer_, second_len);

  return {first_span, second_span};
}

template <typename T>
std::pair<base::Span<const T>, base::Span<const T>> RingBuffer<T>::split(size_t start, size_t len) const noexcept {
  // Handle empty request
  if (len == 0) {
    return {base::Span<const T>(), base::Span<const T>()};
  }

  // Overflow protection
  if (len > std::numeric_limits<size_t>::max() - start) {
    len = std::numeric_limits<size_t>::max() - start;
  }

  // Clamp len to capacity for safety
  if (len > capacity_) {
    len = capacity_;
  }

  // Normalize start position to be within bounds
  start = start % capacity_;

#ifndef NDEBUG
  // Debug boundary checks
  assert(start < capacity_);
  assert(len <= capacity_);
#endif

  // Calculate the end position
  size_t end = start + len;

  // Case 1: No wrap-around - the entire range fits in one contiguous block
  if (end <= capacity_) {
    return {base::Span<const T>(buffer_ + start, len), base::Span<const T>()};
  }

  // Case 2: Wrap-around - split into two spans
  size_t first_len = capacity_ - start;
  size_t second_len = len - first_len;

  base::Span<const T> first_span(buffer_ + start, first_len);
  base::Span<const T> second_span(buffer_, second_len);

  return {first_span, second_span};
}

template <typename T>
void RingBuffer<T>::shadow_sync(size_t bytes) {
  // TODO: Implement in PR 7 - synchronize frame_size bytes after frame accumulation
  (void)bytes;  // Suppress unused parameter warning
}

// Explicit template instantiations for common types
template class RingBuffer<float>;
template class RingBuffer<double>;
template class RingBuffer<int16_t>;
template class RingBuffer<int32_t>;
template class RingBuffer<uint16_t>;
template class RingBuffer<uint32_t>;

}  // namespace ring
}  // namespace dsp