#include "dsp/ring/ring_buffer.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <cstring>
#include "dsp/base/aligned_alloc.h"
#include "dsp/ola/kernels.h"

namespace dsp {
namespace ring {

template <typename T>
RingBuffer<T>::RingBuffer(size_t capacity, bool shadow)
    : capacity_(capacity), write_pos_(0), shadow_(shadow) {
  // Shadow ring requires trivially copyable types for memcpy
  static_assert(std::is_trivially_copyable<T>::value, "Shadow ring requires trivially copyable T");

  // Precondition: capacity must be > 0 to avoid division by zero
  if (capacity == 0) {
    throw std::invalid_argument("RingBuffer capacity must be > 0");
  }

  // Shadow buffer일 경우 2*capacity로 물리 버퍼 할당 (소프트웨어 미러 전략)
  size_t alloc_capacity = shadow_ ? (capacity_ * 2) : capacity_;

  // Allocate 64-byte aligned memory
  buffer_ = base::AllocateAligned<T>(alloc_capacity);

  // Initialize entire physical buffer to zero
  std::fill(buffer_, buffer_ + alloc_capacity, T{});
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
  if (!shadow_ || bytes == 0) {
    return;  // Shadow 모드가 아니거나 복사할 데이터가 없으면 아무것도 하지 않음
  }

  // 헤드 영역(0..bytes-1)을 미러 영역(capacity_..capacity_+bytes-1)으로 복사
  // bytes는 요소 개수 단위
  const size_t to_copy = std::min(bytes, capacity_);
  std::memcpy(buffer_ + capacity_, buffer_, to_copy * sizeof(T));
}

template <typename T>
T* RingBuffer<T>::contiguous_read_ptr(size_t read_pos) noexcept {
  if (!shadow_) {
    // Shadow 모드가 아니면 기존 방식 사용
    return buffer_ + (read_pos % capacity_);
  }

  // Shadow 모드에서는 미러 영역을 활용하여 항상 연속 메모리 보장
  return buffer_ + read_pos;
}

template <typename T>
const T* RingBuffer<T>::contiguous_read_ptr(size_t read_pos) const noexcept {
  if (!shadow_) {
    // Shadow 모드가 아니면 기존 방식 사용
    return buffer_ + (read_pos % capacity_);
  }

  // Shadow 모드에서는 미러 영역을 활용하여 항상 연속 메모리 보장
  return buffer_ + read_pos;
}

template <typename T>
size_t RingBuffer<T>::write(const T* src, size_t n) {
  if (src == nullptr || n == 0) {
    return 0;
  }

  size_t written = 0;
  size_t tail = capacity_ - write_pos_;

  if (n <= tail) {
    // 테일에 모두 쓸 수 있음
    std::memcpy(buffer_ + write_pos_, src, n * sizeof(T));
    write_pos_ += n;
    if (write_pos_ == capacity_) {
      write_pos_ = 0;  // 정확히 capacity만큼 쓴 경우
    }
    written = n;
  } else {
    // wrap 발생
    // 1. 테일에 먼저 쓰기
    std::memcpy(buffer_ + write_pos_, src, tail * sizeof(T));
    written += tail;

    // 2. 헤드에 나머지 쓰기
    size_t head = n - tail;
    if (head > 0) {
      std::memcpy(buffer_, src + tail, head * sizeof(T));
      written += head;
    }

    // 3. write_pos_ 업데이트
    write_pos_ = head;

    // 4. Shadow 모드일 때 미러 동기화
    if (shadow_) {
      shadow_sync(head);
    }
  }

  return written;
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