#pragma once

#include <cstddef>
#include <utility>
#include <cstdint>
#include "dsp/base/span.h"

namespace dsp {
namespace ring {

/**
 * RingBuffer - A circular buffer with 64-byte aligned memory allocation.
 *
 * Provides efficient circular indexing and the ability to split contiguous
 * ranges into up to two spans for zero-copy access.
 */
template <typename T>
class RingBuffer {
 public:
  /**
   * Constructor - Creates a ring buffer with the specified capacity.
   *
   * @param capacity The maximum number of elements the buffer can hold (must be > 0)
   * @param shadow Whether this is a shadow buffer (for future use)
   */
  RingBuffer(size_t capacity, bool shadow = false);

  /**
   * Destructor - Frees the allocated memory.
   */
  ~RingBuffer();

  // Disable copy and move operations for simplicity
  RingBuffer(const RingBuffer&) = delete;
  RingBuffer& operator=(const RingBuffer&) = delete;
  RingBuffer(RingBuffer&&) = delete;
  RingBuffer& operator=(RingBuffer&&) = delete;

  /**
   * Returns the capacity of the buffer.
   */
  size_t capacity() const noexcept { return capacity_; }

  /**
   * Returns the current write position (for future use).
   */
  size_t write_pos() const noexcept { return write_pos_; }

  /**
   * Splits a contiguous range starting from 'start' with length 'len' into
   * up to two spans. This handles wrap-around in the circular buffer.
   *
   * @param start The starting position in the buffer
   * @param len The length of the range to split, clamped to capacity
   * @return A pair of spans: first span is always valid, second may be empty
   */
  [[nodiscard]] std::pair<base::Span<T>, base::Span<T>> split(size_t start, size_t len) noexcept;

  /**
   * Const version of split that returns spans to const data.
   */
  [[nodiscard]] std::pair<base::Span<const T>, base::Span<const T>> split(size_t start, size_t len) const noexcept;

  /**
   * Shadow synchronization method (to be implemented in PR 7).
   * This will synchronize frame_size bytes after frame accumulation.
   *
   * @param bytes Number of bytes to synchronize
   */
  void shadow_sync(size_t bytes);

  /**
   * Direct access to the underlying buffer (for internal use).
   */
  T* data() noexcept { return buffer_; }
  const T* data() const noexcept { return buffer_; }

 private:
  T* buffer_;           // 64-byte aligned buffer
  size_t capacity_;     // Total capacity
  size_t write_pos_;    // Current write position (for future use)
  bool shadow_;         // Shadow buffer flag
};

}  // namespace ring
}  // namespace dsp