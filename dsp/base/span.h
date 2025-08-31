#ifndef DSP_BASE_SPAN_H_
#define DSP_BASE_SPAN_H_

#include <cstddef>

namespace dsp {
namespace base {

// A minimal Span class representing a contiguous range of elements.
// Provides a safe way to pass around pointers with size information.
template <typename T>
class Span {
 public:
  Span() : ptr_(nullptr), len_(0) {}
  Span(T* ptr, size_t len) : ptr_(ptr), len_(len) {}

  T* data() const { return ptr_; }
  size_t size() const { return len_; }
  bool empty() const { return len_ == 0; }

  T& operator[](size_t index) const {
    // Note: No bounds checking for performance, assume valid usage.
    return ptr_[index];
  }

  T* begin() const { return ptr_; }
  T* end() const { return ptr_ + len_; }

 private:
  T* ptr_;
  size_t len_;
};

}  // namespace base
}  // namespace dsp

#endif  // DSP_BASE_SPAN_H_