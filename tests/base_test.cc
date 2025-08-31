#include <gtest/gtest.h>
#include <limits>

#include "dsp/base/aligned_alloc.h"
#include "dsp/base/span.h"

namespace dsp {
namespace base {
namespace {

TEST(AlignedAllocTest, AllocateAndDeallocate) {
  int* ptr = AllocateAligned<int>(10);
  ASSERT_NE(ptr, nullptr);
  // Check alignment
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
  DeallocateAligned(ptr);
}

TEST(AlignedAllocTest, AllocateZero) {
  int* ptr = AllocateAligned<int>(0);
  EXPECT_EQ(ptr, nullptr);
  DeallocateAligned(ptr);  // Should be safe
}

TEST(AlignedAllocTest, AllocateLarge) {
  const size_t large_n = 1000000;
  double* ptr = AllocateAligned<double>(large_n);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
  // Initialize and check
  for (size_t i = 0; i < large_n; ++i) {
    ptr[i] = static_cast<double>(i);
  }
  EXPECT_EQ(ptr[0], 0.0);
  EXPECT_EQ(ptr[large_n - 1], static_cast<double>(large_n - 1));
  DeallocateAligned(ptr);
}

TEST(AlignedAllocTest, OverflowProtection) {
  // Test overflow protection for large n
  const size_t max_n = std::numeric_limits<size_t>::max() / sizeof(int) + 1;
  EXPECT_THROW(AllocateAligned<int>(max_n), std::bad_alloc);
}

TEST(SpanTest, DefaultConstructor) {
  Span<int> span;
  EXPECT_EQ(span.data(), nullptr);
  EXPECT_EQ(span.size(), 0);
  EXPECT_TRUE(span.empty());
}

TEST(SpanTest, ConstructorWithData) {
  int arr[5] = {1, 2, 3, 4, 5};
  Span<int> span(arr, 5);
  EXPECT_EQ(span.data(), arr);
  EXPECT_EQ(span.size(), 5);
  EXPECT_FALSE(span.empty());
}

TEST(SpanTest, AccessOperator) {
  int arr[3] = {10, 20, 30};
  Span<int> span(arr, 3);
  EXPECT_EQ(span[0], 10);
  EXPECT_EQ(span[1], 20);
  EXPECT_EQ(span[2], 30);
}

TEST(SpanTest, Iterators) {
  int arr[4] = {1, 2, 3, 4};
  Span<int> span(arr, 4);
  auto it = span.begin();
  EXPECT_EQ(*it, 1);
  ++it;
  EXPECT_EQ(*it, 2);
  EXPECT_EQ(span.end() - span.begin(), 4);
}

TEST(SpanTest, EmptySpan) {
  Span<double> span(nullptr, 0);
  EXPECT_TRUE(span.empty());
  EXPECT_EQ(span.size(), 0);
  EXPECT_EQ(span.begin(), span.end());
}

}  // namespace
}  // namespace base
}  // namespace dsp