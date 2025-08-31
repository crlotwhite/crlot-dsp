#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "dsp/ring/ring_buffer.h"

namespace dsp {
namespace ring {
namespace {

TEST(RingBufferTest, Constructor) {
  RingBuffer<float> rb(1024);
  EXPECT_EQ(rb.capacity(), 1024);
  EXPECT_EQ(rb.write_pos(), 0);
}

TEST(RingBufferTest, ConstructorWithShadow) {
  RingBuffer<float> rb(512, true);
  EXPECT_EQ(rb.capacity(), 512);
}

TEST(RingBufferTest, SplitNoWrapAround) {
  RingBuffer<int> rb(10);

  // Initialize buffer with values 0-9
  for (size_t i = 0; i < 10; ++i) {
    rb.data()[i] = static_cast<int>(i);
  }

  // Split from start=2, len=5 (no wrap-around)
  auto [span1, span2] = rb.split(2, 5);

  EXPECT_EQ(span1.size(), 5);
  EXPECT_TRUE(span2.empty());

  // Verify content
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(span1[i], static_cast<int>(i + 2));
  }
}

TEST(RingBufferTest, SplitWithWrapAround) {
  RingBuffer<int> rb(10);

  // Initialize buffer with values 0-9
  for (size_t i = 0; i < 10; ++i) {
    rb.data()[i] = static_cast<int>(i);
  }

  // Split from start=7, len=5 (wraps around: 7,8,9,0,1)
  auto [span1, span2] = rb.split(7, 5);

  EXPECT_EQ(span1.size(), 3);  // 7,8,9
  EXPECT_EQ(span2.size(), 2);  // 0,1

  // Verify first span
  EXPECT_EQ(span1[0], 7);
  EXPECT_EQ(span1[1], 8);
  EXPECT_EQ(span1[2], 9);

  // Verify second span
  EXPECT_EQ(span2[0], 0);
  EXPECT_EQ(span2[1], 1);
}

TEST(RingBufferTest, SplitEmpty) {
  RingBuffer<int> rb(10);
  auto [span1, span2] = rb.split(5, 0);

  EXPECT_TRUE(span1.empty());
  EXPECT_TRUE(span2.empty());
}

TEST(RingBufferTest, SplitFullBuffer) {
  RingBuffer<int> rb(5);

  // Initialize buffer
  for (size_t i = 0; i < 5; ++i) {
    rb.data()[i] = static_cast<int>(i);
  }

  // Split entire buffer from start=0
  auto [span1, span2] = rb.split(0, 5);

  EXPECT_EQ(span1.size(), 5);
  EXPECT_TRUE(span2.empty());

  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(span1[i], static_cast<int>(i));
  }
}

TEST(RingBufferTest, SplitWrapAroundFullBuffer) {
  RingBuffer<int> rb(5);

  // Initialize buffer
  for (size_t i = 0; i < 5; ++i) {
    rb.data()[i] = static_cast<int>(i);
  }

  // Split from start=3, len=5 (wraps: 3,4,0,1,2)
  auto [span1, span2] = rb.split(3, 5);

  EXPECT_EQ(span1.size(), 2);  // 3,4
  EXPECT_EQ(span2.size(), 3);  // 0,1,2

  EXPECT_EQ(span1[0], 3);
  EXPECT_EQ(span1[1], 4);
  EXPECT_EQ(span2[0], 0);
  EXPECT_EQ(span2[1], 1);
  EXPECT_EQ(span2[2], 2);
}

TEST(RingBufferTest, SplitStartBeyondCapacity) {
  RingBuffer<int> rb(5);

  // Initialize buffer
  for (size_t i = 0; i < 5; ++i) {
    rb.data()[i] = static_cast<int>(i);
  }

  // Start at 7 (7 % 5 = 2), len=3
  auto [span1, span2] = rb.split(7, 3);

  EXPECT_EQ(span1.size(), 3);
  EXPECT_TRUE(span2.empty());

  EXPECT_EQ(span1[0], 2);
  EXPECT_EQ(span1[1], 3);
  EXPECT_EQ(span1[2], 4);
}

TEST(RingBufferTest, SplitSpansConcatenation) {
  RingBuffer<int> rb(8);

  // Initialize buffer with recognizable pattern
  for (size_t i = 0; i < 8; ++i) {
    rb.data()[i] = static_cast<int>(i * 10);
  }

  // Test various split scenarios and verify concatenation
  std::vector<std::pair<size_t, size_t>> test_cases = {
      {0, 3},   // No wrap: 0,1,2
      {5, 4},   // Wrap: 5,6,7,0
      {6, 5},   // Wrap: 6,7,0,1,2
      {2, 8},   // Full wrap: 2,3,4,5,6,7,0,1
  };

  for (const auto& [start, len] : test_cases) {
    auto [span1, span2] = rb.split(start, len);

    // Collect all values from spans
    std::vector<int> concatenated;
    for (size_t i = 0; i < span1.size(); ++i) {
      concatenated.push_back(span1[i]);
    }
    for (size_t i = 0; i < span2.size(); ++i) {
      concatenated.push_back(span2[i]);
    }

    // Verify total length
    EXPECT_EQ(concatenated.size(), len);

    // Verify content matches expected circular buffer content
    for (size_t i = 0; i < len; ++i) {
      size_t buffer_idx = (start + i) % 8;
      EXPECT_EQ(concatenated[i], static_cast<int>(buffer_idx * 10));
    }
  }
}

TEST(RingBufferTest, ShadowSyncStub) {
  RingBuffer<float> rb(100);
  // Just verify it doesn't crash
  rb.shadow_sync(50);
}

TEST(RingBufferTest, ConstSplitReturnsConstSpans) {
  RingBuffer<int> rb(10);

  // Initialize buffer
  for (size_t i = 0; i < 10; ++i) {
    rb.data()[i] = static_cast<int>(i);
  }

  // Test const split
  const RingBuffer<int>& const_rb = rb;
  auto [span1, span2] = const_rb.split(2, 5);

  // Verify types are const
  static_assert(std::is_same_v<decltype(span1), base::Span<const int>>);
  static_assert(std::is_same_v<decltype(span2), base::Span<const int>>);

  // Verify content
  EXPECT_EQ(span1.size(), 5);
  EXPECT_TRUE(span2.empty());
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(span1[i], static_cast<int>(i + 2));
  }
}

TEST(RingBufferTest, OverflowProtection) {
  RingBuffer<int> rb(10);

  // Test with very large len
  size_t large_len = std::numeric_limits<size_t>::max();
  auto [span1, span2] = rb.split(0, large_len);

  // Should be clamped to capacity
  EXPECT_EQ(span1.size() + span2.size(), 10);
}

TEST(RingBufferTest, ZeroCapacityThrows) {
  EXPECT_THROW(RingBuffer<int>(0), std::invalid_argument);
}

TEST(RingBufferTest, LargeLenClamping) {
  RingBuffer<int> rb(5);

  // Initialize buffer
  for (size_t i = 0; i < 5; ++i) {
    rb.data()[i] = static_cast<int>(i);
  }

  // Request len > capacity
  auto [span1, span2] = rb.split(0, 10);

  // Should be clamped to capacity
  EXPECT_EQ(span1.size() + span2.size(), 5);
  EXPECT_EQ(span1.size(), 5);
  EXPECT_TRUE(span2.empty());
}

TEST(RingBufferTest, NoexceptMethods) {
  RingBuffer<int> rb(10);

  // These should all be noexcept
  static_assert(noexcept(rb.capacity()));
  static_assert(noexcept(rb.write_pos()));
  static_assert(noexcept(rb.data()));
  static_assert(noexcept(std::as_const(rb).data()));
}

}  // namespace
}  // namespace ring
}  // namespace dsp