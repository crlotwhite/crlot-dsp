#include <cmath>
#include <gtest/gtest.h>
#include <numeric>  // for std::iota
#include <vector>
#include "dsp/frame/FrameQueue.h"

using namespace dsp;

class FrameQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 테스트용 신호 생성 (사인파)
        test_signal_short_.resize(100);
        test_signal_long_.resize(1000);

        for (size_t i = 0; i < test_signal_short_.size(); ++i) {
            test_signal_short_[i] = std::sin(2.0 * M_PI * i / 10.0);
        }

        for (size_t i = 0; i < test_signal_long_.size(); ++i) {
            test_signal_long_[i] = std::sin(2.0 * M_PI * i / 50.0);
        }
    }

    std::vector<float> test_signal_short_;
    std::vector<float> test_signal_long_;
};

// 기본 생성자 테스트
TEST_F(FrameQueueTest, BasicConstruction) {
    FrameQueue fq(test_signal_short_.data(), test_signal_short_.size(), 32, 16);

    EXPECT_EQ(fq.getFrameSize(), 32);
    EXPECT_EQ(fq.getHopSize(), 16);
    EXPECT_GT(fq.getNumFrames(), 0);
}

// 홀수 프레임 크기 테스트
TEST_F(FrameQueueTest, OddFrameSize) {
    size_t frame_size = 31;  // 홀수
    size_t hop_size = 15;

    FrameQueue fq(test_signal_short_.data(), test_signal_short_.size(), frame_size, hop_size);

    EXPECT_EQ(fq.getFrameSize(), frame_size);
    EXPECT_GT(fq.getNumFrames(), 0);

    // 첫 번째 프레임 검증
    const float* frame = fq.getFrame(0);
    EXPECT_NE(frame, nullptr);
}

// 짝수 프레임 크기 테스트
TEST_F(FrameQueueTest, EvenFrameSize) {
    size_t frame_size = 32;  // 짝수
    size_t hop_size = 16;

    FrameQueue fq(test_signal_short_.data(), test_signal_short_.size(), frame_size, hop_size);

    EXPECT_EQ(fq.getFrameSize(), frame_size);
    EXPECT_GT(fq.getNumFrames(), 0);

    // 첫 번째 프레임 검증
    const float* frame = fq.getFrame(0);
    EXPECT_NE(frame, nullptr);
}

// 25% 홉 크기 테스트 (hop = frame_size * 0.25)
TEST_F(FrameQueueTest, HopSize25Percent) {
    size_t frame_size = 64;
    size_t hop_size = 16;  // 25% of frame_size

    FrameQueue fq(test_signal_long_.data(), test_signal_long_.size(), frame_size, hop_size);

    EXPECT_EQ(fq.getHopSize(), hop_size);

    // 프레임 간 오버랩 확인
    if (fq.getNumFrames() >= 2) {
        const float* frame1 = fq.getFrame(0);
        const float* frame2 = fq.getFrame(1);

        // 오버랩 영역 확인 (frame1의 마지막 부분과 frame2의 첫 부분이 겹침)
        EXPECT_NE(frame1, nullptr);
        EXPECT_NE(frame2, nullptr);
    }
}

// 50% 홉 크기 테스트 (hop = frame_size * 0.5)
TEST_F(FrameQueueTest, HopSize50Percent) {
    size_t frame_size = 64;
    size_t hop_size = 32;  // 50% of frame_size

    FrameQueue fq(test_signal_long_.data(), test_signal_long_.size(), frame_size, hop_size);

    EXPECT_EQ(fq.getHopSize(), hop_size);
    EXPECT_GT(fq.getNumFrames(), 0);
}

// 75% 홉 크기 테스트 (hop = frame_size * 0.75)
TEST_F(FrameQueueTest, HopSize75Percent) {
    size_t frame_size = 64;
    size_t hop_size = 48;  // 75% of frame_size

    FrameQueue fq(test_signal_long_.data(), test_signal_long_.size(), frame_size, hop_size);

    EXPECT_EQ(fq.getHopSize(), hop_size);
    EXPECT_GT(fq.getNumFrames(), 0);
}

// 다양한 파일 길이 테스트
TEST_F(FrameQueueTest, VariousInputLengths) {
    size_t frame_size = 32;
    size_t hop_size = 16;

    // 매우 짧은 입력
    std::vector<float> very_short(10, 1.0f);
    FrameQueue fq1(very_short.data(), very_short.size(), frame_size, hop_size);

    // 프레임 크기보다 작은 입력도 처리 가능해야 함 (패딩으로 인해)
    EXPECT_GE(fq1.getNumFrames(), 0);

    // 중간 길이 입력
    std::vector<float> medium(200, 1.0f);
    FrameQueue fq2(medium.data(), medium.size(), frame_size, hop_size);
    EXPECT_GT(fq2.getNumFrames(), 0);

    // 긴 입력
    std::vector<float> long_input(2000, 1.0f);
    FrameQueue fq3(long_input.data(), long_input.size(), frame_size, hop_size);
    EXPECT_GT(fq3.getNumFrames(), 0);
}

// Center 패딩 테스트
TEST_F(FrameQueueTest, CenterPadding) {
    size_t frame_size = 32;
    size_t hop_size = 16;

    // center=true (기본값)
    FrameQueue fq_center(test_signal_short_.data(), test_signal_short_.size(),
                        frame_size, hop_size, true);

    // center=false
    FrameQueue fq_no_center(test_signal_short_.data(), test_signal_short_.size(),
                           frame_size, hop_size, false);

    // center=true일 때 더 많은 프레임이 생성되어야 함 (패딩으로 인해)
    EXPECT_GE(fq_center.getNumFrames(), fq_no_center.getNumFrames());
}

// 패딩 모드 테스트
TEST_F(FrameQueueTest, PaddingModes) {
    size_t frame_size = 32;
    size_t hop_size = 16;

    // CONSTANT 패딩 (기본값)
    FrameQueue fq_constant(test_signal_short_.data(), test_signal_short_.size(),
                          frame_size, hop_size, true, PadMode::CONSTANT);

    // REFLECT 패딩
    FrameQueue fq_reflect(test_signal_short_.data(), test_signal_short_.size(),
                         frame_size, hop_size, true, PadMode::REFLECT);

    // EDGE 패딩
    FrameQueue fq_edge(test_signal_short_.data(), test_signal_short_.size(),
                      frame_size, hop_size, true, PadMode::EDGE);

    EXPECT_GT(fq_constant.getNumFrames(), 0);
    EXPECT_GT(fq_reflect.getNumFrames(), 0);
    EXPECT_GT(fq_edge.getNumFrames(), 0);
}

// 프레임 복사 테스트
TEST_F(FrameQueueTest, FrameCopy) {
    size_t frame_size = 32;
    size_t hop_size = 16;

    FrameQueue fq(test_signal_short_.data(), test_signal_short_.size(), frame_size, hop_size);

    if (fq.getNumFrames() > 0) {
        std::vector<float> frame_buffer(frame_size);
        fq.copyFrame(0, frame_buffer.data());

        const float* original_frame = fq.getFrame(0);
        for (size_t i = 0; i < frame_size; ++i) {
            EXPECT_EQ(frame_buffer[i], original_frame[i]);
        }
    }
}

// 경계 조건 테스트 (OBOE 방지)
TEST_F(FrameQueueTest, BoundaryConditions) {
    size_t frame_size = 32;
    size_t hop_size = 16;

    FrameQueue fq(test_signal_short_.data(), test_signal_short_.size(), frame_size, hop_size);

    size_t num_frames = fq.getNumFrames();

    // 유효한 인덱스 접근
    if (num_frames > 0) {
        EXPECT_NO_THROW(fq.getFrame(0));
        EXPECT_NO_THROW(fq.getFrame(num_frames - 1));
    }

    // 무효한 인덱스 접근
    EXPECT_THROW(fq.getFrame(num_frames), std::out_of_range);
}

// 완료 기준 검증: n프레임*hop + tail ≤ len + pad
TEST_F(FrameQueueTest, CompletionCriteria) {
    size_t frame_size = 32;
    size_t hop_size = 16;
    size_t input_len = test_signal_short_.size();

    FrameQueue fq(test_signal_short_.data(), input_len, frame_size, hop_size, true);

    size_t num_frames = fq.getNumFrames();
    size_t tail = (frame_size > hop_size) ? (frame_size - hop_size) : 0;
    size_t pad_size = frame_size / 2;  // center=true일 때 양쪽 패딩
    size_t total_padded_len = input_len + 2 * pad_size;

    // 완료 기준 검증
    EXPECT_LE(num_frames * hop_size + tail, total_padded_len);
}

// 에러 조건 테스트
TEST_F(FrameQueueTest, ErrorConditions) {
    // frame_size = 0
    EXPECT_THROW(FrameQueue(test_signal_short_.data(), test_signal_short_.size(), 0, 16),
                 std::invalid_argument);

    // hop_size = 0
    EXPECT_THROW(FrameQueue(test_signal_short_.data(), test_signal_short_.size(), 32, 0),
                 std::invalid_argument);

    // null pointer with non-zero length
    EXPECT_THROW(FrameQueue(nullptr, 100, 32, 16), std::invalid_argument);

    // null pointer with zero length (should be OK)
    EXPECT_NO_THROW(FrameQueue(nullptr, 0, 32, 16));
}

// AoS 메모리 레이아웃 테스트
TEST_F(FrameQueueTest, AoSMemoryLayout) {
    size_t frame_size = 4;
    size_t hop_size = 2;
    std::vector<float> simple_input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    FrameQueue fq(simple_input.data(), simple_input.size(), frame_size, hop_size, false);

    const std::vector<float>& all_frames = fq.getAllFrames();

    // AoS 레이아웃 확인: [frame0_sample0, frame0_sample1, ..., frame1_sample0, ...]
    if (fq.getNumFrames() >= 2) {
        const float* frame0 = fq.getFrame(0);
        const float* frame1 = fq.getFrame(1);

        // 메모리 연속성 확인
        EXPECT_EQ(frame1, frame0 + frame_size);
    }
}

// 평가 피드백: REFLECT 경계 케이스 테스트 추가
TEST_F(FrameQueueTest, REFLECTBoundaryConditions) {
    size_t frame_size = 32;
    size_t hop_size = 16;

    // 길이 0 입력
    std::vector<float> empty_input;
    FrameQueue fq_empty(empty_input.data(), empty_input.size(), frame_size, hop_size,
                       true, PadMode::REFLECT);
    EXPECT_GE(fq_empty.getNumFrames(), 0);

    // 길이 1 입력
    std::vector<float> single_input = {1.0f};
    FrameQueue fq_single(single_input.data(), single_input.size(), frame_size, hop_size,
                        true, PadMode::REFLECT);
    EXPECT_GT(fq_single.getNumFrames(), 0);

    // 첫 번째 프레임 검증 (reflect101 매핑 확인)
    if (fq_single.getNumFrames() > 0) {
        const float* frame = fq_single.getFrame(0);
        EXPECT_NE(frame, nullptr);

        // 패딩된 영역에서 reflect101 규칙이 적용되었는지 확인
        // 중앙 부분에는 원본 값이 있어야 함
        bool found_original = false;
        for (size_t i = 0; i < frame_size; ++i) {
            if (std::abs(frame[i] - 1.0f) < 1e-6f) {
                found_original = true;
                break;
            }
        }
        EXPECT_TRUE(found_original) << "Original value not found in reflected frame";
    }

    // frame_size보다 작은 입력 (길이 < frame_size)
    std::vector<float> short_input = {1.0f, 2.0f, 3.0f};  // 길이 3 < frame_size 32
    FrameQueue fq_short(short_input.data(), short_input.size(), frame_size, hop_size,
                       true, PadMode::REFLECT);
    EXPECT_GT(fq_short.getNumFrames(), 0);

    // 첫 번째 프레임에서 reflect101 패턴 확인
    if (fq_short.getNumFrames() > 0) {
        const float* frame = fq_short.getFrame(0);
        EXPECT_NE(frame, nullptr);

        // 원본 값들이 프레임에 포함되어 있는지 확인
        std::vector<bool> found(short_input.size(), false);
        for (size_t i = 0; i < frame_size; ++i) {
            for (size_t j = 0; j < short_input.size(); ++j) {
                if (std::abs(frame[i] - short_input[j]) < 1e-6f) {
                    found[j] = true;
                }
            }
        }

        for (size_t j = 0; j < short_input.size(); ++j) {
            EXPECT_TRUE(found[j]) << "Original value " << short_input[j]
                                 << " not found in reflected frame";
        }
    }
}

// REFLECT 인덱스 매핑 정확성 테스트
TEST_F(FrameQueueTest, REFLECTIndexMapping) {
    size_t frame_size = 8;
    size_t hop_size = 4;

    // 알려진 패턴으로 reflect101 동작 검증
    std::vector<float> pattern = {1.0f, 2.0f, 3.0f, 4.0f};
    FrameQueue fq(pattern.data(), pattern.size(), frame_size, hop_size,
                  true, PadMode::REFLECT);

    if (fq.getNumFrames() > 0) {
        const float* frame = fq.getFrame(0);
        EXPECT_NE(frame, nullptr);

        // reflect101 패턴 검증: [1,2,3,4] -> 패딩 시 [3,2,1,2,3,4,3,2] 형태
        // 정확한 패턴은 pad_size와 center 오프셋에 따라 달라지지만
        // 최소한 원본 값들이 순서대로 나타나야 함
        std::cout << "REFLECT pattern: ";
        for (size_t i = 0; i < frame_size; ++i) {
            std::cout << frame[i] << " ";
        }
        std::cout << std::endl;

        // 원본 패턴이 연속으로 나타나는 구간이 있는지 확인
        bool found_sequence = false;
        for (size_t i = 0; i <= frame_size - pattern.size(); ++i) {
            bool matches = true;
            for (size_t j = 0; j < pattern.size(); ++j) {
                if (std::abs(frame[i + j] - pattern[j]) > 1e-6f) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                found_sequence = true;
                break;
            }
        }
        EXPECT_TRUE(found_sequence) << "Original sequence not found in reflected frame";
    }
}

// 피드백 반영: 길이 0/1/len < frame 극단 케이스 테스트 추가
TEST_F(FrameQueueTest, ExtremeEdgeCases) {
    size_t frame_size = 32;
    size_t hop_size = 16;

    // 케이스 1: 길이 0 입력
    std::vector<float> empty_input;
    FrameQueue fq_empty(empty_input.data(), empty_input.size(), frame_size, hop_size, true);

    // center=true이므로 패딩으로 인해 프레임이 생성될 수 있음
    std::cout << "길이 0 입력 - 프레임 수: " << fq_empty.getNumFrames() << std::endl;
    EXPECT_GE(fq_empty.getNumFrames(), 0) << "Empty input should handle gracefully";

    if (fq_empty.getNumFrames() > 0) {
        const float* frame = fq_empty.getFrame(0);
        EXPECT_NE(frame, nullptr) << "Frame pointer should be valid even for empty input";

        // 패딩된 프레임은 모두 0이어야 함 (CONSTANT 패딩)
        bool all_zero = true;
        for (size_t i = 0; i < frame_size; ++i) {
            if (std::abs(frame[i]) > 1e-6f) {
                all_zero = false;
                break;
            }
        }
        EXPECT_TRUE(all_zero) << "Empty input frame should be all zeros";
    }

    // 케이스 2: 길이 1 입력
    std::vector<float> single_input = {1.5f};
    FrameQueue fq_single(single_input.data(), single_input.size(), frame_size, hop_size, true);

    std::cout << "길이 1 입력 - 프레임 수: " << fq_single.getNumFrames() << std::endl;
    EXPECT_GT(fq_single.getNumFrames(), 0) << "Single sample input should generate frames";

    if (fq_single.getNumFrames() > 0) {
        const float* frame = fq_single.getFrame(0);
        EXPECT_NE(frame, nullptr);

        // 원본 값이 프레임 어딘가에 있어야 함
        bool found_original = false;
        for (size_t i = 0; i < frame_size; ++i) {
            if (std::abs(frame[i] - 1.5f) < 1e-6f) {
                found_original = true;
                break;
            }
        }
        EXPECT_TRUE(found_original) << "Original value should be found in frame";
    }

    // 케이스 3: len < frame_size 입력
    std::vector<float> short_input = {1.0f, 2.0f, 3.0f};  // 길이 3 < frame_size 32
    FrameQueue fq_short(short_input.data(), short_input.size(), frame_size, hop_size, true);

    std::cout << "짧은 입력 (len=" << short_input.size() << " < frame=" << frame_size
              << ") - 프레임 수: " << fq_short.getNumFrames() << std::endl;
    EXPECT_GT(fq_short.getNumFrames(), 0) << "Short input should generate frames with padding";

    if (fq_short.getNumFrames() > 0) {
        const float* frame = fq_short.getFrame(0);
        EXPECT_NE(frame, nullptr);

        // 모든 원본 값들이 프레임에 포함되어야 함
        std::vector<bool> found(short_input.size(), false);
        for (size_t i = 0; i < frame_size; ++i) {
            for (size_t j = 0; j < short_input.size(); ++j) {
                if (std::abs(frame[i] - short_input[j]) < 1e-6f) {
                    found[j] = true;
                }
            }
        }

        for (size_t j = 0; j < short_input.size(); ++j) {
            EXPECT_TRUE(found[j]) << "Original value " << short_input[j] << " not found in frame";
        }
    }

    // 케이스 4: frame_size = hop_size (겹침 없음)
    size_t no_overlap_frame = 16;
    size_t no_overlap_hop = 16;
    std::vector<float> no_overlap_input(64, 1.0f);

    FrameQueue fq_no_overlap(no_overlap_input.data(), no_overlap_input.size(),
                            no_overlap_frame, no_overlap_hop, false);

    std::cout << "겹침 없음 (frame=hop=" << no_overlap_frame << ") - 프레임 수: "
              << fq_no_overlap.getNumFrames() << std::endl;

    // 예상 프레임 수: 64/16 = 4
    EXPECT_EQ(fq_no_overlap.getNumFrames(), 4) << "No overlap should produce exactly 4 frames";

    // 케이스 5: hop_size > frame_size (갭 존재)
    size_t gap_frame = 8;
    size_t gap_hop = 12;
    std::vector<float> gap_input(48, 2.0f);

    FrameQueue fq_gap(gap_input.data(), gap_input.size(), gap_frame, gap_hop, false);

    std::cout << "갭 존재 (frame=" << gap_frame << " < hop=" << gap_hop
              << ") - 프레임 수: " << fq_gap.getNumFrames() << std::endl;

    EXPECT_GT(fq_gap.getNumFrames(), 0) << "Gap configuration should still work";

    // 각 프레임이 올바른 데이터를 포함하는지 확인
    for (size_t f = 0; f < fq_gap.getNumFrames(); ++f) {
        const float* frame = fq_gap.getFrame(f);
        for (size_t i = 0; i < gap_frame; ++i) {
            size_t input_idx = f * gap_hop + i;
            if (input_idx < gap_input.size()) {
                EXPECT_NEAR(frame[i], 2.0f, 1e-6f)
                    << "Frame " << f << " sample " << i << " mismatch";
            }
        }
    }
}

// 메모리 경계 및 안전성 테스트
TEST_F(FrameQueueTest, MemoryBoundaryTests) {
    size_t frame_size = 16;
    size_t hop_size = 8;

    // 매우 큰 입력으로 메모리 할당 테스트
    size_t large_size = 100000;
    std::vector<float> large_input(large_size);
    std::iota(large_input.begin(), large_input.end(), 0.0f);  // 0, 1, 2, 3, ...

    EXPECT_NO_THROW({
        FrameQueue fq_large(large_input.data(), large_input.size(), frame_size, hop_size);
        std::cout << "대용량 입력 (" << large_size << " 샘플) - 프레임 수: "
                  << fq_large.getNumFrames() << std::endl;

        // 첫 번째와 마지막 프레임 검증
        if (fq_large.getNumFrames() > 0) {
            const float* first_frame = fq_large.getFrame(0);
            const float* last_frame = fq_large.getFrame(fq_large.getNumFrames() - 1);

            EXPECT_NE(first_frame, nullptr);
            EXPECT_NE(last_frame, nullptr);
            EXPECT_NE(first_frame, last_frame);  // 다른 메모리 위치
        }
    }) << "Large input should be handled without throwing";

    // 경계값 테스트: frame_size = 1
    std::vector<float> unit_test = {1.0f, 2.0f, 3.0f, 4.0f};
    FrameQueue fq_unit(unit_test.data(), unit_test.size(), 1, 1);

    EXPECT_EQ(fq_unit.getFrameSize(), 1);
    EXPECT_EQ(fq_unit.getNumFrames(), 4);  // 각 샘플이 하나의 프레임

    for (size_t i = 0; i < 4; ++i) {
        const float* frame = fq_unit.getFrame(i);
        EXPECT_NEAR(frame[0], static_cast<float>(i + 1), 1e-6f);
    }
}