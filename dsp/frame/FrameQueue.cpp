#include "FrameQueue.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dsp {

FrameQueue::FrameQueue(const float* in, size_t len, size_t frame_size, size_t hop_size,
                       bool center, PadMode pad_mode)
    : frame_size_(frame_size), hop_size_(hop_size), center_(center), pad_mode_(pad_mode) {

    if (frame_size == 0) {
        throw std::invalid_argument("Frame size must be greater than 0");
    }
    if (hop_size == 0) {
        throw std::invalid_argument("Hop size must be greater than 0");
    }
    if (in == nullptr && len > 0) {
        throw std::invalid_argument("Input pointer cannot be null when length > 0");
    }

    // 패딩된 입력 데이터 생성
    std::vector<float> padded_input = createPaddedInput(in, len);

    // 프레임 개수 계산
    num_frames_ = calculateNumFrames(padded_input.size());

    // 프레임 데이터 생성 (AoS 레이아웃)
    frames_.resize(num_frames_ * frame_size_);

    for (size_t frame_idx = 0; frame_idx < num_frames_; ++frame_idx) {
        size_t start_pos = frame_idx * hop_size_;
        size_t frame_offset = frame_idx * frame_size_;

        for (size_t sample_idx = 0; sample_idx < frame_size_; ++sample_idx) {
            size_t input_idx = start_pos + sample_idx;

            if (input_idx < padded_input.size()) {
                frames_[frame_offset + sample_idx] = padded_input[input_idx];
            } else {
                // 범위를 벗어난 경우 패딩 값 사용
                frames_[frame_offset + sample_idx] = getPaddingValue(padded_input, input_idx);
            }
        }
    }
}

const float* FrameQueue::getFrame(size_t frame_idx) const {
    if (frame_idx >= num_frames_) {
        throw std::out_of_range("Frame index out of range");
    }
    return &frames_[frame_idx * frame_size_];
}

void FrameQueue::copyFrame(size_t frame_idx, float* output) const {
    if (frame_idx >= num_frames_) {
        throw std::out_of_range("Frame index out of range");
    }
    if (output == nullptr) {
        throw std::invalid_argument("Output buffer cannot be null");
    }

    const float* frame_data = &frames_[frame_idx * frame_size_];
    std::copy(frame_data, frame_data + frame_size_, output);
}

std::vector<float> FrameQueue::createPaddedInput(const float* in, size_t len) const {
    if (!center_) {
        // center=false인 경우 패딩 없이 원본 데이터 반환
        return std::vector<float>(in, in + len);
    }

    // center=true인 경우 앞뒤 패딩 추가
    size_t pad_size = frame_size_ / 2;
    size_t padded_len = len + 2 * pad_size;
    std::vector<float> padded_input(padded_len);

    // 앞쪽 패딩
    for (size_t i = 0; i < pad_size; ++i) {
        switch (pad_mode_) {
            case PadMode::CONSTANT:
                padded_input[i] = 0.0f;
                break;
            case PadMode::REFLECT:
                if (len > 0) {
                    size_t reflect_idx = std::min(pad_size - 1 - i, len - 1);
                    padded_input[i] = in[reflect_idx];
                } else {
                    padded_input[i] = 0.0f;
                }
                break;
            case PadMode::EDGE:
                padded_input[i] = (len > 0) ? in[0] : 0.0f;
                break;
        }
    }

    // 원본 데이터 복사
    std::copy(in, in + len, padded_input.begin() + pad_size);

    // 뒤쪽 패딩
    for (size_t i = 0; i < pad_size; ++i) {
        size_t idx = pad_size + len + i;
        switch (pad_mode_) {
            case PadMode::CONSTANT:
                padded_input[idx] = 0.0f;
                break;
            case PadMode::REFLECT:
                if (len > 0) {
                    size_t reflect_idx = len - 1 - std::min(i, len - 1);
                    padded_input[idx] = in[reflect_idx];
                } else {
                    padded_input[idx] = 0.0f;
                }
                break;
            case PadMode::EDGE:
                padded_input[idx] = (len > 0) ? in[len - 1] : 0.0f;
                break;
        }
    }

    return padded_input;
}

size_t FrameQueue::calculateNumFrames(size_t padded_len) const {
    if (padded_len < frame_size_) {
        return 0;
    }

    // n_frames * hop_size + tail <= padded_len 조건을 만족하는 최대 n_frames 계산
    // tail = frame_size - hop_size (마지막 프레임의 나머지 부분)
    size_t tail = (frame_size_ > hop_size_) ? (frame_size_ - hop_size_) : 0;

    if (padded_len < tail) {
        return 0;
    }

    // n_frames * hop_size <= padded_len - tail
    size_t available_len = padded_len - tail;
    return available_len / hop_size_;  // 내림 계산 (올림이 아닌)
}

float FrameQueue::getPaddingValue(const std::vector<float>& padded_input, int idx) const {
    switch (pad_mode_) {
        case PadMode::CONSTANT:
            return 0.0f;
        case PadMode::REFLECT:
            if (padded_input.empty()) return 0.0f;
            if (idx < 0) {
                return padded_input[std::min(static_cast<size_t>(-idx - 1), padded_input.size() - 1)];
            } else {
                size_t reflect_idx = padded_input.size() - 1 - (idx - padded_input.size());
                return padded_input[std::max(0, static_cast<int>(reflect_idx))];
            }
        case PadMode::EDGE:
            if (padded_input.empty()) return 0.0f;
            return (idx < 0) ? padded_input[0] : padded_input[padded_input.size() - 1];
        default:
            return 0.0f;
    }
}

} // namespace dsp