#include "framer.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <cmath>

namespace dsp {

Framer::Framer()
    : frame_size_(0), hop_size_(0), channels_(1),
      boundary_mode_(BoundaryMode::ZERO_PAD),
      write_pos_(0), read_pos_(0), params_set_(false) {
}

void Framer::set_params(size_t frame_size, size_t hop_size, size_t channels,
                       BoundaryMode boundary_mode) {
    if (frame_size == 0) {
        throw std::invalid_argument("Frame size must be greater than 0");
    }
    if (hop_size == 0) {
        throw std::invalid_argument("Hop size must be greater than 0");
    }
    if (channels == 0) {
        throw std::invalid_argument("Channels must be greater than 0");
    }

    frame_size_ = frame_size;
    hop_size_ = hop_size;
    channels_ = channels;
    boundary_mode_ = boundary_mode;
    params_set_ = true;

    // 버퍼 초기화
    reset();
}

bool Framer::push(const float* interleaved, size_t frames) {
    if (!params_set_) {
        return false;
    }
    if (interleaved == nullptr && frames > 0) {
        return false;
    }

    size_t samples_to_add = frames * channels_;
    if (samples_to_add == 0) {
        return true;
    }

    // 버퍼 크기 조정
    resize_buffer_if_needed(write_pos_ + samples_to_add);

    // 데이터 복사
    std::copy(interleaved, interleaved + samples_to_add,
              buffer_.begin() + write_pos_);
    write_pos_ += samples_to_add;

    return true;
}

bool Framer::pop(float* out_frame) {
    if (!params_set_ || out_frame == nullptr) {
        return false;
    }

    return extract_frame(out_frame);
}

size_t Framer::available_frames() const {
    if (!params_set_) {
        return 0;
    }
    return calculate_available_frames();
}

void Framer::reset() {
    buffer_.clear();
    write_pos_ = 0;
    read_pos_ = 0;

    if (params_set_) {
        // 초기 버퍼 크기 설정 (최소 2개 프레임 분량)
        size_t initial_size = frame_size_ * channels_ * 2;
        buffer_.resize(initial_size, 0.0f);
    }
}

size_t Framer::calculate_available_frames() const {
    if (write_pos_ <= read_pos_) {
        return 0;
    }

    size_t available_samples = write_pos_ - read_pos_;
    size_t available_frames_in_samples = available_samples / channels_;

    // 프레이밍 호환성 공식: len = floor((N - frame)/hop) + 1
    // 하지만 첫 번째 프레임은 frame_size만큼만 있으면 됨
    if (available_frames_in_samples < frame_size_) {
        // ZERO_PAD 모드에서는 부족한 프레임도 허용
        if (boundary_mode_ == BoundaryMode::ZERO_PAD && available_frames_in_samples > 0) {
            return 1;  // 패딩으로 하나의 프레임 생성 가능
        }
        return 0;
    }

    size_t num_frames = (available_frames_in_samples - frame_size_) / hop_size_ + 1;

    // 경계 모드에 따른 조정
    if (boundary_mode_ == BoundaryMode::DROP) {
        // 완전한 프레임만 반환
        size_t last_frame_start = (num_frames - 1) * hop_size_;
        if (last_frame_start + frame_size_ > available_frames_in_samples) {
            num_frames = (num_frames > 0) ? num_frames - 1 : 0;
        }
    }

    return num_frames;
}

void Framer::resize_buffer_if_needed(size_t required_size) {
    if (buffer_.size() < required_size) {
        // 버퍼 크기를 2배씩 증가
        size_t new_size = std::max(required_size, buffer_.size() * 2);
        buffer_.resize(new_size, 0.0f);
    }
}

bool Framer::extract_frame(float* out_frame) {
    if (available_frames() == 0) {
        return false;
    }

    size_t frame_start_sample = read_pos_;
    size_t frame_samples = frame_size_ * channels_;

    // 프레임 데이터 복사
    if (frame_start_sample + frame_samples <= write_pos_) {
        // 완전한 프레임 데이터가 있는 경우
        std::copy(buffer_.begin() + frame_start_sample,
                  buffer_.begin() + frame_start_sample + frame_samples,
                  out_frame);
    } else {
        // 부분적인 데이터만 있는 경우
        if (boundary_mode_ == BoundaryMode::DROP) {
            return false;
        }

        // ZERO_PAD 모드: 부족한 부분을 0으로 채움
        size_t available_samples = write_pos_ - frame_start_sample;

        // 사용 가능한 데이터 복사
        if (available_samples > 0) {
            std::copy(buffer_.begin() + frame_start_sample,
                      buffer_.begin() + frame_start_sample + available_samples,
                      out_frame);
        }

        // 나머지를 0으로 채움
        std::fill(out_frame + available_samples,
                  out_frame + frame_samples, 0.0f);
    }

    // 읽기 위치 업데이트
    read_pos_ += hop_size_ * channels_;

    // 버퍼 정리 (읽은 데이터가 버퍼의 절반 이상일 때)
    if (read_pos_ > buffer_.size() / 2) {
        size_t remaining_samples = write_pos_ - read_pos_;
        if (remaining_samples > 0) {
            std::copy(buffer_.begin() + read_pos_,
                      buffer_.begin() + write_pos_,
                      buffer_.begin());
        }
        write_pos_ = remaining_samples;
        read_pos_ = 0;
    }

    return true;
}

} // namespace dsp