#include "OLAAccumulator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstring>

namespace dsp {

OLAAccumulator::OLAAccumulator(const OLAConfig& cfg)
    : cfg_(cfg), gain_(cfg.gain), ring_len_(0),
      write_head_(0), read_head_(0),
      produced_samples_(0), consumed_samples_(0),
      meter_peak_(0.0f), flushing_(false) {

    if (!cfg_.isValid()) {
        throw std::invalid_argument("Invalid OLA configuration");
    }

    // 링 버퍼 크기 계산 및 할당
    ring_len_ = calculate_ring_size();
    ring_accum_.resize(ring_len_ * cfg_.channels, 0.0f);
    norm_buffer_.resize(ring_len_, 0.0f);
}

void OLAAccumulator::set_window(const float* w, int wlen) {
    if (w == nullptr) {
        throw std::invalid_argument("Window pointer cannot be null");
    }
    if (wlen != cfg_.frame_size) {
        throw std::invalid_argument("Window size must match frame size");
    }

    // 윈도우 함수 복사
    window_.resize(wlen);
    std::copy(w, w + wlen, window_.begin());

    // COLA 정규화 계수 초기화
    initialize_normalization();
}

void OLAAccumulator::set_gain(float g) {
    if (g <= 0.0f) {
        throw std::invalid_argument("Gain must be positive");
    }
    gain_ = g;
}

void OLAAccumulator::push_frame(int64_t frame_index, const float* frame) {
    if (frame == nullptr) {
        throw std::invalid_argument("Frame pointer cannot be null");
    }

    // 프레임 시작 위치 계산 (center 오프셋 적용)
    int64_t center_offset = calculate_center_offset();
    int64_t start_sample = frame_index * cfg_.hop_size - center_offset;

    // 채널별 처리
    if (cfg_.channels == 1) {
        add_frame_mono(start_sample, frame);
    } else {
        add_frame_multi(start_sample, frame);
    }

    // 생산 샘플 수 업데이트
    produced_samples_ = std::max(produced_samples_, start_sample + cfg_.frame_size);
}

int OLAAccumulator::pull(float* dst, int num_samples) {
    if (dst == nullptr) {
        throw std::invalid_argument("Destination buffer cannot be null");
    }
    if (num_samples <= 0) {
        return 0;
    }

    // 사용 가능한 샘플 수 계산
    int64_t available = produced_samples_ - consumed_samples_;
    if (available <= 0) {
        return 0;
    }

    if (available < num_samples) {
        num_samples = static_cast<int>(available);
    }

    // 정규화 및 출력
    normalize_and_clear(dst, num_samples, consumed_samples_);

    // 소비 샘플 수 업데이트
    consumed_samples_ += num_samples;

    // 피크 미터 업데이트
    update_peak_meter(dst, num_samples * cfg_.channels);

    return num_samples;
}

void OLAAccumulator::flush() {
    flushing_ = true;

    // flush 시점에서 링 버퍼에 남은 데이터를 모두 출력할 수 있도록
    // produced_samples_를 소비된 샘플 + 링 버퍼 크기로 설정
    // 이렇게 하면 pull에서 flush 모드일 때 모든 데이터를 출력할 수 있음
    int64_t remaining_in_ring = produced_samples_ - consumed_samples_;
    if (remaining_in_ring < static_cast<int64_t>(ring_len_)) {
        produced_samples_ = consumed_samples_ + ring_len_;
    }
}

void OLAAccumulator::reset() {
    // 버퍼 초기화
    std::fill(ring_accum_.begin(), ring_accum_.end(), 0.0f);

    // 인덱스 리셋
    write_head_ = 0;
    read_head_ = 0;
    produced_samples_ = 0;
    consumed_samples_ = 0;

    // 메트릭 리셋
    meter_peak_ = 0.0f;
    flushing_ = false;

    // 정규화 버퍼는 윈도우가 설정되어 있으면 재초기화
    if (!window_.empty()) {
        initialize_normalization();
    }
}

size_t OLAAccumulator::calculate_ring_size() const {
    // 최소 중첩 수: ceil(N/H)
    size_t min_overlaps = (cfg_.frame_size + cfg_.hop_size - 1) / cfg_.hop_size;

    // 안전 마진 추가: +20 for large inputs
    size_t K = min_overlaps + 20;

    // 링 크기: K * H (홉 단위 정렬)
    return K * cfg_.hop_size;
}

void OLAAccumulator::initialize_normalization() {
    if (window_.empty()) {
        // 윈도우가 없으면 정규화 계수를 1.0으로 설정
        std::fill(norm_buffer_.begin(), norm_buffer_.end(), 1.0f);
        return;
    }

    // 윈도우가 내부에서 적용되는 경우에만 COLA 정규화 계산
    if (!cfg_.apply_window_inside) {
        // 윈도우가 외부에서 적용되면 정규화 계수를 1.0으로 설정
        std::fill(norm_buffer_.begin(), norm_buffer_.end(), 1.0f);
        return;
    }

    // 정규화 버퍼 초기화
    std::fill(norm_buffer_.begin(), norm_buffer_.end(), 0.0f);

    // 링 버퍼의 주기성을 고려한 COLA 정규화
    // 충분히 많은 프레임을 시뮬레이션하여 정상 상태 달성

    // 임시 버퍼: 링 버퍼보다 충분히 큰 크기
    size_t temp_len = ring_len_ * 3;  // 3배 크기로 충분한 여유
    std::vector<float> temp_norm(temp_len, 0.0f);

    // 충분한 수의 프레임으로 정상 상태 시뮬레이션
    int num_frames = (temp_len / cfg_.hop_size) + 10;

    for (int frame = 0; frame < num_frames; ++frame) {
        int64_t frame_start = frame * cfg_.hop_size;

        for (int t = 0; t < cfg_.frame_size; ++t) {
            int64_t pos = frame_start + t;
            if (pos >= 0 && pos < static_cast<int64_t>(temp_len)) {
                float w = window_[t];
                temp_norm[pos] += w;
            }
        }
    }

    // 정상 상태 구간에서 한 주기만 추출하여 링 버퍼에 복사
    // 중앙 부분에서 안정된 값들을 사용
    size_t stable_start = ring_len_ * 2;  // 충분히 안정된 구간

    for (size_t i = 0; i < ring_len_; ++i) {
        norm_buffer_[i] = std::max(temp_norm[stable_start + i], 1e-8f);
    }
}

void OLAAccumulator::add_frame_mono(int64_t start_sample, const float* frame) {
    // Phase 3: SIMD 최적화 준비 - 벡터화 친화적 루프 구조
    // TODO: 향후 #pragma omp simd 또는 AVX2 intrinsics 적용 가능

    // 음수 시작 위치 처리를 위한 오프셋 계산
    int start_offset = 0;
    if (start_sample < 0) {
        start_offset = static_cast<int>(-start_sample);
        if (start_offset >= cfg_.frame_size) {
            return; // 전체 프레임이 음수 범위
        }
    }

    // 벡터화 가능한 메인 루프
    const int effective_size = cfg_.frame_size - start_offset;
    const int64_t effective_start = start_sample + start_offset;

    // 윈도우 적용 여부에 따른 분기 (브랜치 예측 최적화)
    if (cfg_.apply_window_inside && !window_.empty()) {
        // 윈도우 적용 버전 (향후 SIMD 최적화 대상)
        for (int t = 0; t < effective_size; ++t) {
            int64_t sample_pos = effective_start + t;
            size_t idx = ring_index(sample_pos);
            int frame_idx = start_offset + t;

            float x = frame[frame_idx] * gain_ * window_[frame_idx];
            ring_accum_[idx] += x;
        }
    } else {
        // 윈도우 미적용 버전 (향후 SIMD 최적화 대상)
        for (int t = 0; t < effective_size; ++t) {
            int64_t sample_pos = effective_start + t;
            size_t idx = ring_index(sample_pos);
            int frame_idx = start_offset + t;

            float x = frame[frame_idx] * gain_;
            ring_accum_[idx] += x;
        }
    }
}

void OLAAccumulator::add_frame_multi(int64_t start_sample, const float* frame) {
    // 다채널 처리: SoA 레이아웃 (채널별 독립 처리)
    for (int c = 0; c < cfg_.channels; ++c) {
        size_t channel_offset = c * ring_len_;

        for (int t = 0; t < cfg_.frame_size; ++t) {
            int64_t sample_pos = start_sample + t;

            // 음수 인덱스 처리 - 음수면 스킵
            if (sample_pos < 0) {
                continue;
            }

            size_t idx = ring_index(sample_pos);
            float x = frame[t * cfg_.channels + c] * gain_;

            // 내부 윈도우 적용 (옵션)
            if (cfg_.apply_window_inside && !window_.empty()) {
                x *= window_[t];
            }

            ring_accum_[channel_offset + idx] += x;
        }
    }
}

void OLAAccumulator::update_peak_meter(const float* data, int num_samples) {
    for (int i = 0; i < num_samples; ++i) {
        float abs_val = std::abs(data[i]);
        meter_peak_ = std::max(meter_peak_, abs_val);
    }
}

void OLAAccumulator::normalize_and_clear(float* dst, int num_samples, int64_t start_idx) {
    const float eps = 1e-8f;

    if (cfg_.channels == 1) {
        // Phase 3: 단일 채널 SIMD 최적화 준비
        // TODO: 향후 벡터화 - 정규화와 소거를 분리하여 메모리 접근 최적화

        // 범위 체크를 먼저 수행하여 유효한 구간만 처리
        int valid_start = 0;
        int valid_end = num_samples;

        // 앞쪽 무효 구간 처리
        while (valid_start < num_samples && (start_idx + valid_start < 0 || start_idx + valid_start >= produced_samples_)) {
            dst[valid_start] = 0.0f;
            valid_start++;
        }

        // 뒤쪽 무효 구간 찾기
        while (valid_end > valid_start && (start_idx + valid_end - 1 < 0 || start_idx + valid_end - 1 >= produced_samples_)) {
            dst[valid_end - 1] = 0.0f;
            valid_end--;
        }

        // 유효 구간 벡터화 가능한 처리 (향후 SIMD 최적화 대상)
        for (int i = valid_start; i < valid_end; ++i) {
            int64_t sample_pos = start_idx + i;
            size_t idx = ring_index(sample_pos);

            // COLA 정규화 적용 (벡터화 가능)
            float norm = std::max(eps, norm_buffer_[idx]);
            dst[i] = ring_accum_[idx] / norm;

            // 버퍼 소거 (별도 루프로 분리 가능)
            ring_accum_[idx] = 0.0f;
        }
    } else {
        // Phase 3: 다채널 SoA → interleaved 변환 최적화 준비
        // TODO: 향후 채널별 벡터화 및 인터리빙 최적화

        for (int i = 0; i < num_samples; ++i) {
            int64_t sample_pos = start_idx + i;

            if (sample_pos < 0 || sample_pos >= produced_samples_) {
                // 무효 샘플 처리 (벡터화 가능)
                for (int c = 0; c < cfg_.channels; ++c) {
                    dst[i * cfg_.channels + c] = 0.0f;
                }
                continue;
            }

            size_t idx = ring_index(sample_pos);
            float norm = std::max(eps, norm_buffer_[idx]);

            // 채널별 처리 (향후 SIMD 최적화 대상)
            for (int c = 0; c < cfg_.channels; ++c) {
                size_t channel_offset = c * ring_len_;
                size_t src_idx = channel_offset + idx;
                size_t dst_idx = i * cfg_.channels + c;

                // COLA 정규화 적용
                dst[dst_idx] = ring_accum_[src_idx] / norm;

                // 버퍼 소거
                ring_accum_[src_idx] = 0.0f;
            }
        }
    }
}

} // namespace dsp