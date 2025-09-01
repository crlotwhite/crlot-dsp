#include "OLAAccumulator.h"
#include "norm_builder.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstring>

// SIMD 최적화 지원
#if defined(_OPENMP) && SIMD_AVAILABLE
#include <omp.h>
#endif

// 컴파일러별 SIMD 지원 확인
#if defined(__AVX2__)
#include <immintrin.h>
#define SIMD_AVAILABLE 1
#define SIMD_WIDTH 8  // AVX2: 8 floats
#elif defined(__SSE2__)
#include <emmintrin.h>
#define SIMD_AVAILABLE 1
#define SIMD_WIDTH 4  // SSE2: 4 floats
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_AVAILABLE 1
#define SIMD_WIDTH 4  // NEON: 4 floats
#else
#define SIMD_AVAILABLE 0
#define SIMD_WIDTH 1
#endif

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

    // 피크 미터 업데이트 (인터리브된 샘플 수 전달)
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

    // 최적화된 선형 누적 COLA 정규화 계산
    dsp::ola::build_norm_linear(norm_buffer_.data(), window_.data(),
                               ring_len_, static_cast<size_t>(cfg_.frame_size),
                               static_cast<size_t>(cfg_.hop_size));
}

void OLAAccumulator::add_frame_mono(int64_t start_sample, const float* frame) {
    // 피드백 반영: SIMD 최적화 1차 투입 (OMP+SIMD)

    // 음수 시작 위치 처리를 위한 오프셋 계산
    int start_offset = 0;
    if (start_sample < 0) {
        start_offset = static_cast<int>(-start_sample);
        if (start_offset >= cfg_.frame_size) {
            return; // 전체 프레임이 음수 범위
        }
    }

    const int effective_size = cfg_.frame_size - start_offset;
    const int64_t effective_start = start_sample + start_offset;

    // SIMD 최적화 적용 조건 확인
    const bool use_simd = (effective_size >= SIMD_WIDTH * 2) && SIMD_AVAILABLE;

    if (cfg_.apply_window_inside && !window_.empty()) {
        // 윈도우 적용 버전 - SIMD 최적화
        if (use_simd) {
            add_frame_mono_windowed_simd(effective_start, frame, start_offset, effective_size);
        } else {
            add_frame_mono_windowed_scalar(effective_start, frame, start_offset, effective_size);
        }
    } else {
        // 윈도우 미적용 버전 - SIMD 최적화
        if (use_simd) {
            add_frame_mono_plain_simd(effective_start, frame, start_offset, effective_size);
        } else {
            add_frame_mono_plain_scalar(effective_start, frame, start_offset, effective_size);
        }
    }
}

// SIMD 최적화된 윈도우 적용 버전
void OLAAccumulator::add_frame_mono_windowed_simd(int64_t effective_start, const float* frame,
                                                  int start_offset, int effective_size) {
#if SIMD_AVAILABLE && defined(_OPENMP)
    const int simd_end = (effective_size / SIMD_WIDTH) * SIMD_WIDTH;

    // SIMD 처리 가능한 구간
    #pragma omp simd safelen(SIMD_WIDTH)
    for (int t = 0; t < simd_end; t += SIMD_WIDTH) {
        // 벡터화된 처리를 위한 루프 언롤링
        for (int i = 0; i < SIMD_WIDTH && (t + i) < effective_size; ++i) {
            int64_t sample_pos = effective_start + t + i;
            size_t idx = ring_index(sample_pos);
            int frame_idx = start_offset + t + i;

            float x = frame[frame_idx] * gain_ * window_[frame_idx];
            ring_accum_[idx] += x;
        }
    }

    // 나머지 처리 (스칼라)
    for (int t = simd_end; t < effective_size; ++t) {
        int64_t sample_pos = effective_start + t;
        size_t idx = ring_index(sample_pos);
        int frame_idx = start_offset + t;

        float x = frame[frame_idx] * gain_ * window_[frame_idx];
        ring_accum_[idx] += x;
    }
#else
    // SIMD 미지원 시 스칼라 버전으로 폴백
    add_frame_mono_windowed_scalar(effective_start, frame, start_offset, effective_size);
#endif
}

// 스칼라 윈도우 적용 버전
void OLAAccumulator::add_frame_mono_windowed_scalar(int64_t effective_start, const float* frame,
                                                   int start_offset, int effective_size) {
    for (int t = 0; t < effective_size; ++t) {
        int64_t sample_pos = effective_start + t;
        size_t idx = ring_index(sample_pos);
        int frame_idx = start_offset + t;

        float x = frame[frame_idx] * gain_ * window_[frame_idx];
        ring_accum_[idx] += x;
    }
}

// SIMD 최적화된 윈도우 미적용 버전
void OLAAccumulator::add_frame_mono_plain_simd(int64_t effective_start, const float* frame,
                                               int start_offset, int effective_size) {
#if SIMD_AVAILABLE && defined(_OPENMP)
    const int simd_end = (effective_size / SIMD_WIDTH) * SIMD_WIDTH;

    // SIMD 처리 가능한 구간
    #pragma omp simd aligned(frame:32) safelen(SIMD_WIDTH)
    for (int t = 0; t < simd_end; t += SIMD_WIDTH) {
        // 벡터화된 처리를 위한 루프 언롤링
        for (int i = 0; i < SIMD_WIDTH && (t + i) < effective_size; ++i) {
            int64_t sample_pos = effective_start + t + i;
            size_t idx = ring_index(sample_pos);
            int frame_idx = start_offset + t + i;

            float x = frame[frame_idx] * gain_;
            ring_accum_[idx] += x;
        }
    }

    // 나머지 처리 (스칼라)
    for (int t = simd_end; t < effective_size; ++t) {
        int64_t sample_pos = effective_start + t;
        size_t idx = ring_index(sample_pos);
        int frame_idx = start_offset + t;

        float x = frame[frame_idx] * gain_;
        ring_accum_[idx] += x;
    }
#else
    // SIMD 미지원 시 스칼라 버전으로 폴백
    add_frame_mono_plain_scalar(effective_start, frame, start_offset, effective_size);
#endif
}

// 스칼라 윈도우 미적용 버전
void OLAAccumulator::add_frame_mono_plain_scalar(int64_t effective_start, const float* frame,
                                                 int start_offset, int effective_size) {
    for (int t = 0; t < effective_size; ++t) {
        int64_t sample_pos = effective_start + t;
        size_t idx = ring_index(sample_pos);
        int frame_idx = start_offset + t;

        float x = frame[frame_idx] * gain_;
        ring_accum_[idx] += x;
    }
}

void OLAAccumulator::add_frame_multi(int64_t start_sample, const float* frame) {
    // 피드백 반영: 다채널 SIMD 최적화

    // 음수 시작 위치 처리
    int start_offset = 0;
    if (start_sample < 0) {
        start_offset = static_cast<int>(-start_sample);
        if (start_offset >= cfg_.frame_size) {
            return; // 전체 프레임이 음수 범위
        }
    }

    const int effective_size = cfg_.frame_size - start_offset;
    const int64_t effective_start = start_sample + start_offset;

    // SIMD 최적화 적용 조건 확인
    const bool use_simd = (effective_size >= SIMD_WIDTH * 2) && SIMD_AVAILABLE;

    // 다채널 처리: SoA 레이아웃 (채널별 독립 처리)
    for (int c = 0; c < cfg_.channels; ++c) {
        size_t channel_offset = c * ring_len_;

        if (cfg_.apply_window_inside && !window_.empty()) {
            // 윈도우 적용 버전
            if (use_simd) {
                add_frame_multi_windowed_simd(effective_start, frame, start_offset,
                                            effective_size, c, channel_offset);
            } else {
                add_frame_multi_windowed_scalar(effective_start, frame, start_offset,
                                               effective_size, c, channel_offset);
            }
        } else {
            // 윈도우 미적용 버전
            if (use_simd) {
                add_frame_multi_plain_simd(effective_start, frame, start_offset,
                                         effective_size, c, channel_offset);
            } else {
                add_frame_multi_plain_scalar(effective_start, frame, start_offset,
                                           effective_size, c, channel_offset);
            }
        }
    }
}

// SIMD 최적화된 다채널 윈도우 적용 버전
void OLAAccumulator::add_frame_multi_windowed_simd(int64_t effective_start, const float* frame,
                                                   int start_offset, int effective_size,
                                                   int channel, size_t channel_offset) {
#if SIMD_AVAILABLE && defined(_OPENMP)
    const int simd_end = (effective_size / SIMD_WIDTH) * SIMD_WIDTH;

    // SIMD 처리 가능한 구간
    #pragma omp simd safelen(SIMD_WIDTH)
    for (int t = 0; t < simd_end; t += SIMD_WIDTH) {
        for (int i = 0; i < SIMD_WIDTH && (t + i) < effective_size; ++i) {
            int64_t sample_pos = effective_start + t + i;
            size_t idx = ring_index(sample_pos);
            int frame_idx = start_offset + t + i;

            float x = frame[frame_idx * cfg_.channels + channel] * gain_ * window_[frame_idx];
            ring_accum_[channel_offset + idx] += x;
        }
    }

    // 나머지 처리 (스칼라)
    for (int t = simd_end; t < effective_size; ++t) {
        int64_t sample_pos = effective_start + t;
        size_t idx = ring_index(sample_pos);
        int frame_idx = start_offset + t;

        float x = frame[frame_idx * cfg_.channels + channel] * gain_ * window_[frame_idx];
        ring_accum_[channel_offset + idx] += x;
    }
#else
    add_frame_multi_windowed_scalar(effective_start, frame, start_offset,
                                   effective_size, channel, channel_offset);
#endif
}

// 스칼라 다채널 윈도우 적용 버전
void OLAAccumulator::add_frame_multi_windowed_scalar(int64_t effective_start, const float* frame,
                                                     int start_offset, int effective_size,
                                                     int channel, size_t channel_offset) {
    for (int t = 0; t < effective_size; ++t) {
        int64_t sample_pos = effective_start + t;
        size_t idx = ring_index(sample_pos);
        int frame_idx = start_offset + t;

        float x = frame[frame_idx * cfg_.channels + channel] * gain_ * window_[frame_idx];
        ring_accum_[channel_offset + idx] += x;
    }
}

// SIMD 최적화된 다채널 윈도우 미적용 버전
void OLAAccumulator::add_frame_multi_plain_simd(int64_t effective_start, const float* frame,
                                                int start_offset, int effective_size,
                                                int channel, size_t channel_offset) {
#if SIMD_AVAILABLE && defined(_OPENMP)
    const int simd_end = (effective_size / SIMD_WIDTH) * SIMD_WIDTH;

    // SIMD 처리 가능한 구간
    #pragma omp simd safelen(SIMD_WIDTH)
    for (int t = 0; t < simd_end; t += SIMD_WIDTH) {
        for (int i = 0; i < SIMD_WIDTH && (t + i) < effective_size; ++i) {
            int64_t sample_pos = effective_start + t + i;
            size_t idx = ring_index(sample_pos);
            int frame_idx = start_offset + t + i;

            float x = frame[frame_idx * cfg_.channels + channel] * gain_;
            ring_accum_[channel_offset + idx] += x;
        }
    }

    // 나머지 처리 (스칼라)
    for (int t = simd_end; t < effective_size; ++t) {
        int64_t sample_pos = effective_start + t;
        size_t idx = ring_index(sample_pos);
        int frame_idx = start_offset + t;

        float x = frame[frame_idx * cfg_.channels + channel] * gain_;
        ring_accum_[channel_offset + idx] += x;
    }
#else
    add_frame_multi_plain_scalar(effective_start, frame, start_offset,
                                effective_size, channel, channel_offset);
#endif
}

// 스칼라 다채널 윈도우 미적용 버전
void OLAAccumulator::add_frame_multi_plain_scalar(int64_t effective_start, const float* frame,
                                                  int start_offset, int effective_size,
                                                  int channel, size_t channel_offset) {
    for (int t = 0; t < effective_size; ++t) {
        int64_t sample_pos = effective_start + t;
        size_t idx = ring_index(sample_pos);
        int frame_idx = start_offset + t;

        float x = frame[frame_idx * cfg_.channels + channel] * gain_;
        ring_accum_[channel_offset + idx] += x;
    }
}

void OLAAccumulator::update_peak_meter(const float* data, int num_samples) {
    for (int i = 0; i < num_samples; ++i) {
        float abs_val = std::abs(data[i]);
        meter_peak_ = std::max(meter_peak_, abs_val);
    }
}

void OLAAccumulator::normalize_and_clear(float* dst, int num_samples, int64_t start_idx) {
    // 피드백 반영: normalize_and_clear SIMD 최적화
    const float eps = 1e-8f;

    if (cfg_.channels == 1) {
        // 단일 채널 SIMD 최적화
        normalize_and_clear_mono_simd(dst, num_samples, start_idx, eps);
    } else {
        // 다채널 SIMD 최적화
        normalize_and_clear_multi_simd(dst, num_samples, start_idx, eps);
    }
}

// SIMD 최적화된 단일 채널 정규화 및 소거
void OLAAccumulator::normalize_and_clear_mono_simd(float* dst, int num_samples,
                                                   int64_t start_idx, float eps) {
    // 범위 체크를 먼저 수행하여 유효한 구간만 처리
    int valid_start = 0;
    int valid_end = num_samples;

    // 앞쪽 무효 구간 처리
    while (valid_start < num_samples &&
           (start_idx + valid_start < 0 || start_idx + valid_start >= produced_samples_)) {
        dst[valid_start] = 0.0f;
        valid_start++;
    }

    // 뒤쪽 무효 구간 찾기
    while (valid_end > valid_start &&
           (start_idx + valid_end - 1 < 0 || start_idx + valid_end - 1 >= produced_samples_)) {
        dst[valid_end - 1] = 0.0f;
        valid_end--;
    }

    const int valid_size = valid_end - valid_start;
    const bool use_simd = (valid_size >= SIMD_WIDTH * 2) && SIMD_AVAILABLE;

    if (use_simd) {
#if SIMD_AVAILABLE && defined(_OPENMP)
        const int simd_end = valid_start + (valid_size / SIMD_WIDTH) * SIMD_WIDTH;

        // SIMD 처리 가능한 구간 - 정규화와 소거를 분리하여 메모리 접근 최적화
        #pragma omp simd aligned(dst:32) safelen(SIMD_WIDTH)
        for (int i = valid_start; i < simd_end; i += SIMD_WIDTH) {
            for (int j = 0; j < SIMD_WIDTH && (i + j) < valid_end; ++j) {
                int64_t sample_pos = start_idx + i + j;
                size_t idx = ring_index(sample_pos);

                // COLA 정규화 적용
                float norm = std::max(eps, norm_buffer_[idx]);
                dst[i + j] = ring_accum_[idx] / norm;
            }
        }

        // 버퍼 소거를 별도 루프로 분리 (메모리 접근 최적화)
        #pragma omp simd safelen(SIMD_WIDTH)
        for (int i = valid_start; i < simd_end; i += SIMD_WIDTH) {
            for (int j = 0; j < SIMD_WIDTH && (i + j) < valid_end; ++j) {
                int64_t sample_pos = start_idx + i + j;
                size_t idx = ring_index(sample_pos);
                ring_accum_[idx] = 0.0f;
            }
        }

        // 나머지 처리 (스칼라)
        for (int i = simd_end; i < valid_end; ++i) {
            int64_t sample_pos = start_idx + i;
            size_t idx = ring_index(sample_pos);

            float norm = std::max(eps, norm_buffer_[idx]);
            dst[i] = ring_accum_[idx] / norm;
            ring_accum_[idx] = 0.0f;
        }
#endif
    } else {
        // 스칼라 처리
        for (int i = valid_start; i < valid_end; ++i) {
            int64_t sample_pos = start_idx + i;
            size_t idx = ring_index(sample_pos);

            float norm = std::max(eps, norm_buffer_[idx]);
            dst[i] = ring_accum_[idx] / norm;
            ring_accum_[idx] = 0.0f;
        }
    }
}

// SIMD 최적화된 다채널 정규화 및 소거
void OLAAccumulator::normalize_and_clear_multi_simd(float* dst, int num_samples,
                                                    int64_t start_idx, float eps) {
    const bool use_simd = (num_samples >= SIMD_WIDTH * 2) && SIMD_AVAILABLE;

    if (use_simd) {
#if SIMD_AVAILABLE && defined(_OPENMP)
        const int simd_end = (num_samples / SIMD_WIDTH) * SIMD_WIDTH;

        // SIMD 처리 가능한 구간
        #pragma omp simd aligned(dst:32) safelen(SIMD_WIDTH)
        for (int i = 0; i < simd_end; i += SIMD_WIDTH) {
            for (int j = 0; j < SIMD_WIDTH && (i + j) < num_samples; ++j) {
                int sample_idx = i + j;
                int64_t sample_pos = start_idx + sample_idx;

                if (sample_pos < 0 || sample_pos >= produced_samples_) {
                    // 무효 샘플 처리
                    for (int c = 0; c < cfg_.channels; ++c) {
                        dst[sample_idx * cfg_.channels + c] = 0.0f;
                    }
                    continue;
                }

                size_t idx = ring_index(sample_pos);
                float norm = std::max(eps, norm_buffer_[idx]);

                // 채널별 처리
                for (int c = 0; c < cfg_.channels; ++c) {
                    size_t channel_offset = c * ring_len_;
                    size_t src_idx = channel_offset + idx;
                    size_t dst_idx = sample_idx * cfg_.channels + c;

                    dst[dst_idx] = ring_accum_[src_idx] / norm;
                    ring_accum_[src_idx] = 0.0f;
                }
            }
        }

        // 나머지 처리 (스칼라)
        for (int i = simd_end; i < num_samples; ++i) {
            int64_t sample_pos = start_idx + i;

            if (sample_pos < 0 || sample_pos >= produced_samples_) {
                for (int c = 0; c < cfg_.channels; ++c) {
                    dst[i * cfg_.channels + c] = 0.0f;
                }
                continue;
            }

            size_t idx = ring_index(sample_pos);
            float norm = std::max(eps, norm_buffer_[idx]);

            for (int c = 0; c < cfg_.channels; ++c) {
                size_t channel_offset = c * ring_len_;
                size_t src_idx = channel_offset + idx;
                size_t dst_idx = i * cfg_.channels + c;

                dst[dst_idx] = ring_accum_[src_idx] / norm;
                ring_accum_[src_idx] = 0.0f;
            }
        }
#endif
    } else {
        // 스칼라 처리
        for (int i = 0; i < num_samples; ++i) {
            int64_t sample_pos = start_idx + i;

            if (sample_pos < 0 || sample_pos >= produced_samples_) {
                for (int c = 0; c < cfg_.channels; ++c) {
                    dst[i * cfg_.channels + c] = 0.0f;
                }
                continue;
            }

            size_t idx = ring_index(sample_pos);
            float norm = std::max(eps, norm_buffer_[idx]);

            for (int c = 0; c < cfg_.channels; ++c) {
                size_t channel_offset = c * ring_len_;
                size_t src_idx = channel_offset + idx;
                size_t dst_idx = i * cfg_.channels + c;

                dst[dst_idx] = ring_accum_[src_idx] / norm;
                ring_accum_[src_idx] = 0.0f;
            }
        }
    }
}

} // namespace dsp