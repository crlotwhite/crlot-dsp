#include "OLAAccumulator.h"
#include "kernels.h"
#include "norm_builder.h"
#include "aos_to_soa.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <vector>

namespace dsp {

OLAAccumulator::OLAAccumulator(const OLAConfig& cfg)
    : cfg_(cfg), ring_len_(0),
      read_pos_(0), produced_(0),
      meter_peak_(0.0f), flushing_(false) {

    if (!cfg_.isValid()) {
        throw std::invalid_argument("Invalid OLA configuration");
    }

    // 링 버퍼 크기 계산
    ring_len_ = calculate_ring_size();

    // 채널별 RingBuffer 초기화
    for (size_t c = 0; c < cfg_.channels; ++c) {
        ring_.emplace_back(std::make_unique<dsp::ring::RingBuffer<float>>(ring_len_, false));
    }

    // 정규화 계수 버퍼 초기화
    norm_.resize(ring_len_, 0.0f);
    initialize_normalization();  // 안전한 기본값(1.0)으로 초기화

    // AoS 변환용 스크래치 버퍼 초기화 (최대 크기로 reserve)
    scratch_.reserve(cfg_.channels * cfg_.frame_size);
}

void OLAAccumulator::set_window(const float* w, int wlen) {
    if (w == nullptr) {
        throw std::invalid_argument("Window pointer cannot be null");
    }
    if (wlen != static_cast<int>(cfg_.frame_size)) {
        throw std::invalid_argument("Window size must match frame size");
    }

    // 윈도우 함수 복사
    window_.resize(wlen);
    std::copy(w, w + wlen, window_.begin());

    // COLA 정규화 계수 초기화
    initialize_normalization();
}

void OLAAccumulator::add_frame_SoA(const float* const* ch_frames, const float* window,
                                  size_t start_sample, size_t start_off, size_t size, float gain) {
    if (ch_frames == nullptr) {
        throw std::invalid_argument("Channel frames pointer cannot be null");
    }
    if (size == 0) {
        return;
    }

    // 경계 클램핑
    size_t eff_start = start_sample;
    size_t eff_size = size;

    if (start_off >= cfg_.frame_size) {
        return; // 전체 프레임이 범위 밖
    }

    if (start_off + size > cfg_.frame_size) {
        eff_size = cfg_.frame_size - start_off;
    }

    // 각 채널별로 처리
    for (size_t c = 0; c < cfg_.channels; ++c) {
        if (ch_frames[c] == nullptr) {
            throw std::invalid_argument("Channel frame pointer cannot be null");
        }

        // 윈도우 사용 정책: apply_window_inside면 내부 window_ 사용, 아니면 외부 window 사용
        const bool use_win = cfg_.apply_window_inside ? !window_.empty() : (window != nullptr);
        const float* w     = use_win ? (cfg_.apply_window_inside ? window_.data() : window) : nullptr;

        // RingBuffer에서 해당 구간 분할
        auto [span1, span2] = ring_[c]->split(eff_start, eff_size);

        // 첫 번째 스팬 처리
        if (!span1.empty()) {
            if (use_win) {
                // 윈도우 적용
                axpy_windowed(span1.data(), ch_frames[c] + start_off,
                             w + start_off, gain, span1.size());
            } else {
                // 윈도우 미적용
                axpy(span1.data(), ch_frames[c] + start_off, gain, span1.size());
            }
        }

        // 두 번째 스팬 처리 (래핑)
        if (!span2.empty()) {
            if (use_win) {
                // 윈도우 적용
                axpy_windowed(span2.data(), ch_frames[c] + start_off + span1.size(),
                             w + start_off + span1.size(), gain, span2.size());
            } else {
                // 윈도우 미적용
                axpy(span2.data(), ch_frames[c] + start_off + span1.size(), gain, span2.size());
            }
        }
    }

    // 생산된 샘플 수 업데이트
    produced_ = std::max(produced_, eff_start + eff_size);
}

void OLAAccumulator::push_frame_AoS(const float* interleaved, const float* window,
                                   size_t start_sample, size_t start_off, size_t size, float gain) {
    if (interleaved == nullptr) {
        throw std::invalid_argument("Interleaved input pointer cannot be null");
    }
    if (size == 0) {
        return;
    }

    // 경계 클램핑 (add_frame_SoA와 동일)
    size_t eff_start = start_sample;
    size_t eff_size = size;

    if (start_off >= cfg_.frame_size) {
        return; // 전체 프레임이 범위 밖
    }

    if (start_off + size > cfg_.frame_size) {
        eff_size = cfg_.frame_size - start_off;
    }

    // scratch 버퍼 리사이즈 (SoA 형식)
    scratch_.resize(cfg_.channels * eff_size);

    // 디인터리브: AoS -> SoA
    dsp::ola::deinterleave_to_scratch(interleaved + start_off * cfg_.channels,
                                     eff_size, cfg_.channels, scratch_.data());

    // 채널별 포인터 배열 생성
    std::vector<const float*> ch_frames(cfg_.channels);
    for (size_t c = 0; c < cfg_.channels; ++c) {
        ch_frames[c] = scratch_.data() + c * eff_size;
    }

    // 기존 SoA 메서드 호출
    add_frame_SoA(ch_frames.data(), window, eff_start, 0, eff_size, gain);
}

size_t OLAAccumulator::produce(float* const* ch_out, size_t n) {
    if (ch_out == nullptr) {
        throw std::invalid_argument("Output channel buffer cannot be null");
    }
    if (n == 0) {
        return 0;
    }

    // 채널별 null 포인터 체크
    for (size_t c = 0; c < cfg_.channels; ++c) {
        if (ch_out[c] == nullptr) {
            throw std::invalid_argument("Output channel buffer cannot be null");
        }
    }

    // 사용 가능한 샘플 수 계산
    size_t available = (produced_ > read_pos_) ? (produced_ - read_pos_) : 0;
    if (available == 0) {
        return 0;
    }

    if (available < n) {
        n = available;
    }

    // 각 채널별로 처리
    for (size_t c = 0; c < cfg_.channels; ++c) {
        if (ch_out[c] == nullptr) {
            throw std::invalid_argument("Output channel buffer cannot be null");
        }

        // RingBuffer에서 읽기 구간 분할
        auto [span1, span2] = ring_[c]->split(read_pos_, n);

        // 첫 번째 스팬 처리
        if (!span1.empty()) {
            size_t norm_start = (span1.data() - ring_[c]->data()) % ring_len_;
            normalize_and_clear(ch_out[c], span1.data(),
                               norm_.data() + norm_start, cfg_.eps, span1.size());
        }

        // 두 번째 스팬 처리 (래핑)
        if (!span2.empty()) {
            size_t offset = span1.size();
            size_t norm_start = (span2.data() - ring_[c]->data()) % ring_len_;
            normalize_and_clear(ch_out[c] + offset, span2.data(),
                               norm_.data() + norm_start, cfg_.eps, span2.size());
        }
    }

    // 읽기 위치 업데이트 (래핑 처리)
    read_pos_ = (read_pos_ + n) % ring_len_;

    // 피크 미터 업데이트 (첫 번째 채널만)
    if (cfg_.channels > 0) {
        update_peak_meter(ch_out[0], n);
    }

    return n;
}

void OLAAccumulator::flush() {
    flushing_ = true;
    // flush 시점에서 남은 꼬리(프레임 길이만큼)만 보장
    // 오래된 0 영역까지 과도하게 배출하지 않도록 제한
    produced_ = std::max(produced_, read_pos_ + cfg_.frame_size);
}

void OLAAccumulator::reset() {
    // RingBuffer 초기화 (모든 요소를 0으로)
    for (auto& rb : ring_) {
        std::fill(rb->data(), rb->data() + rb->capacity(), 0.0f);
    }

    // 인덱스 리셋
    read_pos_ = 0;
    produced_ = 0;

    // 메트릭 리셋
    meter_peak_ = 0.0f;
    flushing_ = false;

    // 윈도우 클리어 (SoA 버전에서는 윈도우를 재설정하지 않음)
    window_.clear();
    initialize_normalization();  // 안전한 기본값(1.0)으로 복구
}

size_t OLAAccumulator::calculate_ring_size() const {
    // 최소 중첩 수: ceil(N/H)
    size_t min_overlaps = (cfg_.frame_size + cfg_.hop_size - 1) / cfg_.hop_size;

    // 안전 마진 추가: 실시간 안전 여유(드롭 방지) +20 for large inputs
    size_t K = min_overlaps + 20;

    // 링 크기: K * H (홉 단위 정렬)
    return K * cfg_.hop_size;
}

void OLAAccumulator::initialize_normalization() {
    if (window_.empty()) {
        // 윈도우가 없으면 정규화 계수를 1.0으로 설정
        std::fill(norm_.begin(), norm_.end(), 1.0f);
        return;
    }

    // 윈도우가 내부에서 적용되는 경우에만 COLA 정규화 계산
    if (!cfg_.apply_window_inside) {
        // 윈도우가 외부에서 적용되면 정규화 계수를 1.0으로 설정
        std::fill(norm_.begin(), norm_.end(), 1.0f);
        return;
    }

    // 최적화된 선형 누적 COLA 정규화 계산
    dsp::ola::build_norm_linear(norm_.data(), window_.data(),
                               ring_len_, static_cast<size_t>(cfg_.frame_size),
                               static_cast<size_t>(cfg_.hop_size));
}

void OLAAccumulator::update_peak_meter(const float* data, size_t num_samples) {
    for (size_t i = 0; i < num_samples; ++i) {
        float abs_val = std::fabs(data[i]);
        meter_peak_ = std::max(meter_peak_, abs_val);
    }
}

} // namespace dsp