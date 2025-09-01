#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>
#include "dsp/ring/ring_buffer.h"
#include <memory>

namespace dsp {

/**
 * OLA 설정 구조체
 */
struct OLAConfig {
    int sample_rate;                    // 샘플링 레이트
    size_t frame_size;                  // N (프레임 크기)
    size_t hop_size;                    // H (홉 크기)
    size_t channels;                    // 채널 수
    float eps = 1e-8f;                  // 정규화 eps
    bool apply_window_inside;           // 내부 윈도우 적용 여부

    // 검증 함수
    bool isValid() const {
        return sample_rate > 0 && frame_size > 0 && hop_size > 0 &&
                channels > 0 && eps > 0.0f;
    }
};

/**
 * OLAAccumulator - Overlap-Add 누산기
 *
 * 주요 기능:
 * - 윈도우가 곱해진 프레임들을 시간축에 정확히 겹쳐 더함
 * - COLA(Constant Overlap-Add) 조건 만족을 위한 정규화
 * - 원형 버퍼를 통한 효율적인 스트리밍 처리
 * - 다채널 지원 (SoA 레이아웃)
 * - center 모드 지원 (패딩 오프셋 보정)
 * - 수치 안정성 보장
 * - AoS 입력 지원 (입구에서 SoA로 변환)
 *
 * 내부 아키텍처:
 * - SoA(Structure of Arrays) 레이아웃으로 메모리 접근 최적화
 * - AoS 입력은 push_frame_AoS()에서 입구 1회 변환 후 SoA 경로 사용
 * - 모든 내부 연산은 SoA를 유지하여 캐시 효율성 극대화
 *
 * 사용 패턴:
 * 1. OLAConfig로 초기화
 * 2. set_window()로 윈도우 함수 설정
 * 3. push_frame_SoA() 또는 push_frame_AoS()로 프레임 누산
 * 4. produce()로 연속 오디오 출력
 * 5. flush()로 남은 데이터 배출
 */
class OLAAccumulator {
public:
    /**
     * OLAAccumulator 생성자
     *
     * @param cfg OLA 설정
     * @throws std::invalid_argument 잘못된 설정값
     */
    explicit OLAAccumulator(const OLAConfig& cfg);

    /**
     * 소멸자
     */
    ~OLAAccumulator() = default;

    // 복사/이동 생성자 및 대입 연산자 삭제 (RAII)
    OLAAccumulator(const OLAAccumulator&) = delete;
    OLAAccumulator& operator=(const OLAAccumulator&) = delete;
    OLAAccumulator(OLAAccumulator&&) = delete;
    OLAAccumulator& operator=(OLAAccumulator&&) = delete;

    /**
     * 윈도우 함수 설정
     *
     * @param w 윈도우 함수 데이터 포인터
     * @param wlen 윈도우 길이 (frame_size와 일치해야 함)
     * @throws std::invalid_argument 잘못된 윈도우 크기
     */
    void set_window(const float* w, int wlen);


    /**
     * 프레임 누산 (SoA)
     *
     * @param ch_frames 채널별 프레임 데이터 [channels][frame_size]
     * @param window 윈도우 함수 (nullptr이면 적용 안 함)
     * @param start_sample 시작 샘플 위치
     * @param start_off 프레임 내 시작 오프셋
     * @param size 처리할 샘플 수
     * @param gain 게인 계수
     */
    void add_frame_SoA(const float* const* ch_frames, const float* window,
                      size_t start_sample, size_t start_off, size_t size, float gain);

    /**
     * 프레임 누산 (AoS 인터페이스)
     *
     * 인터리브된 AoS 입력을 받아 내부적으로 SoA로 변환하여 add_frame_SoA를 호출
     *
     * @param interleaved 인터리브된 프레임 데이터 [size * channels]
     *                   형식: [ch0_s0, ch1_s0, ..., ch0_s1, ch1_s1, ...]
     * @param window 윈도우 함수 (nullptr이면 적용 안 함)
     * @param start_sample 시작 샘플 위치
     * @param start_off 프레임 내 시작 오프셋
     * @param size 처리할 샘플 수
     * @param gain 게인 계수
     */
    void push_frame_AoS(const float* interleaved, const float* window,
                       size_t start_sample, size_t start_off, size_t size, float gain);

    /**
     * 누적 버퍼에서 연속 오디오 출력 (SoA)
     *
     * @param ch_out 채널별 출력 버퍼 [channels][num_samples]
     * @param n 요청 샘플 수
     * @return 실제 제공한 샘플 수
     */
    size_t produce(float* const* ch_out, size_t n);

    /**
     * 남은 꼬리까지 모두 출력
     */
    void flush();

    /**
     * 리셋 (초기 상태로 복원)
     */
    void reset();

    // === 상태 조회 ===

    /**
     * 생산된 총 샘플 수
     */
    size_t produced_samples() const { return produced_; }

    /**
     * 읽기 위치
     */
    size_t read_pos() const { return read_pos_; }

    /**
     * 피크 레벨 미터
     */
    float meter_peak() const { return meter_peak_; }

    /**
     * 설정 정보 반환
     */
    const OLAConfig& config() const { return cfg_; }

    /**
     * 현재 윈도우 설정 여부
     */
    bool has_window() const { return !window_.empty(); }

    /**
     * 링 버퍼 크기
     */
    size_t ring_size() const { return ring_len_; }

private:
    // === 설정 ===
    OLAConfig cfg_;
    std::vector<float> window_;         // 윈도우 함수 복사본

    // === 링 버퍼 (SoA) ===
    std::vector<std::unique_ptr<dsp::ring::RingBuffer<float>>> ring_;  // 채널별 RingBuffer
    std::vector<float> norm_;                    // COLA 정규화 계수 [ring_len_]
    size_t ring_len_;                            // 링 버퍼 크기

    // === AoS 변환용 스크래치 버퍼 ===
    std::vector<float> scratch_;                 // 재사용 가능한 디인터리브 버퍼

    // === 인덱스 관리 ===
    size_t read_pos_;                            // 읽기 위치
    size_t produced_;                            // 생산된 총 샘플 수

    // === 메트릭 ===
    float meter_peak_;                  // 피크 레벨
    bool flushing_;                     // flush 모드 여부

    // === 내부 함수 ===

    /**
     * 링 버퍼 크기 계산
     */
    size_t calculate_ring_size() const;

    /**
     * COLA 정규화 계수 초기화
     */
    void initialize_normalization();

    /**
     * 링 인덱스 계산 (모듈로 연산)
     */
    size_t ring_index(size_t sample_idx) const {
        return sample_idx % ring_len_;
    }

    /**
     * center 모드 오프셋 계산 (SoA에서는 사용하지 않음)
     */
    int64_t calculate_center_offset() const {
        return 0;  // SoA 버전에서는 center 모드 지원 안 함
    }

    /**
     * 피크 미터 업데이트
     */
    void update_peak_meter(const float* data, size_t num_samples);
};

} // namespace dsp