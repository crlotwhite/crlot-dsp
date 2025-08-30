#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace dsp {

/**
 * OLA 설정 구조체
 */
struct OLAConfig {
    int sample_rate;                    // 샘플링 레이트
    int frame_size;                     // N (프레임 크기)
    int hop_size;                       // H (홉 크기)
    int channels = 1;                   // 채널 수
    bool center = false;                // center 모드 (패딩 오프셋 보정)
    bool apply_window_inside = false;   // 내부 윈도우 적용 여부
    float gain = 1.0f;                  // 전역 게인

    // 검증 함수
    bool isValid() const {
        return sample_rate > 0 && frame_size > 0 && hop_size > 0 &&
               channels > 0 && gain > 0.0f;
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
 *
 * 사용 패턴:
 * 1. OLAConfig로 초기화
 * 2. set_window()로 윈도우 함수 설정
 * 3. push_frame()으로 프레임 누산
 * 4. pull()로 연속 오디오 출력
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
     * 전역 게인 설정
     *
     * @param g 게인 값 (> 0)
     */
    void set_gain(float g);

    /**
     * 프레임 누산
     *
     * @param frame_index 프레임 인덱스 (타임라인 위치 결정)
     * @param frame 프레임 데이터 [frame_size * channels]
     * @throws std::invalid_argument 잘못된 프레임 데이터
     */
    void push_frame(int64_t frame_index, const float* frame);

    /**
     * 누적 버퍼에서 연속 오디오 출력
     *
     * @param dst 출력 버퍼 [num_samples * channels]
     * @param num_samples 요청 샘플 수
     * @return 실제 제공한 샘플 수
     */
    int pull(float* dst, int num_samples);

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
    int64_t produced_samples() const { return produced_samples_; }

    /**
     * 소비된 총 샘플 수
     */
    int64_t consumed_samples() const { return consumed_samples_; }

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
    float gain_;                        // 현재 게인

    // === 링 버퍼 ===
    std::vector<float> ring_accum_;     // 원형 누산 버퍼 [ring_len * channels]
    std::vector<float> norm_buffer_;    // COLA 정규화 계수 [ring_len]
    size_t ring_len_;                   // 링 버퍼 크기

    // === 인덱스 관리 ===
    int64_t write_head_;                // 쓰기 위치 (절대 샘플 인덱스)
    int64_t read_head_;                 // 읽기 위치 (절대 샘플 인덱스)
    int64_t produced_samples_;          // 생산된 총 샘플 수
    int64_t consumed_samples_;          // 소비된 총 샘플 수

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
     * 단일 채널 프레임 누산
     */
    void add_frame_mono(int64_t start_sample, const float* frame);

    /**
     * 다채널 프레임 누산
     */
    void add_frame_multi(int64_t start_sample, const float* frame);

    /**
     * 링 인덱스 계산 (모듈로 연산)
     */
    size_t ring_index(int64_t sample_idx) const {
        return static_cast<size_t>(sample_idx % static_cast<int64_t>(ring_len_));
    }

    /**
     * center 모드 오프셋 계산
     */
    int64_t calculate_center_offset() const {
        return cfg_.center ? (cfg_.frame_size / 2) : 0;
    }

    /**
     * 피크 미터 업데이트
     */
    void update_peak_meter(const float* data, int num_samples);

    /**
     * 정규화 적용 및 버퍼 소거
     */
    void normalize_and_clear(float* dst, int num_samples, int64_t start_idx);
};

} // namespace dsp