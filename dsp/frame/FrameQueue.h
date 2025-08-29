#pragma once

#include <vector>
#include <cstddef>

namespace dsp {

enum class PadMode {
    CONSTANT,  // Zero padding
    REFLECT,   // Reflect padding
    EDGE       // Edge padding
};

/**
 * FrameQueue - 오디오 신호를 프레임 단위로 분할하는 클래스
 *
 * 주요 기능:
 * - 입력 신호를 지정된 프레임 크기와 홉 크기로 분할
 * - center 옵션으로 앞뒤 패딩 지원
 * - 다양한 패딩 모드 지원
 * - FFT 호환성을 위한 AoS(Array of Structures) 메모리 레이아웃
 */
class FrameQueue {
public:
    /**
     * FrameQueue 생성자
     *
     * @param in 입력 오디오 데이터 포인터
     * @param len 입력 데이터 길이
     * @param frame_size 프레임 크기 (샘플 수)
     * @param hop_size 홉 크기 (프레임 간 이동 거리)
     * @param center 중앙 정렬 여부 (true시 앞뒤 패딩 적용)
     * @param pad_mode 패딩 모드
     */
    FrameQueue(const float* in, size_t len, size_t frame_size, size_t hop_size,
               bool center = true, PadMode pad_mode = PadMode::CONSTANT);

    /**
     * 총 프레임 개수 반환
     */
    size_t getNumFrames() const { return num_frames_; }

    /**
     * 프레임 크기 반환
     */
    size_t getFrameSize() const { return frame_size_; }

    /**
     * 홉 크기 반환
     */
    size_t getHopSize() const { return hop_size_; }

    /**
     * 지정된 인덱스의 프레임 데이터 반환
     *
     * @param frame_idx 프레임 인덱스
     * @return 프레임 데이터 포인터 (frame_size 길이)
     */
    const float* getFrame(size_t frame_idx) const;

    /**
     * 지정된 인덱스의 프레임을 출력 버퍼에 복사
     *
     * @param frame_idx 프레임 인덱스
     * @param output 출력 버퍼 (최소 frame_size 크기)
     */
    void copyFrame(size_t frame_idx, float* output) const;

    /**
     * 모든 프레임 데이터에 대한 직접 접근
     * AoS 레이아웃: [frame0_sample0, frame0_sample1, ..., frame1_sample0, ...]
     */
    const std::vector<float>& getAllFrames() const { return frames_; }

private:
    size_t frame_size_;
    size_t hop_size_;
    size_t num_frames_;
    bool center_;
    PadMode pad_mode_;

    std::vector<float> frames_;  // AoS 레이아웃으로 모든 프레임 저장

    /**
     * 패딩된 입력 데이터 생성
     */
    std::vector<float> createPaddedInput(const float* in, size_t len) const;

    /**
     * 프레임 개수 계산
     */
    size_t calculateNumFrames(size_t padded_len) const;

    /**
     * 패딩 값 계산
     */
    float getPaddingValue(const std::vector<float>& padded_input, int idx) const;
};

} // namespace dsp