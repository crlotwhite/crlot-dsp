#pragma once

#include <vector>
#include <cstddef>

namespace dsp {

/**
 * 경계 조건 처리 모드
 */
enum class BoundaryMode {
    ZERO_PAD,   // 부족한 프레임을 제로 패딩
    DROP        // 부족한 프레임을 드롭
};

/**
 * Framer - 실시간 스트리밍 오디오 프레이밍 클래스
 *
 * 주요 기능:
 * - push/pop API를 통한 실시간 스트리밍 처리
 * - 인터리브 유지 (mono/stereo)
 * - 정밀한 경계 처리 (부족 프레임 zero-pad or 보류)
 * - 다채널 지원
 * - 프레이밍 호환성: len = floor((N - frame)/hop) + 1
 */
class Framer {
public:
    /**
     * Framer 생성자
     */
    Framer();

    /**
     * 소멸자
     */
    ~Framer() = default;

    /**
     * 프레이머 파라미터 설정
     *
     * @param frame_size 프레임 크기 (샘플 수)
     * @param hop_size 홉 크기 (프레임 간 이동 거리)
     * @param channels 채널 수 (1=mono, 2=stereo, etc.)
     * @param boundary_mode 경계 조건 처리 모드
     */
    void set_params(size_t frame_size, size_t hop_size, size_t channels = 1,
                   BoundaryMode boundary_mode = BoundaryMode::ZERO_PAD);

    /**
     * 인터리브된 오디오 데이터 추가
     *
     * @param interleaved 인터리브된 입력 데이터 (channels * frames 크기)
     * @param frames 입력 프레임 수 (샘플 수가 아닌 프레임 수)
     * @return 성공 여부
     */
    bool push(const float* interleaved, size_t frames);

    /**
     * 프레임 데이터 추출
     *
     * @param out_frame 출력 프레임 버퍼 (channels * frame_size 크기)
     * @return 프레임 추출 성공 여부
     */
    bool pop(float* out_frame);

    /**
     * 사용 가능한 프레임 개수 반환
     */
    size_t available_frames() const;

    /**
     * 내부 버퍼 초기화
     */
    void reset();

    /**
     * 프레임 크기 반환
     */
    size_t frame_size() const { return frame_size_; }

    /**
     * 홉 크기 반환
     */
    size_t hop_size() const { return hop_size_; }

    /**
     * 채널 수 반환
     */
    size_t channels() const { return channels_; }

    /**
     * 경계 모드 반환
     */
    BoundaryMode boundary_mode() const { return boundary_mode_; }

    /**
     * 내부 버퍼 크기 반환 (디버깅용)
     */
    size_t buffer_size() const { return buffer_.size(); }

private:
    size_t frame_size_;
    size_t hop_size_;
    size_t channels_;
    BoundaryMode boundary_mode_;

    std::vector<float> buffer_;     // 인터리브된 내부 버퍼
    size_t write_pos_;              // 쓰기 위치 (샘플 단위)
    size_t read_pos_;               // 읽기 위치 (샘플 단위)

    bool params_set_;               // 파라미터 설정 여부

    /**
     * 사용 가능한 프레임 수 계산
     */
    size_t calculate_available_frames() const;

    /**
     * 버퍼 크기 조정
     */
    void resize_buffer_if_needed(size_t required_size);

    /**
     * 프레임 추출 (내부 구현)
     */
    bool extract_frame(float* out_frame);
};

} // namespace dsp