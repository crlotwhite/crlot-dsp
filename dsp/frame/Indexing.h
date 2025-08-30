#pragma once

namespace dsp {

// Forward declaration - PadMode is defined in FrameQueue.h
enum class PadMode;

/**
 * 안전한 reflect-101 인덱스 매핑 함수
 *
 * reflect-101은 경계에서 반복되지 않는 반사를 의미합니다.
 * 예: [1,2,3,4] -> [3,2,1,2,3,4,3,2]
 *
 * @param i 원본 인덱스 (음수 가능)
 * @param n 배열 크기 (> 0)
 * @return 유효한 인덱스 [0, n-1]
 */
inline int reflect101(int i, int n) {
    if (n <= 1) return 0;

    // 인덱스가 유효 범위에 있을 때까지 반사 적용
    while (i < 0 || i >= n) {
        if (i < 0) {
            // 음수 인덱스: -1 -> 0, -2 -> 1, -3 -> 2, ...
            i = -i - 1;
        } else {
            // 범위 초과: n -> n-2, n+1 -> n-3, ...
            i = 2 * n - 2 - i;
        }
    }

    return i;
}

/**
 * 안전한 패딩 값 계산 함수
 *
 * @param data 원본 데이터 포인터
 * @param len 원본 데이터 길이
 * @param idx 요청된 인덱스 (음수 가능)
 * @param pad_mode 패딩 모드
 * @return 패딩된 값
 */
inline float getPaddingValueSafe(const float* data, int len, int idx, PadMode pad_mode) {
    if (len <= 0) {
        return 0.0f;
    }

    switch (pad_mode) {
        case PadMode::CONSTANT:
            return 0.0f;

        case PadMode::EDGE:
            if (idx < 0) {
                return data[0];
            } else if (idx >= len) {
                return data[len - 1];
            } else {
                return data[idx];
            }

        case PadMode::REFLECT: {
            int safe_idx = reflect101(idx, len);
            return data[safe_idx];
        }

        default:
            return 0.0f;
    }
}

} // namespace dsp