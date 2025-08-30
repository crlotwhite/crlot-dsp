#include "dsp/ola/OLAAccumulator.h"
#include "dsp/window/WindowLUT.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace dsp;

int main() {
    // 간단한 설정
    OLAConfig config;
    config.sample_rate = 48000;
    config.frame_size = 8;
    config.hop_size = 4;
    config.channels = 1;
    config.center = false;
    config.apply_window_inside = false;
    config.gain = 1.0f;

    OLAAccumulator ola(config);

    // 간단한 윈도우 (모두 1.0)
    std::vector<float> window(8, 1.0f);
    ola.set_window(window.data(), 8);

    std::cout << "Ring size: " << ola.ring_size() << std::endl;

    // 단위 임펄스 프레임들
    std::vector<float> frame1(8, 0.0f);
    frame1[0] = 1.0f;  // 첫 번째 샘플만 1

    std::vector<float> frame2(8, 0.0f);
    frame2[0] = 1.0f;  // 첫 번째 샘플만 1

    // 프레임 추가
    ola.push_frame(0, frame1.data());
    std::cout << "After frame 0: produced=" << ola.produced_samples() << std::endl;

    ola.push_frame(1, frame2.data());
    std::cout << "After frame 1: produced=" << ola.produced_samples() << std::endl;

    // 더 많은 프레임 추가하여 정상 상태 확인
    ola.push_frame(2, frame2.data());
    ola.push_frame(3, frame2.data());
    std::cout << "After more frames: produced=" << ola.produced_samples() << std::endl;

    // 출력 확인
    std::vector<float> output(24);
    int samples = ola.pull(output.data(), 24);
    std::cout << "Pulled " << samples << " samples" << std::endl;

    for (int i = 0; i < samples; ++i) {
        std::cout << "output[" << i << "] = " << output[i] << std::endl;
    }

    return 0;
}