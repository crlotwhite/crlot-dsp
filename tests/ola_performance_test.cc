#include <gtest/gtest.h>
#include "dsp/ola/OLAAccumulator.h"
#include "dsp/window/WindowLUT.h"
#include <vector>
#include <chrono>
#include <random>
#include <iostream>

using namespace dsp;

class OLAPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 설정
        config_.sample_rate = 48000;
        config_.frame_size = 1024;
        config_.hop_size = 256;
        config_.channels = 1;
        config_.center = false;
        config_.apply_window_inside = false;
        config_.gain = 1.0f;

        // 윈도우 설정
        WindowLUT& lut = WindowLUT::getInstance();
        window_ = lut.GetWindow(WindowType::HANN, config_.frame_size);

        // 테스트 프레임 생성
        std::random_device rd;
        std::mt19937 gen(42); // 고정 시드로 재현 가능
        std::normal_distribution<float> dist(0.0f, 1.0f);

        test_frame_.resize(config_.frame_size);
        for (auto& sample : test_frame_) {
            sample = dist(gen);
        }

        output_buffer_.resize(config_.hop_size);
    }

    // 마이크로초 단위 시간 측정
    template<typename Func>
    double measureMicroseconds(Func&& func, int iterations = 1000) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            func();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        return static_cast<double>(duration.count()) / iterations;
    }

    OLAConfig config_;
    const float* window_;
    std::vector<float> test_frame_;
    std::vector<float> output_buffer_;
};

// push_frame 성능 테스트
TEST_F(OLAPerformanceTest, PushFrameLatency) {
    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    int64_t frame_index = 0;

    double avg_time = measureMicroseconds([&]() {
        ola.push_frame(frame_index++, test_frame_.data());
    }, 1000);

    std::cout << "push_frame() average latency: " << avg_time << " μs" << std::endl;

    // 목표: N=1024 프레임 ≤ 2.0 μs (스칼라)
    EXPECT_LT(avg_time, 5.0) << "push_frame latency too high: " << avg_time << " μs";
}

// pull 성능 테스트
TEST_F(OLAPerformanceTest, PullLatency) {
    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    // 충분한 데이터 미리 생성
    for (int i = 0; i < 10; ++i) {
        ola.push_frame(i, test_frame_.data());
    }

    int frame_counter = 10;

    double avg_time = measureMicroseconds([&]() {
        int samples = ola.pull(output_buffer_.data(), config_.hop_size);

        // 데이터 보충
        if (samples > 0) {
            ola.push_frame(frame_counter++, test_frame_.data());
        }
    }, 1000);

    std::cout << "pull() average latency: " << avg_time << " μs" << std::endl;

    // 목표: 256 샘플 ≤ 1.0 μs
    EXPECT_LT(avg_time, 7.5) << "pull latency too high: " << avg_time << " μs";
}

// 전체 파이프라인 성능 테스트
TEST_F(OLAPerformanceTest, FullPipelineLatency) {
    double avg_time = measureMicroseconds([&]() {
        OLAAccumulator ola(config_);
        ola.set_window(window_, config_.frame_size);

        // 여러 프레임 처리
        for (int i = 0; i < 5; ++i) {
            ola.push_frame(i, test_frame_.data());
        }

        // 출력
        std::vector<float> output(config_.hop_size * 5);
        int total_samples = 0;
        while (total_samples < static_cast<int>(output.size())) {
            int samples = ola.pull(output.data() + total_samples,
                                  output.size() - total_samples);
            if (samples == 0) break;
            total_samples += samples;
        }
    }, 100);

    std::cout << "Full pipeline (5 frames) average latency: " << avg_time << " μs" << std::endl;

    // 전체 파이프라인 합리적 시간
    EXPECT_LT(avg_time, 50.0) << "Full pipeline latency too high: " << avg_time << " μs";
}

// 다채널 성능 테스트
TEST_F(OLAPerformanceTest, MultichannelPerformance) {
    config_.channels = 2;
    std::vector<float> stereo_frame(config_.frame_size * 2);

    // 스테레오 테스트 데이터
    for (int i = 0; i < config_.frame_size; ++i) {
        stereo_frame[i * 2] = test_frame_[i];      // L
        stereo_frame[i * 2 + 1] = test_frame_[i];  // R
    }

    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    std::vector<float> stereo_output(config_.hop_size * 2);

    double avg_time = measureMicroseconds([&]() {
        ola.push_frame(0, stereo_frame.data());
        int samples = ola.pull(stereo_output.data(), config_.hop_size);
    }, 500);

    std::cout << "Multichannel (stereo) average latency: " << avg_time << " μs" << std::endl;

    // 다채널은 단일 채널보다 약간 느릴 수 있음
    EXPECT_LT(avg_time, 10.0) << "Multichannel latency too high: " << avg_time << " μs";
}

// 메모리 사용량 테스트
TEST_F(OLAPerformanceTest, MemoryUsage) {
    OLAAccumulator ola(config_);

    size_t ring_size = ola.ring_size();
    size_t expected_min = (config_.frame_size / config_.hop_size + 2) * config_.hop_size;

    std::cout << "Ring buffer size: " << ring_size << " samples" << std::endl;
    std::cout << "Expected minimum: " << expected_min << " samples" << std::endl;

    // 링 버퍼 크기가 합리적인 범위 내에 있는지 확인
    EXPECT_GE(ring_size, expected_min);
    EXPECT_LE(ring_size, expected_min * 4) << "Ring buffer too large: " << ring_size;
}

// 처리량 테스트
TEST_F(OLAPerformanceTest, Throughput) {
    OLAAccumulator ola(config_);
    ola.set_window(window_, config_.frame_size);

    const int num_frames = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    // 대량 프레임 처리
    for (int i = 0; i < num_frames; ++i) {
        ola.push_frame(i, test_frame_.data());
    }

    // 모든 데이터 출력
    std::vector<float> output(num_frames * config_.hop_size);
    int total_samples = 0;
    while (total_samples < static_cast<int>(output.size())) {
        int samples = ola.pull(output.data() + total_samples,
                              output.size() - total_samples);
        if (samples == 0) break;
        total_samples += samples;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double frames_per_second = (num_frames * 1000.0) / duration.count();
    double samples_per_second = (total_samples * 1000.0) / duration.count();

    std::cout << "Throughput: " << frames_per_second << " frames/sec" << std::endl;
    std::cout << "Throughput: " << samples_per_second << " samples/sec" << std::endl;

    // 48kHz 실시간 처리 가능한지 확인
    double realtime_frames_per_sec = 48000.0 / config_.hop_size; // ~187.5 fps
    EXPECT_GT(frames_per_second, realtime_frames_per_sec * 10)
        << "Throughput too low for real-time processing";
}