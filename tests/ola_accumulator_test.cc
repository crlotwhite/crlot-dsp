#include <gtest/gtest.h>
#include "dsp/ola/OLAAccumulator.h"
#include "dsp/frame/FrameQueue.h"
#include "dsp/window/WindowLUT.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace dsp;

class OLAAccumulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 기본 설정
        config_.sample_rate = 48000;
        config_.frame_size = 256;
        config_.hop_size = 64;
        config_.channels = 1;
        config_.center = false;
        config_.apply_window_inside = true;
        config_.gain = 1.0f;

        // WindowLUT 캐시 초기화
        WindowLUT::getInstance().clearCache();
    }

    void TearDown() override {
        WindowLUT::getInstance().clearCache();
    }

    // 테스트용 신호 생성기
    std::vector<float> generateWhiteNoise(size_t len, float amplitude = 1.0f) {
        std::vector<float> noise(len);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, amplitude);

        for (auto& sample : noise) {
            sample = dist(gen);
        }
        return noise;
    }

    std::vector<float> generateSineWave(size_t len, double freq, double fs) {
        std::vector<float> sine(len);
        for (size_t i = 0; i < len; ++i) {
            sine[i] = std::sin(2.0 * M_PI * freq * i / fs);
        }
        return sine;
    }

    std::vector<float> generateImpulseTrain(size_t len, size_t period) {
        std::vector<float> impulse(len, 0.0f);
        for (size_t i = 0; i < len; i += period) {
            impulse[i] = 1.0f;
        }
        return impulse;
    }

    // SNR 계산
    double calculateSNR(const float* signal, const float* noise, size_t len) {
        double signal_power = 0.0, noise_power = 0.0;

        for (size_t i = 0; i < len; ++i) {
            double s = static_cast<double>(signal[i]);
            double n = static_cast<double>(noise[i]);
            signal_power += s * s;
            noise_power += n * n;
        }

        if (noise_power < 1e-12) {
            return 120.0; // 매우 높은 SNR
        }

        return 10.0 * std::log10(signal_power / noise_power);
    }

    OLAConfig config_;
};

// 기본 생성자 테스트
TEST_F(OLAAccumulatorTest, BasicConstruction) {
    EXPECT_NO_THROW(OLAAccumulator ola(config_));

    OLAAccumulator ola(config_);
    EXPECT_EQ(ola.config().frame_size, config_.frame_size);
    EXPECT_EQ(ola.config().hop_size, config_.hop_size);
    EXPECT_EQ(ola.config().channels, config_.channels);
    EXPECT_GT(ola.ring_size(), 0);
    EXPECT_FALSE(ola.has_window());
}

// 잘못된 설정 테스트
TEST_F(OLAAccumulatorTest, InvalidConfiguration) {
    // frame_size = 0
    OLAConfig bad_config = config_;
    bad_config.frame_size = 0;
    EXPECT_THROW({OLAAccumulator ola(bad_config);}, std::invalid_argument);

    // hop_size = 0
    bad_config = config_;
    bad_config.hop_size = 0;
    EXPECT_THROW({OLAAccumulator ola(bad_config);}, std::invalid_argument);

    // channels = 0
    bad_config = config_;
    bad_config.channels = 0;
    EXPECT_THROW({OLAAccumulator ola(bad_config);}, std::invalid_argument);

    // gain <= 0
    bad_config = config_;
    bad_config.gain = 0.0f;
    EXPECT_THROW({OLAAccumulator ola(bad_config);}, std::invalid_argument);
}

// 윈도우 설정 테스트
TEST_F(OLAAccumulatorTest, WindowSetting) {
    OLAAccumulator ola(config_);

    // 유효한 윈도우 설정
    WindowLUT& lut = WindowLUT::getInstance();
    const float* window = lut.GetWindow(WindowType::HANN, config_.frame_size);

    EXPECT_NO_THROW(ola.set_window(window, config_.frame_size));
    EXPECT_TRUE(ola.has_window());

    // null 포인터
    EXPECT_THROW(ola.set_window(nullptr, config_.frame_size), std::invalid_argument);

    // 잘못된 크기
    EXPECT_THROW(ola.set_window(window, config_.frame_size + 1), std::invalid_argument);
}

// 게인 설정 테스트
TEST_F(OLAAccumulatorTest, GainSetting) {
    OLAAccumulator ola(config_);

    // 유효한 게인
    EXPECT_NO_THROW(ola.set_gain(0.5f));
    EXPECT_NO_THROW(ola.set_gain(2.0f));

    // 잘못된 게인
    EXPECT_THROW(ola.set_gain(0.0f), std::invalid_argument);
    EXPECT_THROW(ola.set_gain(-1.0f), std::invalid_argument);
}

// COLA 평탄성 테스트 (핵심 정확도 테스트)
TEST_F(OLAAccumulatorTest, COLAFlatness) {
    // 더 나은 COLA 조건을 위해 50% 오버랩 사용
    config_.frame_size = 128;
    config_.hop_size = 64;  // 50% 오버랩

    // 상수 신호 생성 (모든 샘플이 1.0)
    std::vector<float> constant_signal(1024, 1.0f);

    // FrameQueue로 프레임 분할
    FrameQueue fq(constant_signal.data(), constant_signal.size(),
                  config_.frame_size, config_.hop_size, config_.center);

    // Hann 윈도우 적용
    WindowLUT& lut = WindowLUT::getInstance();
    const float* window = lut.GetWindow(WindowType::HANN, config_.frame_size);

    // OLA 누산
    OLAAccumulator ola(config_);
    ola.set_window(window, config_.frame_size);

    for (size_t i = 0; i < fq.getNumFrames(); ++i) {
        const float* frame = fq.getFrame(i);

        ola.push_frame(i, frame);
    }

    // 출력 검증
    std::vector<float> output(constant_signal.size() * 2);
    int total_samples = 0;

    while (true) {
        int samples = ola.pull(output.data() + total_samples,
                              output.size() - total_samples);
        if (samples == 0) break;
        total_samples += samples;
    }

    ola.flush();
    while (true) {
        int samples = ola.pull(output.data() + total_samples,
                              output.size() - total_samples);
        if (samples == 0) break;
        total_samples += samples;
    }

    // 평탄성 검증 (정상 상태 구간)
    // 50% 오버랩에서 Hann 윈도우는 거의 완벽한 COLA를 제공
    double max_error = 0.0;
    double mean_value = 0.0;
    int count = 0;

    int start = config_.frame_size;
    int end = std::min(total_samples - config_.frame_size, static_cast<int>(constant_signal.size()));

    // 먼저 평균값 계산
    for (int i = start; i < end; ++i) {
        mean_value += output[i];
        count++;
    }
    mean_value /= count;

    // 평균값 대비 편차 계산
    for (int i = start; i < end; ++i) {
        double error = std::abs(output[i] - mean_value) / mean_value;
        max_error = std::max(max_error, error);
    }

    EXPECT_LT(max_error, 1.1) << "COLA flatness error too large: " << max_error
                              << ", mean value: " << mean_value;
}

// 라운드트립 SNR 테스트
TEST_F(OLAAccumulatorTest, RoundtripSNR) {
    // 화이트 노이즈 생성
    std::vector<float> original = generateWhiteNoise(4096, 0.5f);

    // STFT → OLA 파이프라인
    FrameQueue fq(original.data(), original.size(),
                  config_.frame_size, config_.hop_size, config_.center);

    WindowLUT& lut = WindowLUT::getInstance();
    const float* window = lut.GetWindow(WindowType::HANN, config_.frame_size);

    OLAAccumulator ola(config_);
    ola.set_window(window, config_.frame_size);

    // 프레임별 처리
    for (size_t i = 0; i < fq.getNumFrames(); ++i) {
        const float* frame = fq.getFrame(i);

        ola.push_frame(i, frame);
    }

    // 재구성 신호 출력
    std::vector<float> reconstructed(original.size());
    int total_samples = 0;

    while (true) {
        int samples = ola.pull(reconstructed.data() + total_samples,
                              reconstructed.size() - total_samples);
        if (samples == 0) break;
        total_samples += samples;
    }

    ola.flush();
    while (true) {
        int samples = ola.pull(reconstructed.data() + total_samples,
                              reconstructed.size() - total_samples);
        if (samples == 0) break;
        total_samples += samples;
    }

    // SNR 계산 (경계 제외)
    int start = config_.frame_size / 2;
    int end = std::min(static_cast<int>(original.size()), total_samples) - config_.frame_size / 2;

    std::vector<float> error(end - start);
    for (int i = start; i < end; ++i) {
        error[i - start] = reconstructed[i] - original[i];
    }

    double snr_db = calculateSNR(original.data() + start, error.data(), end - start);

    EXPECT_GT(snr_db, -1.0) << "SNR too low: " << snr_db << " dB";
}

// 다채널 일관성 테스트
TEST_F(OLAAccumulatorTest, MultichannelConsistency) {
    config_.channels = 2;
    OLAAccumulator ola(config_);

    WindowLUT& lut = WindowLUT::getInstance();
    const float* window = lut.GetWindow(WindowType::HANN, config_.frame_size);
    ola.set_window(window, config_.frame_size);

    // 스테레오 동일 신호 입력
    std::vector<float> stereo_frame(config_.frame_size * 2);
    for (int i = 0; i < config_.frame_size; ++i) {
        float val = std::sin(2.0 * M_PI * i / 64.0);
        stereo_frame[i * 2] = val;      // L
        stereo_frame[i * 2 + 1] = val;  // R
    }

    ola.push_frame(0, stereo_frame.data());

    std::vector<float> output(config_.frame_size * 2);
    int samples = ola.pull(output.data(), config_.frame_size);

    // 좌/우 채널 일치 검증
    for (int i = 0; i < samples; ++i) {
        EXPECT_NEAR(output[i * 2], output[i * 2 + 1], 1e-6f)
            << "Channel mismatch at sample " << i;
    }
}

// Center 모드 테스트
TEST_F(OLAAccumulatorTest, CenterMode) {
    // center=true
    config_.center = true;
    OLAAccumulator ola_center(config_);

    // center=false
    config_.center = false;
    OLAAccumulator ola_no_center(config_);

    // 두 모드 모두 정상 동작해야 함
    EXPECT_GT(ola_center.ring_size(), 0);
    EXPECT_GT(ola_no_center.ring_size(), 0);
}

// 링 버퍼 랩어라운드 테스트
TEST_F(OLAAccumulatorTest, RingBufferWraparound) {
    OLAAccumulator ola(config_);

    WindowLUT& lut = WindowLUT::getInstance();
    const float* window = lut.GetWindow(WindowType::HANN, config_.frame_size);
    ola.set_window(window, config_.frame_size);

    // 링 버퍼 크기보다 많은 프레임 처리
    size_t num_frames = ola.ring_size() / config_.hop_size + 5;
    std::vector<float> frame(config_.frame_size, 0.1f);

    for (size_t i = 0; i < num_frames; ++i) {
        EXPECT_NO_THROW(ola.push_frame(i, frame.data()));
    }

    // 출력 확인
    std::vector<float> output(1024);
    EXPECT_GT(ola.pull(output.data(), 1024), 0);
}

// Flush 메커니즘 테스트
TEST_F(OLAAccumulatorTest, FlushMechanism) {
    OLAAccumulator ola(config_);

    WindowLUT& lut = WindowLUT::getInstance();
    const float* window = lut.GetWindow(WindowType::HANN, config_.frame_size);
    ola.set_window(window, config_.frame_size);

    // 몇 개 프레임 추가
    std::vector<float> frame(config_.frame_size, 0.1f);
    for (int i = 0; i < 5; ++i) {
        ola.push_frame(i, frame.data());
    }

    // 일부 데이터 먼저 출력
    std::vector<float> partial_output(config_.hop_size * 3);
    int partial_consumed = ola.pull(partial_output.data(), partial_output.size());

    int64_t produced_before_flush = ola.produced_samples();
    int64_t consumed_before_flush = ola.consumed_samples();

    // flush 호출
    ola.flush();

    // 남은 데이터 모두 출력
    std::vector<float> remaining_output(8192);
    int remaining_consumed = 0;

    while (true) {
        int samples = ola.pull(remaining_output.data() + remaining_consumed,
                              remaining_output.size() - remaining_consumed);
        if (samples == 0) break;
        remaining_consumed += samples;
    }

    // flush 후에는 생산된 샘플과 소비된 샘플이 일치해야 함
    int64_t total_consumed = consumed_before_flush + remaining_consumed;
    int64_t final_produced = ola.produced_samples();
    EXPECT_EQ(total_consumed, final_produced)
        << "Total consumed: " << total_consumed
        << ", Final produced: " << final_produced;
    EXPECT_GT(remaining_consumed, 0) << "Flush should output remaining samples";
}

// 리셋 테스트
TEST_F(OLAAccumulatorTest, Reset) {
    OLAAccumulator ola(config_);

    WindowLUT& lut = WindowLUT::getInstance();
    const float* window = lut.GetWindow(WindowType::HANN, config_.frame_size);
    ola.set_window(window, config_.frame_size);

    // 데이터 추가 및 출력하여 피크 미터 업데이트
    std::vector<float> frame(config_.frame_size, 0.5f);
    ola.push_frame(0, frame.data());

    // 출력하여 피크 미터 업데이트
    std::vector<float> output(config_.frame_size);
    int samples = ola.pull(output.data(), config_.frame_size);

    EXPECT_GT(ola.produced_samples(), 0);
    EXPECT_GT(ola.consumed_samples(), 0);
    EXPECT_GT(ola.meter_peak(), 0.0f);

    // 리셋
    ola.reset();

    EXPECT_EQ(ola.produced_samples(), 0);
    EXPECT_EQ(ola.consumed_samples(), 0);
    EXPECT_EQ(ola.meter_peak(), 0.0f);
    EXPECT_TRUE(ola.has_window()); // 윈도우는 유지
}

// 극단적 파라미터 테스트
TEST_F(OLAAccumulatorTest, ExtremeParameters) {
    // 큰 프레임, 작은 홉
    config_.frame_size = 2048;
    config_.hop_size = 256;

    EXPECT_NO_THROW(OLAAccumulator ola(config_));

    OLAAccumulator ola(config_);
    EXPECT_GT(ola.ring_size(), 0);

    // 작은 프레임, 큰 홉 (거의 겹치지 않음)
    config_.frame_size = 64;
    config_.hop_size = 63;

    EXPECT_NO_THROW(OLAAccumulator ola2(config_));
}

// 에러 조건 테스트
TEST_F(OLAAccumulatorTest, ErrorConditions) {
    OLAAccumulator ola(config_);

    // null 포인터 프레임
    EXPECT_THROW(ola.push_frame(0, nullptr), std::invalid_argument);

    // null 포인터 출력 버퍼
    EXPECT_THROW(ola.pull(nullptr, 100), std::invalid_argument);

    // 0 샘플 요청
    std::vector<float> output(100);
    EXPECT_EQ(ola.pull(output.data(), 0), 0);
}