#include <gtest/gtest.h>
#include "dsp/ola/OLAAccumulator.h"
#include <vector>
#include <cmath>
#include <random>

using namespace dsp;

class OLAAccumulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 기본 설정
        config_.sample_rate = 48000;
        config_.frame_size = 256;
        config_.hop_size = 64;
        config_.channels = 1;
        config_.eps = 1e-8f;
        config_.apply_window_inside = true;
    }

    // 테스트용 신호 생성기
    std::vector<float> generateConstantSignal(size_t len, float value = 1.0f) {
        return std::vector<float>(len, value);
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

    // eps <= 0
    bad_config = config_;
    bad_config.eps = 0.0f;
    EXPECT_THROW({OLAAccumulator ola(bad_config);}, std::invalid_argument);
}

// 윈도우 설정 테스트
TEST_F(OLAAccumulatorTest, WindowSetting) {
    OLAAccumulator ola(config_);

    // 유효한 윈도우 설정
    std::vector<float> window(config_.frame_size, 0.5f);
    EXPECT_NO_THROW(ola.set_window(window.data(), config_.frame_size));
    EXPECT_TRUE(ola.has_window());

    // null 포인터
    EXPECT_THROW(ola.set_window(nullptr, config_.frame_size), std::invalid_argument);

    // 잘못된 크기
    EXPECT_THROW(ola.set_window(window.data(), config_.frame_size + 1), std::invalid_argument);
}

// 단일 채널 SoA 테스트
TEST_F(OLAAccumulatorTest, SingleChannelSoA) {
    OLAAccumulator ola(config_);

    // 윈도우 설정
    std::vector<float> window(config_.frame_size, 1.0f); // 직사각 윈도우
    ola.set_window(window.data(), config_.frame_size);

    // 입력 프레임 준비 (SoA)
    std::vector<float> frame(config_.frame_size, 0.5f);
    const float* ch_frames[1] = {frame.data()};

    // 출력 버퍼 준비 (SoA)
    std::vector<float> output(config_.frame_size);
    float* ch_out[1] = {output.data()};

    // 프레임 추가
    ola.add_frame_SoA(ch_frames, window.data(), 0, 0, config_.frame_size, 1.0f);

    // 출력
    size_t produced = ola.produce(ch_out, config_.frame_size);

    // 기본 검증
    EXPECT_GT(produced, 0);
    EXPECT_EQ(ola.produced_samples(), config_.frame_size);
    EXPECT_EQ(ola.read_pos(), produced);
}

// 다채널 SoA 테스트
TEST_F(OLAAccumulatorTest, MultiChannelSoA) {
    config_.channels = 2;
    OLAAccumulator ola(config_);

    // 윈도우 설정
    std::vector<float> window(config_.frame_size, 1.0f);
    ola.set_window(window.data(), config_.frame_size);

    // 입력 프레임 준비 (SoA)
    std::vector<float> ch0_frame(config_.frame_size, 0.5f);
    std::vector<float> ch1_frame(config_.frame_size, 0.3f);
    const float* ch_frames[2] = {ch0_frame.data(), ch1_frame.data()};

    // 출력 버퍼 준비 (SoA)
    std::vector<float> ch0_out(config_.frame_size);
    std::vector<float> ch1_out(config_.frame_size);
    float* ch_out[2] = {ch0_out.data(), ch1_out.data()};

    // 프레임 추가
    ola.add_frame_SoA(ch_frames, window.data(), 0, 0, config_.frame_size, 1.0f);

    // 출력
    size_t produced = ola.produce(ch_out, config_.frame_size);

    // 기본 검증
    EXPECT_GT(produced, 0);
    EXPECT_EQ(ola.produced_samples(), config_.frame_size);
}

// 경계 케이스 테스트: 시작 오프셋
TEST_F(OLAAccumulatorTest, StartOffset) {
    OLAAccumulator ola(config_);

    std::vector<float> window(config_.frame_size, 1.0f);
    ola.set_window(window.data(), config_.frame_size);

    // 입력 프레임 준비
    std::vector<float> frame(config_.frame_size, 1.0f);
    const float* ch_frames[1] = {frame.data()};

    // 출력 버퍼 준비
    std::vector<float> output(config_.frame_size);
    float* ch_out[1] = {output.data()};

    // 시작 오프셋 적용
    size_t start_off = 32;
    ola.add_frame_SoA(ch_frames, window.data(), 0, start_off, config_.frame_size - start_off, 1.0f);

    // 출력
    size_t produced = ola.produce(ch_out, config_.frame_size);

    // 검증
    EXPECT_GT(produced, 0);
    EXPECT_EQ(ola.produced_samples(), config_.frame_size - start_off);
}

// 경계 케이스 테스트: 빈 요청
TEST_F(OLAAccumulatorTest, EmptyRequests) {
    OLAAccumulator ola(config_);

    // 빈 프레임 추가
    const float* ch_frames[1] = {nullptr};
    ola.add_frame_SoA(ch_frames, nullptr, 0, 0, 0, 1.0f);

    // 빈 출력 요청
    float* ch_out[1] = {nullptr};
    size_t produced = ola.produce(ch_out, 0);

    EXPECT_EQ(produced, 0);
}

// 에러 조건 테스트
TEST_F(OLAAccumulatorTest, ErrorConditions) {
    OLAAccumulator ola(config_);

    // null 포인터 채널 프레임
    const float* ch_frames[1] = {nullptr};
    EXPECT_THROW(ola.add_frame_SoA(ch_frames, nullptr, 0, 0, 1, 1.0f), std::invalid_argument);

    // null 포인터 출력 채널
    float* ch_out[1] = {nullptr};
    EXPECT_THROW(ola.produce(ch_out, 1), std::invalid_argument);
}

// 리셋 테스트
TEST_F(OLAAccumulatorTest, Reset) {
    OLAAccumulator ola(config_);

    // 윈도우 설정
    std::vector<float> window(config_.frame_size, 1.0f);
    ola.set_window(window.data(), config_.frame_size);

    // 데이터 추가
    std::vector<float> frame(config_.frame_size, 0.5f);
    const float* ch_frames[1] = {frame.data()};
    ola.add_frame_SoA(ch_frames, window.data(), 0, 0, config_.frame_size, 1.0f);

    // 출력하여 상태 변경
    std::vector<float> output(config_.frame_size);
    float* ch_out[1] = {output.data()};
    ola.produce(ch_out, config_.frame_size);

    // 리셋
    ola.reset();

    EXPECT_EQ(ola.produced_samples(), 0);
    EXPECT_EQ(ola.read_pos(), 0);
    EXPECT_EQ(ola.meter_peak(), 0.0f);
    EXPECT_FALSE(ola.has_window());
}