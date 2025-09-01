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

// 단일 채널 AoS 테스트
TEST_F(OLAAccumulatorTest, SingleChannelAoS) {
    OLAAccumulator ola(config_);

    // 윈도우 설정
    std::vector<float> window(config_.frame_size, 1.0f);
    ola.set_window(window.data(), config_.frame_size);

    // AoS 입력 생성 (인터리브)
    std::vector<float> aos_input(config_.frame_size * config_.channels, 0.5f);

    // 출력 버퍼 준비
    std::vector<float> output(config_.frame_size);
    float* ch_out[1] = {output.data()};

    // AoS 프레임 추가
    ola.push_frame_AoS(aos_input.data(), window.data(), 0, 0, config_.frame_size, 1.0f);

    // 출력
    size_t produced = ola.produce(ch_out, config_.frame_size);

    // 검증
    EXPECT_GT(produced, 0);
    EXPECT_EQ(ola.produced_samples(), config_.frame_size);
}

// 다채널 AoS 테스트
TEST_F(OLAAccumulatorTest, MultiChannelAoS) {
    config_.channels = 2;
    OLAAccumulator ola(config_);

    // 윈도우 설정
    std::vector<float> window(config_.frame_size, 1.0f);
    ola.set_window(window.data(), config_.frame_size);

    // AoS 입력 생성 (인터리브: ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...)
    std::vector<float> aos_input(config_.frame_size * config_.channels);
    for (size_t i = 0; i < config_.frame_size; ++i) {
        aos_input[i * 2 + 0] = 0.5f; // ch0
        aos_input[i * 2 + 1] = 0.3f; // ch1
    }

    // 출력 버퍼 준비
    std::vector<float> ch0_out(config_.frame_size);
    std::vector<float> ch1_out(config_.frame_size);
    float* ch_out[2] = {ch0_out.data(), ch1_out.data()};

    // AoS 프레임 추가
    ola.push_frame_AoS(aos_input.data(), window.data(), 0, 0, config_.frame_size, 1.0f);

    // 출력
    size_t produced = ola.produce(ch_out, config_.frame_size);

    // 검증
    EXPECT_GT(produced, 0);
    EXPECT_EQ(ola.produced_samples(), config_.frame_size);
}

// AoS vs SoA 일치성 테스트
TEST_F(OLAAccumulatorTest, AoSvsSoAConsistency) {
    config_.channels = 2;
    OLAAccumulator ola_aos(config_);
    OLAAccumulator ola_soa(config_);

    // 윈도우 설정
    std::vector<float> window(config_.frame_size, 1.0f);
    ola_aos.set_window(window.data(), config_.frame_size);
    ola_soa.set_window(window.data(), config_.frame_size);

    // 입력 데이터 생성
    std::vector<float> ch0_frame(config_.frame_size, 0.5f);
    std::vector<float> ch1_frame(config_.frame_size, 0.3f);
    const float* ch_frames[2] = {ch0_frame.data(), ch1_frame.data()};

    // AoS 입력 생성
    std::vector<float> aos_input(config_.frame_size * config_.channels);
    for (size_t i = 0; i < config_.frame_size; ++i) {
        aos_input[i * 2 + 0] = ch0_frame[i];
        aos_input[i * 2 + 1] = ch1_frame[i];
    }

    // 출력 버퍼 준비
    std::vector<float> aos_ch0_out(config_.frame_size);
    std::vector<float> aos_ch1_out(config_.frame_size);
    float* aos_ch_out[2] = {aos_ch0_out.data(), aos_ch1_out.data()};

    std::vector<float> soa_ch0_out(config_.frame_size);
    std::vector<float> soa_ch1_out(config_.frame_size);
    float* soa_ch_out[2] = {soa_ch0_out.data(), soa_ch1_out.data()};

    // AoS와 SoA로 동일한 데이터 추가
    ola_aos.push_frame_AoS(aos_input.data(), window.data(), 0, 0, config_.frame_size, 1.0f);
    ola_soa.add_frame_SoA(ch_frames, window.data(), 0, 0, config_.frame_size, 1.0f);

    // 출력
    ola_aos.produce(aos_ch_out, config_.frame_size);
    ola_soa.produce(soa_ch_out, config_.frame_size);

    // 샘플-레벨 일치 검증
    for (size_t i = 0; i < config_.frame_size; ++i) {
        EXPECT_FLOAT_EQ(aos_ch0_out[i], soa_ch0_out[i]);
        EXPECT_FLOAT_EQ(aos_ch1_out[i], soa_ch1_out[i]);
    }
}

// Shadow Ring 옵션 테스트
TEST_F(OLAAccumulatorTest, ShadowRingOption) {
    // Shadow Ring 비활성화 설정
    OLAConfig config_normal = config_;
    config_normal.shadow_ring = false;

    // Shadow Ring 활성화 설정
    OLAConfig config_shadow = config_;
    config_shadow.shadow_ring = true;

    OLAAccumulator ola_normal(config_normal);
    OLAAccumulator ola_shadow(config_shadow);

    // 윈도우 설정
    std::vector<float> window(config_.frame_size, 1.0f);
    ola_normal.set_window(window.data(), config_.frame_size);
    ola_shadow.set_window(window.data(), config_.frame_size);

    // 입력 프레임 준비
    std::vector<float> frame(config_.frame_size, 0.5f);
    const float* ch_frames[1] = {frame.data()};

    // 출력 버퍼 준비
    std::vector<float> output_normal(config_.frame_size);
    std::vector<float> output_shadow(config_.frame_size);
    float* ch_out_normal[1] = {output_normal.data()};
    float* ch_out_shadow[1] = {output_shadow.data()};

    // 동일한 프레임 추가
    ola_normal.add_frame_SoA(ch_frames, window.data(), 0, 0, config_.frame_size, 1.0f);
    ola_shadow.add_frame_SoA(ch_frames, window.data(), 0, 0, config_.frame_size, 1.0f);

    // 출력
    size_t produced_normal = ola_normal.produce(ch_out_normal, config_.frame_size);
    size_t produced_shadow = ola_shadow.produce(ch_out_shadow, config_.frame_size);

    // 기본 검증: 동일한 출력 결과를 생성해야 함
    EXPECT_EQ(produced_normal, produced_shadow);
    EXPECT_EQ(ola_normal.produced_samples(), ola_shadow.produced_samples());

    // 샘플-레벨 일치 검증
    for (size_t i = 0; i < config_.frame_size; ++i) {
        EXPECT_FLOAT_EQ(output_normal[i], output_shadow[i]);
    }
}

// Shadow Ring 메모리 할당 및 기본 속성 테스트
TEST_F(OLAAccumulatorTest, ShadowRingMemoryAllocation) {
    // 큰 프레임 크기로 테스트
    config_.frame_size = 1024;
    config_.hop_size = 256;

    OLAConfig config_normal = config_;
    config_normal.shadow_ring = false;

    OLAConfig config_shadow = config_;
    config_shadow.shadow_ring = true;

    OLAAccumulator ola_normal(config_normal);
    OLAAccumulator ola_shadow(config_shadow);

    // RingBuffer 크기 검증
    size_t ring_size_normal = ola_normal.ring_size();
    size_t ring_size_shadow = ola_shadow.ring_size();

    // 기본 링 크기는 동일
    EXPECT_EQ(ring_size_normal, ring_size_shadow);
}

// Shadow Ring contiguous_read_ptr 테스트
TEST_F(OLAAccumulatorTest, ShadowRingContiguousReadPtr) {
    config_.channels = 1;
    config_.frame_size = 256;
    config_.hop_size = 64;
    config_.shadow_ring = true;

    OLAAccumulator ola(config_);

    // 윈도우 설정
    std::vector<float> window(config_.frame_size, 1.0f);
    ola.set_window(window.data(), config_.frame_size);

    // 입력 프레임 준비
    std::vector<float> frame(config_.frame_size, 0.5f);
    const float* ch_frames[1] = {frame.data()};

    // 프레임 추가
    ola.add_frame_SoA(ch_frames, window.data(), 0, 0, config_.frame_size, 1.0f);

    // 기본적인 동작 검증
    // Shadow Ring 모드에서는 내부적으로 contiguous_read_ptr가 사용됨
    // 여기서는 OLA Accumulator의 정상 동작만 검증
    EXPECT_TRUE(ola.has_window());
    EXPECT_EQ(ola.config().shadow_ring, true);
    EXPECT_EQ(ola.produced_samples(), config_.frame_size);
}

// Shadow Ring 직접 테스트 (RingBuffer 단위)
TEST(RingBufferShadowTest, ShadowRingBasicProperties) {
    const size_t capacity = 256;

    // 일반 RingBuffer
    dsp::ring::RingBuffer<float> rb_normal(capacity, false);
    EXPECT_EQ(rb_normal.capacity(), capacity);
    EXPECT_EQ(rb_normal.physical_capacity(), capacity);
    EXPECT_FALSE(rb_normal.has_shadow());

    // Shadow RingBuffer
    dsp::ring::RingBuffer<float> rb_shadow(capacity, true);
    EXPECT_EQ(rb_shadow.capacity(), capacity);
    EXPECT_EQ(rb_shadow.physical_capacity(), capacity * 2);
    EXPECT_TRUE(rb_shadow.has_shadow());
}

// Shadow Ring wrap 없는 쓰기 테스트
TEST(RingBufferShadowTest, ShadowRingWriteWithoutWrap) {
    const size_t capacity = 256;
    dsp::ring::RingBuffer<float> rb(capacity, true);

    // 초기 상태 확인
    EXPECT_EQ(rb.write_pos(), 0);

    // capacity보다 작은 데이터 쓰기 (wrap 없음)
    std::vector<float> data(capacity / 2, 1.0f);
    size_t written = rb.write(data.data(), data.size());

    EXPECT_EQ(written, data.size());
    EXPECT_EQ(rb.write_pos(), data.size());

    // 미러 영역은 변경되지 않아야 함
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(rb.data()[capacity + i], 0.0f);  // 초기값 0
    }
}

// Shadow Ring wrap 있는 쓰기 및 미러 복사 테스트
TEST(RingBufferShadowTest, ShadowRingWriteWithWrap) {
    const size_t capacity = 256;
    dsp::ring::RingBuffer<float> rb(capacity, true);

    // capacity보다 큰 데이터 쓰기 (wrap 발생)
    std::vector<float> data(capacity + capacity / 2, 2.0f);
    size_t written = rb.write(data.data(), data.size());

    EXPECT_EQ(written, data.size());
    EXPECT_EQ(rb.write_pos(), capacity / 2);  // 헤드에 쓴 만큼

    // 헤드 영역 데이터 확인
    for (size_t i = 0; i < capacity / 2; ++i) {
        EXPECT_EQ(rb.data()[i], 2.0f);
    }

    // 미러 영역에 복사되었는지 확인
    for (size_t i = 0; i < capacity / 2; ++i) {
        EXPECT_EQ(rb.data()[capacity + i], 2.0f);
    }
}

// Shadow Ring 정확히 capacity만큼 쓰기 테스트
TEST(RingBufferShadowTest, ShadowRingWriteExactCapacity) {
    const size_t capacity = 256;
    dsp::ring::RingBuffer<float> rb(capacity, true);

    // 정확히 capacity만큼 쓰기
    std::vector<float> data(capacity, 3.0f);
    size_t written = rb.write(data.data(), data.size());

    EXPECT_EQ(written, data.size());
    EXPECT_EQ(rb.write_pos(), 0);  // wrap 발생하지 않음

    // 미러 영역은 변경되지 않아야 함
    for (size_t i = 0; i < capacity; ++i) {
        EXPECT_EQ(rb.data()[capacity + i], 0.0f);  // 초기값 0
    }
}

// Shadow Ring 연속 뷰 제공 테스트
TEST(RingBufferShadowTest, ShadowRingContiguousView) {
    const size_t capacity = 256;
    dsp::ring::RingBuffer<float> rb(capacity, true);

    // 일부 데이터 쓰기
    std::vector<float> data(capacity / 4, 4.0f);
    rb.write(data.data(), data.size());

    // contiguous_read_ptr 테스트
    const float* ptr = rb.contiguous_read_ptr(0);
    EXPECT_EQ(ptr, rb.data());

    // Shadow 모드에서는 wrap 경계를 넘어도 연속 메모리
    const float* ptr2 = rb.contiguous_read_ptr(capacity / 2);
    EXPECT_EQ(ptr2, rb.data() + capacity / 2);
}

// Shadow Ring split 유틸 호환성 테스트
TEST(RingBufferShadowTest, ShadowRingSplitCompatibility) {
    const size_t capacity = 256;
    dsp::ring::RingBuffer<float> rb(capacity, true);

    // 데이터 쓰기
    std::vector<float> data(capacity / 2, 5.0f);
    rb.write(data.data(), data.size());

    // split 유틸 테스트
    auto [span1, span2] = rb.split(0, capacity / 4);

    EXPECT_FALSE(span1.empty());
    EXPECT_TRUE(span2.empty());  // wrap 없으므로 두 번째 스팬은 비어있음

    // 데이터 일치 확인
    for (size_t i = 0; i < span1.size(); ++i) {
        EXPECT_EQ(span1[i], 5.0f);
    }
}