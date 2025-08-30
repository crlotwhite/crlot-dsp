#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>
#include "dsp/fft/api/fft_api.h"

namespace {

using namespace dsp::fft;

// FFT 테스트 픽스처
class FftTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 기본 FFT 설정
    desc.domain = FftDomain::Real;
    desc.nfft = 512;  // 기본값, 테스트에서 변경
    desc.in_place = false;
    desc.batch = 1;
    desc.stride_in = 1;
    desc.stride_out = 1;
  }

  FftPlanDesc desc;
};

// 정현파 생성 헬퍼 함수
std::vector<float> GenerateSineWave(int nfft, float frequency, float amplitude = 1.0f, float phase = 0.0f) {
  std::vector<float> signal(nfft);
  for (int i = 0; i < nfft; ++i) {
    float t = static_cast<float>(i) / nfft;
    signal[i] = amplitude * std::sin(2.0f * M_PI * frequency * t + phase);
  }
  return signal;
}

// 복소수 벡터의 크기 계산
float Magnitude(const std::complex<float>& c) {
  return std::abs(c);
}

// 복소수 벡터의 위상 계산
float Phase(const std::complex<float>& c) {
  return std::arg(c);
}

// RMS 오차 계산
float CalculateRmsError(const std::vector<float>& original, const std::vector<float>& reconstructed) {
  float sum_squared_error = 0.0f;
  for (size_t i = 0; i < original.size(); ++i) {
    float error = original[i] - reconstructed[i];
    sum_squared_error += error * error;
  }
  return std::sqrt(sum_squared_error / original.size());
}

TEST_F(FftTest, BasicForwardInverse512) {
  desc.nfft = 512;
  auto plan = MakeFftPlan(desc);

  // 정현파 입력 생성 (주파수 = 10/NFFT)
  auto input = GenerateSineWave(512, 10.0f / 512.0f, 1.0f);
  std::vector<std::complex<float>> fft_output(512 / 2 + 1);
  std::vector<float> ifft_output(512);

  // 순방향 FFT
  plan->forward(input.data(), fft_output.data());

  // 역방향 FFT
  plan->inverse(fft_output.data(), ifft_output.data());

  // 재구성 오차 검증 (매우 작은 값이어야 함)
  float rms_error = CalculateRmsError(input, ifft_output);
  EXPECT_LT(rms_error, 1e-6f);
}

TEST_F(FftTest, BasicForwardInverse1024) {
  desc.nfft = 1024;
  auto plan = MakeFftPlan(desc);

  auto input = GenerateSineWave(1024, 20.0f / 1024.0f, 1.0f);
  std::vector<std::complex<float>> fft_output(1024 / 2 + 1);
  std::vector<float> ifft_output(1024);

  plan->forward(input.data(), fft_output.data());
  plan->inverse(fft_output.data(), ifft_output.data());

  float rms_error = CalculateRmsError(input, ifft_output);
  EXPECT_LT(rms_error, 1e-6f);
}

TEST_F(FftTest, BasicForwardInverse2048) {
  desc.nfft = 2048;
  auto plan = MakeFftPlan(desc);

  auto input = GenerateSineWave(2048, 40.0f / 2048.0f, 1.0f);
  std::vector<std::complex<float>> fft_output(2048 / 2 + 1);
  std::vector<float> ifft_output(2048);

  plan->forward(input.data(), fft_output.data());
  plan->inverse(fft_output.data(), ifft_output.data());

  float rms_error = CalculateRmsError(input, ifft_output);
  EXPECT_LT(rms_error, 1e-6f);
}

TEST_F(FftTest, AmplitudeAndPhaseCorrectness512) {
  desc.nfft = 512;
  auto plan = MakeFftPlan(desc);

  // DC 성분 테스트 (진폭 1.0)
  std::vector<float> input(512, 1.0f);
  std::vector<std::complex<float>> fft_output(512 / 2 + 1);

  plan->forward(input.data(), fft_output.data());

  // DC 성분 (인덱스 0) 진폭 검증
  EXPECT_NEAR(Magnitude(fft_output[0]), 512.0f, 1e-3f);
  EXPECT_NEAR(Phase(fft_output[0]), 0.0f, 1e-3f);

  // 다른 성분들은 거의 0이어야 함
  for (size_t i = 1; i < fft_output.size(); ++i) {
    EXPECT_LT(Magnitude(fft_output[i]), 1e-3f);
  }
}

TEST_F(FftTest, AmplitudeAndPhaseCorrectness1024) {
  desc.nfft = 1024;
  auto plan = MakeFftPlan(desc);

  // 주파수 10번째 bin 테스트 (cos 함수 사용)
  std::vector<float> input(1024);
  for (int i = 0; i < 1024; ++i) {
    float t = static_cast<float>(i) / 1024.0f;
    input[i] = 2.0f * std::cos(2.0f * M_PI * 10.0f * t);
  }
  std::vector<std::complex<float>> fft_output(1024 / 2 + 1);

  plan->forward(input.data(), fft_output.data());

  // 10번째 bin 진폭 검증 (KissFFT는 정규화하지 않으므로 N/2로 나눔)
  float expected_magnitude = 2.0f * 1024.0f / 2.0f;  // 입력 진폭 * N / 2
  EXPECT_NEAR(Magnitude(fft_output[10]), expected_magnitude, 1e-3f);
  EXPECT_NEAR(Phase(fft_output[10]), 0.0f, 1e-3f);  // cos 함수의 위상

  // DC 성분과 Nyquist 성분 검증
  EXPECT_NEAR(Magnitude(fft_output[0]), 0.0f, 1e-3f);  // DC 성분 (cos 함수)
  EXPECT_NEAR(Magnitude(fft_output[512]), 0.0f, 1e-3f);  // Nyquist 성분
}

TEST_F(FftTest, AmplitudeAndPhaseCorrectness2048) {
  desc.nfft = 2048;
  auto plan = MakeFftPlan(desc);

  // 주파수 50번째 bin 테스트 (cos 함수 사용)
  std::vector<float> input(2048);
  for (int i = 0; i < 2048; ++i) {
    float t = static_cast<float>(i) / 2048.0f;
    input[i] = 3.0f * std::cos(2.0f * M_PI * 50.0f * t);
  }
  std::vector<std::complex<float>> fft_output(2048 / 2 + 1);

  plan->forward(input.data(), fft_output.data());

  // 50번째 bin 진폭 검증 (KissFFT는 정규화하지 않으므로 N/2로 나눔)
  float expected_magnitude = 3.0f * 2048.0f / 2.0f;  // 입력 진폭 * N / 2
  EXPECT_NEAR(Magnitude(fft_output[50]), expected_magnitude, 1e-3f);
  EXPECT_NEAR(Phase(fft_output[50]), 0.0f, 1e-3f);  // cos 함수의 위상

  // DC 성분과 Nyquist 성분 검증
  EXPECT_NEAR(Magnitude(fft_output[0]), 0.0f, 1e-3f);  // DC 성분 (cos 함수)
  EXPECT_NEAR(Magnitude(fft_output[1024]), 0.0f, 1e-3f);  // Nyquist 성분
}

TEST_F(FftTest, NaNAndDenormalHandling) {
  desc.nfft = 512;
  auto plan = MakeFftPlan(desc);

  std::vector<float> input(512, 0.0f);
  // NaN과 매우 작은 값 추가
  input[0] = std::numeric_limits<float>::quiet_NaN();
  input[1] = 1e-40f;  // denormal
  input[2] = std::numeric_limits<float>::infinity();

  std::vector<std::complex<float>> fft_output(512 / 2 + 1);
  std::vector<float> ifft_output(512);

  // FFT 실행 (크래시 없이 완료되어야 함)
  EXPECT_NO_THROW(plan->forward(input.data(), fft_output.data()));
  EXPECT_NO_THROW(plan->inverse(fft_output.data(), ifft_output.data()));

  // 출력에 NaN이나 inf가 없어야 함
  for (size_t i = 0; i < ifft_output.size(); ++i) {
    EXPECT_FALSE(std::isnan(ifft_output[i]));
    EXPECT_FALSE(std::isinf(ifft_output[i]));
  }
}

TEST_F(FftTest, InvalidConfiguration) {
  // 홀수 NFFT는 실패해야 함
  desc.nfft = 513;
  EXPECT_THROW(MakeFftPlan(desc), std::runtime_error);

  // Complex 도메인은 실패해야 함
  desc.nfft = 512;
  desc.domain = FftDomain::Complex;
  EXPECT_THROW(MakeFftPlan(desc), std::runtime_error);

  // Batch > 1은 실패해야 함
  desc.domain = FftDomain::Real;
  desc.batch = 2;
  EXPECT_THROW(MakeFftPlan(desc), std::runtime_error);
}

}  // namespace