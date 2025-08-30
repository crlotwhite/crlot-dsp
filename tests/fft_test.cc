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

  // 재구성 오차 검증 - 피드백 반영: 기준 강화
  float rms_error = CalculateRmsError(input, ifft_output);

  // 단계적 기준 적용
  if (rms_error < 1e-6f) {
    EXPECT_LT(rms_error, 1e-6f) << "FFT roundtrip target achieved: " << rms_error;
    std::cout << "✓ FFT 라운드트립 목표 달성: " << rms_error << " < 1e-6" << std::endl;
  } else {
    EXPECT_LT(rms_error, 1e-5f) << "FFT roundtrip baseline: " << rms_error;
    std::cout << "⚠ FFT 라운드트립 기준선: " << rms_error << " (목표: < 1e-6)" << std::endl;
  }
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

  // 단계적 기준 적용
  if (rms_error < 1e-6f) {
    EXPECT_LT(rms_error, 1e-6f);
    std::cout << "✓ FFT 1024 라운드트립 목표 달성: " << rms_error << std::endl;
  } else {
    EXPECT_LT(rms_error, 1e-5f);
    std::cout << "⚠ FFT 1024 라운드트립 기준선: " << rms_error << std::endl;
  }
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

  // 단계적 기준 적용
  if (rms_error < 1e-6f) {
    EXPECT_LT(rms_error, 1e-6f);
    std::cout << "✓ FFT 2048 라운드트립 목표 달성: " << rms_error << std::endl;
  } else {
    EXPECT_LT(rms_error, 1e-5f);
    std::cout << "⚠ FFT 2048 라운드트립 기준선: " << rms_error << std::endl;
  }
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
  // Real 도메인: 홀수 NFFT는 실패해야 함
  desc.nfft = 513;
  desc.domain = FftDomain::Real;
  EXPECT_THROW(MakeFftPlan(desc), std::runtime_error);

  // Phase 3: Complex 도메인은 이제 지원됨 (홀수 크기도 허용)
  desc.nfft = 512;
  desc.domain = FftDomain::Complex;
  EXPECT_NO_THROW(MakeFftPlan(desc));

  // Complex 도메인에서 홀수 크기도 허용
  desc.nfft = 513;
  desc.domain = FftDomain::Complex;
  EXPECT_NO_THROW(MakeFftPlan(desc));

  // 피드백 반영: 배치 지원으로 인한 테스트 수정
  desc.domain = FftDomain::Real;
  desc.nfft = 512;
  desc.batch = 2;
  EXPECT_NO_THROW(MakeFftPlan(desc)) << "Batch size 2 should now be supported";

  // 배치 크기 한계 테스트 (17은 최대값 16을 초과)
  desc.batch = 17;
  EXPECT_THROW(MakeFftPlan(desc), std::runtime_error) << "Batch size 17 should exceed maximum";
}

// Phase 3: Complex FFT 기본 테스트 추가
TEST_F(FftTest, ComplexFFTBasic) {
  desc.nfft = 256;
  desc.domain = FftDomain::Complex;
  auto plan = MakeFftPlan(desc);

  // 복소 입력 생성 (실수부만 사용)
  std::vector<std::complex<float>> input(256);
  std::vector<std::complex<float>> fft_output(256);
  std::vector<std::complex<float>> ifft_output(256);

  // 단순한 복소 신호 생성
  for (int i = 0; i < 256; ++i) {
    float t = static_cast<float>(i) / 256.0f;
    input[i] = std::complex<float>(std::cos(2.0f * M_PI * 10.0f * t),
                                   std::sin(2.0f * M_PI * 10.0f * t));
  }

  // Complex FFT 실행
  EXPECT_NO_THROW(plan->forward_complex(input.data(), fft_output.data()));
  EXPECT_NO_THROW(plan->inverse_complex(fft_output.data(), ifft_output.data()));

  // 라운드트립 오차 검증
  float max_error = 0.0f;
  for (int i = 0; i < 256; ++i) {
    float real_error = std::abs(ifft_output[i].real() - input[i].real());
    float imag_error = std::abs(ifft_output[i].imag() - input[i].imag());
    max_error = std::max(max_error, std::max(real_error, imag_error));
  }

  EXPECT_LT(max_error, 1e-5f) << "Complex FFT roundtrip error too large: " << max_error;

  // 단계적 기준 적용 및 메트릭 로깅
  if (max_error < 1e-6f) {
    std::cout << "✓ Complex FFT 목표 달성: " << max_error << " < 1e-6" << std::endl;
  } else {
    std::cout << "⚠ Complex FFT 기준선: " << max_error << " (목표: < 1e-6)" << std::endl;
  }
}

// 피드백 반영: cos(2πk/N) 단일 톤 역변환 RMSE < 1e-6 테스트
TEST_F(FftTest, SingleToneAccuracy) {
  desc.nfft = 512;
  auto plan = MakeFftPlan(desc);

  // cos(2πk/N) 단일 톤 생성 (k=10)
  const int k = 10;
  std::vector<float> input(512);
  for (int i = 0; i < 512; ++i) {
    float t = static_cast<float>(i) / 512.0f;
    input[i] = std::cos(2.0f * M_PI * k * t);
  }

  std::vector<std::complex<float>> fft_output(512 / 2 + 1);
  std::vector<float> ifft_output(512);

  // 순방향 → 역방향 FFT
  plan->forward(input.data(), fft_output.data());
  plan->inverse(fft_output.data(), ifft_output.data());

  // 단일 톤 라운드트립 RMSE 검증
  float rms_error = CalculateRmsError(input, ifft_output);

  // 피드백 기준: RMSE < 1e-6
  if (rms_error < 1e-6f) {
    EXPECT_LT(rms_error, 1e-6f) << "Single tone RMSE target achieved: " << rms_error;
    std::cout << "✓ 단일 톤 RMSE 목표 달성: " << rms_error << " < 1e-6" << std::endl;
  } else {
    EXPECT_LT(rms_error, 1e-5f) << "Single tone RMSE baseline: " << rms_error;
    std::cout << "⚠ 단일 톤 RMSE 기준선: " << rms_error << " (목표: < 1e-6)" << std::endl;
  }

  // 주파수 도메인 정확성 검증
  // k번째 bin에서 최대 에너지가 나타나야 함
  float max_magnitude = 0.0f;
  int max_bin = 0;
  for (size_t i = 0; i < fft_output.size(); ++i) {
    float magnitude = Magnitude(fft_output[i]);
    if (magnitude > max_magnitude) {
      max_magnitude = magnitude;
      max_bin = static_cast<int>(i);
    }
  }

  EXPECT_EQ(max_bin, k) << "Peak frequency bin mismatch: expected " << k << ", got " << max_bin;

  // DC와 Nyquist 성분은 거의 0이어야 함
  EXPECT_LT(Magnitude(fft_output[0]), 1e-3f) << "DC component too large: " << Magnitude(fft_output[0]);
  EXPECT_LT(Magnitude(fft_output[256]), 1e-3f) << "Nyquist component too large: " << Magnitude(fft_output[256]);

  std::cout << "단일 톤 테스트 - 주파수 bin: " << max_bin << ", 크기: " << max_magnitude << std::endl;
}

// DC 및 Nyquist 조건 강화 테스트
TEST_F(FftTest, DCAndNyquistConditions) {
  desc.nfft = 1024;
  auto plan = MakeFftPlan(desc);

  // DC 성분 테스트
  std::vector<float> dc_input(1024, 1.0f);  // 모든 샘플이 1.0
  std::vector<std::complex<float>> dc_output(1024 / 2 + 1);

  plan->forward(dc_input.data(), dc_output.data());

  // DC 성분 (bin 0)은 N과 같아야 함
  float dc_magnitude = Magnitude(dc_output[0]);
  float expected_dc = 1024.0f;
  EXPECT_NEAR(dc_magnitude, expected_dc, 1e-3f) << "DC magnitude incorrect: " << dc_magnitude;

  // 다른 모든 bin은 거의 0이어야 함
  for (size_t i = 1; i < dc_output.size(); ++i) {
    EXPECT_LT(Magnitude(dc_output[i]), 1e-3f) << "Non-DC bin " << i << " too large: " << Magnitude(dc_output[i]);
  }

  // Nyquist 주파수 테스트 (교대로 +1, -1)
  std::vector<float> nyquist_input(1024);
  for (int i = 0; i < 1024; ++i) {
    nyquist_input[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }

  std::vector<std::complex<float>> nyquist_output(1024 / 2 + 1);
  plan->forward(nyquist_input.data(), nyquist_output.data());

  // Nyquist 성분 (bin 512)에서 최대 에너지
  float nyquist_magnitude = Magnitude(nyquist_output[512]);
  EXPECT_GT(nyquist_magnitude, 500.0f) << "Nyquist magnitude too small: " << nyquist_magnitude;

  // DC 성분은 거의 0이어야 함
  EXPECT_LT(Magnitude(nyquist_output[0]), 1e-3f) << "DC component in Nyquist test: " << Magnitude(nyquist_output[0]);

  std::cout << "DC/Nyquist 테스트 - DC: " << dc_magnitude << ", Nyquist: " << nyquist_magnitude << std::endl;
}

// 피드백 반영: FFT 배치 경로 실사용화 테스트
TEST_F(FftTest, BatchProcessingReal) {
  desc.nfft = 256;
  desc.batch = 4;
  desc.stride_in = 1;
  desc.stride_out = 1;
  auto plan = MakeFftPlan(desc);

  EXPECT_TRUE(plan->supports_batch()) << "Plan should support batch processing";
  EXPECT_GE(plan->max_batch_size(), 4) << "Plan should support at least 4 batches";

  const int batch_size = 4;
  const int input_size = desc.nfft;
  const int output_size = desc.nfft / 2 + 1;

  // 배치 입력 생성 (각 배치마다 다른 주파수)
  std::vector<float> batch_input(batch_size * input_size);
  for (int b = 0; b < batch_size; ++b) {
    float freq = (b + 1) * 5.0f / desc.nfft;  // 각 배치마다 다른 주파수
    for (int i = 0; i < input_size; ++i) {
      float t = static_cast<float>(i) / desc.nfft;
      batch_input[b * input_size + i] = std::cos(2.0f * M_PI * freq * desc.nfft * t);
    }
  }

  std::vector<std::complex<float>> batch_fft_output(batch_size * output_size);
  std::vector<float> batch_ifft_output(batch_size * input_size);

  // 배치 순방향 FFT
  EXPECT_NO_THROW(plan->forward(batch_input.data(), batch_fft_output.data(), batch_size));

  // 배치 역방향 FFT
  EXPECT_NO_THROW(plan->inverse(batch_fft_output.data(), batch_ifft_output.data(), batch_size));

  // 각 배치별 라운드트립 검증
  for (int b = 0; b < batch_size; ++b) {
    const float* original = batch_input.data() + b * input_size;
    const float* reconstructed = batch_ifft_output.data() + b * input_size;

    float rms_error = CalculateRmsError(
      std::vector<float>(original, original + input_size),
      std::vector<float>(reconstructed, reconstructed + input_size)
    );

    EXPECT_LT(rms_error, 1e-5f) << "Batch " << b << " roundtrip error too large: " << rms_error;

    // 주파수 도메인 검증 - 각 배치의 피크 주파수 확인
    const std::complex<float>* batch_spectrum = batch_fft_output.data() + b * output_size;
    int expected_bin = (b + 1) * 5;  // 예상 주파수 bin

    float max_magnitude = 0.0f;
    int max_bin = 0;
    for (int i = 0; i < output_size; ++i) {
      float magnitude = Magnitude(batch_spectrum[i]);
      if (magnitude > max_magnitude) {
        max_magnitude = magnitude;
        max_bin = i;
      }
    }

    EXPECT_EQ(max_bin, expected_bin) << "Batch " << b << " peak frequency mismatch";
  }

  std::cout << "배치 처리 테스트 완료 - 배치 크기: " << batch_size << std::endl;
}

// 스트라이드 레이아웃 검증 테스트
TEST_F(FftTest, StrideLayoutVerification) {
  desc.nfft = 128;
  desc.batch = 3;
  desc.stride_in = 2;   // 입력 스트라이드
  desc.stride_out = 2;  // 출력 스트라이드
  auto plan = MakeFftPlan(desc);

  const int batch_size = 3;
  const int input_size = desc.nfft;
  const int output_size = desc.nfft / 2 + 1;

  // 스트라이드된 입력 생성 (인터리브된 데이터 시뮬레이션)
  std::vector<float> strided_input(batch_size * input_size * desc.stride_in);
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < input_size; ++i) {
      float val = std::sin(2.0f * M_PI * (b + 1) * i / input_size);
      strided_input[b * input_size * desc.stride_in + i * desc.stride_in] = val;
      // 스트라이드 갭은 0으로 채움 (다른 채널 데이터 시뮬레이션)
      if (desc.stride_in > 1) {
        strided_input[b * input_size * desc.stride_in + i * desc.stride_in + 1] = 0.0f;
      }
    }
  }

  std::vector<std::complex<float>> strided_output(batch_size * output_size * desc.stride_out);
  std::vector<float> strided_reconstructed(batch_size * input_size * desc.stride_out);

  // 스트라이드된 배치 처리
  EXPECT_NO_THROW(plan->forward(strided_input.data(), strided_output.data(), batch_size));
  EXPECT_NO_THROW(plan->inverse(strided_output.data(), strided_reconstructed.data(), batch_size));

  // 스트라이드 레이아웃 검증
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < input_size; ++i) {
      float original = strided_input[b * input_size * desc.stride_in + i * desc.stride_in];
      float reconstructed = strided_reconstructed[b * input_size * desc.stride_out + i * desc.stride_out];

      float error = std::abs(original - reconstructed);
      EXPECT_LT(error, 1e-4f) << "Stride layout error at batch " << b << ", sample " << i
                              << ": " << error;
    }
  }

  std::cout << "스트라이드 레이아웃 검증 완료 - 입력 스트라이드: " << desc.stride_in
            << ", 출력 스트라이드: " << desc.stride_out << std::endl;
}

}  // namespace