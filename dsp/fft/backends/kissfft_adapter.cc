#include "dsp/fft/api/fft_api.h"
#include <kiss_fft.h>
#include <kiss_fftr.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace dsp::fft {

// KissFFT 기반 FFT 플랜 구현체
class KissFftPlan : public IFftPlan {
public:
  explicit KissFftPlan(const FftPlanDesc& desc) : desc_(desc) {
    // Phase 3: Complex 도메인 지원 확장
    if (desc.domain != FftDomain::Real && desc.domain != FftDomain::Complex) {
      throw std::runtime_error("Unsupported FFT domain");
    }

    // 피드백 반영: 배치 지원 확장
    if (desc.batch < 1 || desc.batch > 16) {
      throw std::runtime_error("Batch size must be between 1 and 16");
    }

    // 스트라이드 검증 (배치 처리용)
    if (desc.stride_in < 1 || desc.stride_out < 1) {
      throw std::runtime_error("Stride must be at least 1");
    }

    // in_place는 현재 미지원
    if (desc.in_place) {
      throw std::runtime_error("In-place FFT is not yet supported");
    }

    if (desc.domain == FftDomain::Real) {
      // Real FFT: 크기는 짝수여야 함
      if (desc.nfft % 2 != 0) {
        throw std::runtime_error("FFT size must be even for real FFT");
      }

      // KissFFT Real 설정 할당
      forward_real_cfg_ = kiss_fftr_alloc(desc.nfft, 0, nullptr, nullptr);
      inverse_real_cfg_ = kiss_fftr_alloc(desc.nfft, 1, nullptr, nullptr);

      if (!forward_real_cfg_ || !inverse_real_cfg_) {
        throw std::runtime_error("Failed to allocate KissFFT real configuration");
      }

      // Real FFT 버퍼 할당 (출력 크기: nfft/2 + 1)
      real_output_buffer_.resize(desc.nfft / 2 + 1);
    } else {
      // Complex FFT
      // KissFFT Complex 설정 할당
      forward_complex_cfg_ = kiss_fft_alloc(desc.nfft, 0, nullptr, nullptr);
      inverse_complex_cfg_ = kiss_fft_alloc(desc.nfft, 1, nullptr, nullptr);

      if (!forward_complex_cfg_ || !inverse_complex_cfg_) {
        throw std::runtime_error("Failed to allocate KissFFT complex configuration");
      }

      // Complex FFT 버퍼 할당 (출력 크기: nfft)
      complex_buffer_.resize(desc.nfft);
    }
  }

  ~KissFftPlan() override {
    // Real FFT 정리
    if (forward_real_cfg_) {
      kiss_fftr_free(forward_real_cfg_);
    }
    if (inverse_real_cfg_) {
      kiss_fftr_free(inverse_real_cfg_);
    }

    // Complex FFT 정리
    if (forward_complex_cfg_) {
      kiss_fft_free(forward_complex_cfg_);
    }
    if (inverse_complex_cfg_) {
      kiss_fft_free(inverse_complex_cfg_);
    }
  }

  void forward(const float* in, std::complex<float>* out, int batch = 1) override {
    if (desc_.domain != FftDomain::Real) {
      throw std::runtime_error("Real FFT not supported for Complex domain plan");
    }

    // 피드백 반영: 배치 처리 지원
    if (batch < 1 || batch > desc_.batch) {
      throw std::runtime_error("Invalid batch size");
    }

    const int output_size = desc_.nfft / 2 + 1;

    // 배치별 처리
    for (int b = 0; b < batch; ++b) {
      const float* batch_input = in + b * desc_.stride_in * desc_.nfft;
      std::complex<float>* batch_output = out + b * desc_.stride_out * output_size;

      // NaN/denormal 입력 검증 및 정리
      std::vector<float> cleaned_input(desc_.nfft);
      for (int i = 0; i < desc_.nfft; ++i) {
        float val = batch_input[i * desc_.stride_in];
        if (std::isnan(val) || std::isinf(val)) {
          val = 0.0f;  // NaN/inf를 0으로 대체
        } else if (std::abs(val) < 1e-30f) {
          val = 0.0f;  // denormal을 0으로
        }
        cleaned_input[i] = val;
      }

      // KissFFT Real 순방향 실행
      kiss_fftr(forward_real_cfg_, cleaned_input.data(),
                reinterpret_cast<kiss_fft_cpx*>(real_output_buffer_.data()));

      // 결과를 std::complex<float>로 복사 (스트라이드 적용)
      for (int i = 0; i < output_size; ++i) {
        batch_output[i * desc_.stride_out] = std::complex<float>(
          real_output_buffer_[i].r, real_output_buffer_[i].i);
      }
    }
  }

  void inverse(const std::complex<float>* in, float* out, int batch = 1) override {
    if (desc_.domain != FftDomain::Real) {
      throw std::runtime_error("Real FFT not supported for Complex domain plan");
    }

    // 피드백 반영: 배치 처리 지원
    if (batch < 1 || batch > desc_.batch) {
      throw std::runtime_error("Invalid batch size");
    }

    const int input_size = desc_.nfft / 2 + 1;
    std::vector<float> batch_output(desc_.nfft);

    // 배치별 처리
    for (int b = 0; b < batch; ++b) {
      const std::complex<float>* batch_input = in + b * desc_.stride_in * input_size;
      float* batch_output_ptr = out + b * desc_.stride_out * desc_.nfft;

      // 입력을 kiss_fft_cpx로 변환 (스트라이드 적용)
      for (int i = 0; i < input_size; ++i) {
        const auto& complex_val = batch_input[i * desc_.stride_in];
        real_output_buffer_[i].r = complex_val.real();
        real_output_buffer_[i].i = complex_val.imag();
      }

      // KissFFT Real 역방향 실행
      kiss_fftri(inverse_real_cfg_, reinterpret_cast<kiss_fft_cpx*>(real_output_buffer_.data()),
                 batch_output.data());

      // 정규화 및 출력 복사 (스트라이드 적용)
      float scale = 1.0f / desc_.nfft;
      for (int i = 0; i < desc_.nfft; ++i) {
        float val = batch_output[i] * scale;

        // NaN/denormal 출력 정리
        if (std::isnan(val) || std::isinf(val)) {
          val = 0.0f;
        } else if (std::abs(val) < 1e-30f) {
          val = 0.0f;
        }

        batch_output_ptr[i * desc_.stride_out] = val;
      }
    }
  }

  // Phase 3: Complex FFT 메서드 구현
  void forward_complex(const std::complex<float>* in, std::complex<float>* out, int batch = 1) override {
    if (desc_.domain != FftDomain::Complex) {
      throw std::runtime_error("Complex FFT not supported for Real domain plan");
    }

    // 피드백 반영: 배치 처리 지원
    if (batch < 1 || batch > desc_.batch) {
      throw std::runtime_error("Invalid batch size");
    }

    // 배치별 처리
    for (int b = 0; b < batch; ++b) {
      const std::complex<float>* batch_input = in + b * desc_.stride_in * desc_.nfft;
      std::complex<float>* batch_output = out + b * desc_.stride_out * desc_.nfft;

      // 입력을 kiss_fft_cpx로 변환 (스트라이드 적용)
      for (int i = 0; i < desc_.nfft; ++i) {
        const auto& complex_val = batch_input[i * desc_.stride_in];
        complex_buffer_[i].r = complex_val.real();
        complex_buffer_[i].i = complex_val.imag();
      }

      // KissFFT Complex 순방향 실행
      kiss_fft(forward_complex_cfg_, complex_buffer_.data(), complex_buffer_.data());

      // 결과를 std::complex<float>로 복사 (스트라이드 적용)
      for (int i = 0; i < desc_.nfft; ++i) {
        batch_output[i * desc_.stride_out] = std::complex<float>(
          complex_buffer_[i].r, complex_buffer_[i].i);
      }
    }
  }

  void inverse_complex(const std::complex<float>* in, std::complex<float>* out, int batch = 1) override {
    if (desc_.domain != FftDomain::Complex) {
      throw std::runtime_error("Complex FFT not supported for Real domain plan");
    }

    // 피드백 반영: 배치 처리 지원
    if (batch < 1 || batch > desc_.batch) {
      throw std::runtime_error("Invalid batch size");
    }

    // 배치별 처리
    for (int b = 0; b < batch; ++b) {
      const std::complex<float>* batch_input = in + b * desc_.stride_in * desc_.nfft;
      std::complex<float>* batch_output = out + b * desc_.stride_out * desc_.nfft;

      // 입력을 kiss_fft_cpx로 변환 (스트라이드 적용)
      for (int i = 0; i < desc_.nfft; ++i) {
        const auto& complex_val = batch_input[i * desc_.stride_in];
        complex_buffer_[i].r = complex_val.real();
        complex_buffer_[i].i = complex_val.imag();
      }

      // KissFFT Complex 역방향 실행
      kiss_fft(inverse_complex_cfg_, complex_buffer_.data(), complex_buffer_.data());

      // 정규화 및 결과 복사 (스트라이드 적용)
      float scale = 1.0f / desc_.nfft;
      for (int i = 0; i < desc_.nfft; ++i) {
        float real_val = complex_buffer_[i].r * scale;
        float imag_val = complex_buffer_[i].i * scale;

        // NaN/denormal 정리
        if (std::isnan(real_val) || std::isinf(real_val) || std::abs(real_val) < 1e-30f) {
          real_val = 0.0f;
        }
        if (std::isnan(imag_val) || std::isinf(imag_val) || std::abs(imag_val) < 1e-30f) {
          imag_val = 0.0f;
        }

        batch_output[i * desc_.stride_out] = std::complex<float>(real_val, imag_val);
      }
    }
  }

  // 플랜 정보 조회 메서드
  FftDomain domain() const override { return desc_.domain; }
  int size() const override { return desc_.nfft; }

private:
  FftPlanDesc desc_;

  // Real FFT 관련
  kiss_fftr_cfg forward_real_cfg_ = nullptr;
  kiss_fftr_cfg inverse_real_cfg_ = nullptr;
  std::vector<kiss_fft_cpx> real_output_buffer_;

  // Complex FFT 관련
  kiss_fft_cfg forward_complex_cfg_ = nullptr;
  kiss_fft_cfg inverse_complex_cfg_ = nullptr;
  std::vector<kiss_fft_cpx> complex_buffer_;
};

// 팩토리 함수 구현
std::unique_ptr<IFftPlan> MakeFftPlan(const FftPlanDesc& desc) {
  return std::make_unique<KissFftPlan>(desc);
}

}  // namespace dsp::fft