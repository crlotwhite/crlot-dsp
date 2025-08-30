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

    // stride/batch는 1로 제한
    if (desc.batch != 1 || desc.stride_in != 1 || desc.stride_out != 1) {
      throw std::runtime_error("Batch and stride must be 1 (currently limited)");
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
    if (batch != 1) {
      throw std::runtime_error("Batch size must be 1");
    }

    // NaN/denormal 입력 검증 및 정리
    std::vector<float> cleaned_input(desc_.nfft);
    for (int i = 0; i < desc_.nfft; ++i) {
      float val = in[i];
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

    // 결과를 std::complex<float>로 복사
    for (size_t i = 0; i < real_output_buffer_.size(); ++i) {
      out[i] = std::complex<float>(real_output_buffer_[i].r, real_output_buffer_[i].i);
    }
  }

  void inverse(const std::complex<float>* in, float* out, int batch = 1) override {
    if (desc_.domain != FftDomain::Real) {
      throw std::runtime_error("Real FFT not supported for Complex domain plan");
    }
    if (batch != 1) {
      throw std::runtime_error("Batch size must be 1");
    }

    // 입력을 kiss_fft_cpx로 변환
    for (size_t i = 0; i < real_output_buffer_.size(); ++i) {
      real_output_buffer_[i].r = in[i].real();
      real_output_buffer_[i].i = in[i].imag();
    }

    // KissFFT Real 역방향 실행
    kiss_fftri(inverse_real_cfg_, reinterpret_cast<kiss_fft_cpx*>(real_output_buffer_.data()), out);

    // 정규화 (FFT 크기로 나누기)
    float scale = 1.0f / desc_.nfft;
    for (int i = 0; i < desc_.nfft; ++i) {
      out[i] *= scale;

      // NaN/denormal 출력 정리
      if (std::isnan(out[i]) || std::isinf(out[i])) {
        out[i] = 0.0f;
      } else if (std::abs(out[i]) < 1e-30f) {
        out[i] = 0.0f;
      }
    }
  }

  // Phase 3: Complex FFT 메서드 구현
  void forward_complex(const std::complex<float>* in, std::complex<float>* out, int batch = 1) override {
    if (desc_.domain != FftDomain::Complex) {
      throw std::runtime_error("Complex FFT not supported for Real domain plan");
    }
    if (batch != 1) {
      throw std::runtime_error("Batch size must be 1");
    }

    // 입력을 kiss_fft_cpx로 변환
    for (int i = 0; i < desc_.nfft; ++i) {
      complex_buffer_[i].r = in[i].real();
      complex_buffer_[i].i = in[i].imag();
    }

    // KissFFT Complex 순방향 실행
    kiss_fft(forward_complex_cfg_, complex_buffer_.data(), complex_buffer_.data());

    // 결과를 std::complex<float>로 복사
    for (int i = 0; i < desc_.nfft; ++i) {
      out[i] = std::complex<float>(complex_buffer_[i].r, complex_buffer_[i].i);
    }
  }

  void inverse_complex(const std::complex<float>* in, std::complex<float>* out, int batch = 1) override {
    if (desc_.domain != FftDomain::Complex) {
      throw std::runtime_error("Complex FFT not supported for Real domain plan");
    }
    if (batch != 1) {
      throw std::runtime_error("Batch size must be 1");
    }

    // 입력을 kiss_fft_cpx로 변환
    for (int i = 0; i < desc_.nfft; ++i) {
      complex_buffer_[i].r = in[i].real();
      complex_buffer_[i].i = in[i].imag();
    }

    // KissFFT Complex 역방향 실행
    kiss_fft(inverse_complex_cfg_, complex_buffer_.data(), complex_buffer_.data());

    // 정규화 및 결과 복사
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

      out[i] = std::complex<float>(real_val, imag_val);
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