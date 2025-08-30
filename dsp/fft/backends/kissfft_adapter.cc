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
    // 현재 Real 도메인만 지원
    if (desc.domain != FftDomain::Real) {
      throw std::runtime_error("Only Real domain is currently supported");
    }

    // stride/batch는 1로 제한
    if (desc.batch != 1 || desc.stride_in != 1 || desc.stride_out != 1) {
      throw std::runtime_error("Batch and stride must be 1 (currently limited)");
    }

    // in_place는 현재 미지원
    if (desc.in_place) {
      throw std::runtime_error("In-place FFT is not yet supported");
    }

    // FFT 크기는 짝수여야 함
    if (desc.nfft % 2 != 0) {
      throw std::runtime_error("FFT size must be even for real FFT");
    }

    // KissFFT 설정 할당
    forward_cfg_ = kiss_fftr_alloc(desc.nfft, 0, nullptr, nullptr);
    inverse_cfg_ = kiss_fftr_alloc(desc.nfft, 1, nullptr, nullptr);

    if (!forward_cfg_ || !inverse_cfg_) {
      throw std::runtime_error("Failed to allocate KissFFT configuration");
    }

    // 내부 버퍼 할당 (출력 크기: nfft/2 + 1)
    output_buffer_.resize(desc.nfft / 2 + 1);
  }

  ~KissFftPlan() override {
    if (forward_cfg_) {
      kiss_fftr_free(forward_cfg_);
    }
    if (inverse_cfg_) {
      kiss_fftr_free(inverse_cfg_);
    }
  }

  void forward(const float* in, std::complex<float>* out, int batch = 1) override {
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

    // KissFFT 순방향 실행
    kiss_fftr(forward_cfg_, cleaned_input.data(),
              reinterpret_cast<kiss_fft_cpx*>(output_buffer_.data()));

    // 결과를 std::complex<float>로 복사
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
      out[i] = std::complex<float>(output_buffer_[i].r, output_buffer_[i].i);
    }
  }

  void inverse(const std::complex<float>* in, float* out, int batch = 1) override {
    if (batch != 1) {
      throw std::runtime_error("Batch size must be 1");
    }

    // 입력을 kiss_fft_cpx로 변환
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
      output_buffer_[i].r = in[i].real();
      output_buffer_[i].i = in[i].imag();
    }

    // KissFFT 역방향 실행
    kiss_fftri(inverse_cfg_, reinterpret_cast<kiss_fft_cpx*>(output_buffer_.data()), out);

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

private:
  FftPlanDesc desc_;
  kiss_fftr_cfg forward_cfg_ = nullptr;
  kiss_fftr_cfg inverse_cfg_ = nullptr;
  std::vector<kiss_fft_cpx> output_buffer_;
};

// 팩토리 함수 구현
std::unique_ptr<IFftPlan> MakeFftPlan(const FftPlanDesc& desc) {
  return std::make_unique<KissFftPlan>(desc);
}

}  // namespace dsp::fft