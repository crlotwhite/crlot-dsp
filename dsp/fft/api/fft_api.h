#ifndef DSP_FFT_API_H_
#define DSP_FFT_API_H_

#include <memory>
#include <complex>

namespace dsp::fft {

// FFT 도메인 타입
enum class FftDomain {
  Real,    // 실수 입력 -> 복소 출력 (RFFT)
  Complex  // 복소 입력 -> 복소 출력 (CFFT)
};

// FFT 플랜 설명 구조체
struct FftPlanDesc {
  FftDomain domain;  // FFT 타입
  int nfft;          // FFT 크기 (샘플 수)
  bool in_place;     // 제자리 연산 여부 (현재 미지원, 향후 확장)
  int batch;         // 배치 크기 (현재 1로 제한)
  int stride_in;     // 입력 스트라이드 (현재 1로 제한)
  int stride_out;    // 출력 스트라이드 (현재 1로 제한)
};

// FFT 플랜 인터페이스
class IFftPlan {
public:
  virtual ~IFftPlan() = default;

  // 순방향 FFT: 실수 입력 -> 복소 출력 (RFFT)
  virtual void forward(const float* in, std::complex<float>* out, int batch = 1) = 0;

  // 역방향 FFT: 복소 입력 -> 실수 출력 (IRFFT)
  virtual void inverse(const std::complex<float>* in, float* out, int batch = 1) = 0;
};

// FFT 플랜 팩토리 함수
std::unique_ptr<IFftPlan> MakeFftPlan(const FftPlanDesc& desc);

}  // namespace dsp::fft

#endif  // DSP_FFT_API_H_