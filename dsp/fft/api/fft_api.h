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
  int batch;         // 배치 크기 (1 이상, 피드백 반영: 배치 지원)
  int stride_in;     // 입력 스트라이드 (배치 처리용)
  int stride_out;    // 출력 스트라이드 (배치 처리용)
};

// FFT 플랜 인터페이스
class IFftPlan {
public:
  virtual ~IFftPlan() = default;

  // 순방향 FFT: 실수 입력 -> 복소 출력 (RFFT)
  virtual void forward(const float* in, std::complex<float>* out, int batch = 1) = 0;

  // 역방향 FFT: 복소 입력 -> 실수 출력 (IRFFT)
  virtual void inverse(const std::complex<float>* in, float* out, int batch = 1) = 0;

  // Phase 3: Complex 도메인 확장 - 복소 입력/출력 FFT
  // 순방향 Complex FFT: 복소 입력 -> 복소 출력 (CFFT)
  virtual void forward_complex(const std::complex<float>* in, std::complex<float>* out, int batch = 1) = 0;

  // 역방향 Complex FFT: 복소 입력 -> 복소 출력 (ICFFT)
  virtual void inverse_complex(const std::complex<float>* in, std::complex<float>* out, int batch = 1) = 0;

  // 플랜 정보 조회
  virtual FftDomain domain() const = 0;
  virtual int size() const = 0;
  virtual bool supports_batch() const { return true; }   // 피드백 반영: 배치 지원
  virtual int max_batch_size() const { return 16; }      // 최대 배치 크기
};

// FFT 플랜 팩토리 함수
std::unique_ptr<IFftPlan> MakeFftPlan(const FftPlanDesc& desc);

}  // namespace dsp::fft

#endif  // DSP_FFT_API_H_