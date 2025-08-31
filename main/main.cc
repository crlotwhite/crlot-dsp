#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/base.h>
#include <fmt/chrono.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <sstream>

#include "io/wav.h"
#include "dsp/fft/api/fft_api.h"

// Eigen 헤더
#include "Eigen/Dense"

// Google Highway 헤더
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace hwy_example {
namespace HWY_NAMESPACE {

void ScalePcmData(const float* input, float* output, size_t size, float scale) {
  const HWY_FULL(float) d;  // 현재 타겟에 맞는 SIMD 레인 수 결정
  size_t i = 0;

  // SIMD 레인 단위로 처리
  for (; i + Lanes(d) <= size; i += Lanes(d)) {
    const auto v = Load(d, input + i);  // SIMD 레인에 데이터 로드
    const auto scaled = Mul(v, Set(d, scale));  // 스케일링
    Store(scaled, d, output + i);  // 결과 저장
  }

  // 남은 요소들을 스칼라 방식으로 처리
  for (; i < size; ++i) {
    output[i] = input[i] * scale;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy_example

HWY_AFTER_NAMESPACE();

// r8brain-free 헤더
#include "CDSPResampler.h"

// cpu_features 헤더
#include "cpu_features_macros.h"
#if defined(CPU_FEATURES_ARCH_X86)
#include "cpuinfo_x86.h"
#elif defined(CPU_FEATURES_ARCH_AARCH64)
#include "cpuinfo_aarch64.h"
#else
// 다른 아키텍처 지원 가능
#endif

// Bazel 환경에서 파일 경로를 올바르게 처리하는 헬퍼 함수
static std::string RunfilePath(const std::string& relative) {
    const char* srcdir = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    if (!srcdir || !workspace) return relative; // fallback for non-bazel
    return std::string(srcdir) + "/" + workspace + "/" + relative;
}

int main() {
  // spdlog 로거 설정
  auto console = spdlog::stdout_color_mt("console");
  spdlog::set_default_logger(console);
  spdlog::set_level(spdlog::level::info);

  SPDLOG_INFO("프로그램 시작");

  // CPU 기능 정보 출력 (cpu_features 사용)
#if defined(CPU_FEATURES_ARCH_X86)
  const cpu_features::X86Info info = cpu_features::GetX86Info();
  SPDLOG_INFO("CPU 아키텍처: x86");
  SPDLOG_INFO("CPU 브랜드: {}", info.brand_string);
  SPDLOG_INFO("CPU 패밀리: {}, 모델: {}, 스테핑: {}", info.family, info.model, info.stepping);
  SPDLOG_INFO("CPU 기능: SSE={}, SSE2={}, SSE3={}, SSSE3={}, SSE4_1={}, SSE4_2={}, AVX={}",
              info.features.sse, info.features.sse2, info.features.sse3,
              info.features.ssse3, info.features.sse4_1, info.features.sse4_2,
              info.features.avx);
#elif defined(CPU_FEATURES_ARCH_AARCH64)
  const cpu_features::Aarch64Info info = cpu_features::Aarch64Info();
  SPDLOG_INFO("CPU 아키텍처: aarch64");
  SPDLOG_INFO("CPU 구현자: {}, 변형: {}, 파트: {}, 수정: {}", info.implementer, info.variant, info.part, info.revision);
  SPDLOG_INFO("CPU 기능: FP={}, ASIMD={}, AES={}, PMULL={}, SHA1={}, SHA2={}, CRC32={}",
              info.features.fp, info.features.asimd, info.features.aes,
              info.features.pmull, info.features.sha1, info.features.sha2,
              info.features.crc32);
#else
  SPDLOG_INFO("CPU 아키텍처: 지원되지 않는 아키텍처");
#endif

  SPDLOG_INFO("Hello, world!");

  auto now = std::chrono::system_clock::now();
  SPDLOG_INFO("Date and time: {}", now);
  SPDLOG_INFO("Time: {:%H:%M}", now);

  // WavReader를 사용해서 WAV 파일 읽기
  std::string wav_path = RunfilePath("assets/oboe.wav");
  SPDLOG_INFO("WAV 파일 읽기 시작: {}", wav_path);
  WavReader reader;
  if (!reader.open(wav_path)) {
    SPDLOG_ERROR("WAV 파일 열기 실패: {}", wav_path);
    return -1;
  } else {
    SPDLOG_INFO("WAV 파일 열기 성공");
  }

  // 오디오 정보 출력
  SPDLOG_INFO("오디오 정보 - 채널: {}, 샘플레이트: {} Hz, 총 프레임: {}, 비트 심도: {}",
              reader.get_channels(), reader.get_sample_rate(),
              reader.get_total_frames(), reader.get_bits_per_sample());

  // 전체 오디오 데이터 읽기
  SPDLOG_INFO("PCM 데이터 읽기 시작");
  std::vector<float> pcm = reader.read_all();
  if (pcm.empty()) {
    SPDLOG_ERROR("PCM 데이터 읽기 실패");
    reader.close();
    return -1;
  }
  SPDLOG_INFO("PCM 데이터 읽기 완료, 샘플 수: {}", pcm.size());

  size_t totalFrames = reader.get_total_frames();
  size_t channels = reader.get_channels();
  unsigned int sampleRate = reader.get_sample_rate();

  reader.close();

  // Choose FFT size: 4096 or the largest power-of-two <= totalFrames
  SPDLOG_INFO("FFT 크기 결정 시작");
  size_t N = 4096;
  if (totalFrames < N) {
    size_t p = 1;
    while (p * 2 <= totalFrames) p *= 2;
    N = std::max<size_t>(1, p);
  }
  if (N < 2) {
    SPDLOG_ERROR("FFT를 위한 충분한 샘플이 없음");
    return -1;
  }
  SPDLOG_INFO("FFT 크기: {}", N);

  // Prepare real input with a Hann window (mix down to mono)
  SPDLOG_INFO("FFT를 위한 입력 데이터 준비 시작");
  std::vector<float> fft_input(N);
  const float PI = 3.14159265358979323846f;
  for (size_t i = 0; i < N; ++i) {
    // mix to mono by averaging channels (if stereo)
    float sample = 0.0f;
    for (size_t c = 0; c < channels; ++c) {
      sample += pcm[i * channels + c];
    }
    sample /= static_cast<float>(channels);

    // Hann window
    float w = 0.5f * (1.0f - std::cos(2.0f * PI * static_cast<float>(i) / static_cast<float>(N - 1)));
    fft_input[i] = sample * w;
  }
  SPDLOG_INFO("FFT 입력 데이터 준비 완료");

  // FFT 설정 및 실행
  SPDLOG_INFO("FFT 설정 및 실행 시작");
  dsp::fft::FftPlanDesc desc{
    dsp::fft::FftDomain::Real,
    static_cast<int>(N),
    false,  // in_place
    1,      // batch
    1,      // stride_in
    1       // stride_out
  };

  auto fft_plan = dsp::fft::MakeFftPlan(desc);
  if (!fft_plan) {
    SPDLOG_ERROR("FFT 플랜 생성 실패");
    return -1;
  }
  SPDLOG_INFO("FFT 플랜 생성 완료, FFT 실행 중...");

  std::vector<std::complex<float>> fft_output(N / 2 + 1);
  fft_plan->forward(fft_input.data(), fft_output.data());
  SPDLOG_INFO("FFT 실행 완료");

  // Compute magnitudes for first N/2 bins and find top peaks
  size_t half = N / 2;
  std::vector<std::pair<float, size_t>> mags;
  mags.reserve(half);
  for (size_t k = 0; k < half; ++k) {
    float m = std::abs(fft_output[k]);
    mags.emplace_back(m, k);
  }
  std::sort(mags.begin(), mags.end(), [](auto &a, auto &b){ return a.first > b.first; });

  // Print top 10 peaks with frequency in Hz
  size_t topN = std::min<size_t>(10, mags.size());
  SPDLOG_INFO("상위 {}개 주파수 분석 결과:", topN);
  for (size_t i = 0; i < topN; ++i) {
    float magnitude = mags[i].first;
    size_t bin = mags[i].second;
    double freq = static_cast<double>(bin) * static_cast<double>(sampleRate) / static_cast<double>(N);
    SPDLOG_INFO("  {:2}: 크기={:.6f}, 빈={}, 주파수={:.2f} Hz", (int)i + 1, magnitude, bin, freq);
  }

  SPDLOG_INFO("FFT 처리 완료");

  // WavWriter를 사용해서 간단한 톤 생성 및 저장 테스트
  SPDLOG_INFO("WAV 파일 쓰기 테스트 시작");
  WavWriter writer;
  if (writer.open("test_output.wav", 1, 44100, 16)) {
    SPDLOG_INFO("WavWriter 열기 성공");

    // 간단한 사인파 생성 (440Hz, 1초)
    const size_t tone_frames = 44100;
    std::vector<float> tone_data(tone_frames);
    for (size_t i = 0; i < tone_frames; ++i) {
      tone_data[i] = 0.5f * std::sin(2.0f * PI * 440.0f * static_cast<float>(i) / 44100.0f);
    }

    size_t frames_written;
    if (writer.write(tone_data.data(), tone_frames, &frames_written)) {
      SPDLOG_INFO("WAV 파일에 {} 프레임 쓰기 성공: test_output.wav", frames_written);
    } else {
      SPDLOG_ERROR("WAV 파일 쓰기 실패");
    }

    writer.close();
    SPDLOG_INFO("WavWriter 닫기 완료");
  } else {
    SPDLOG_ERROR("WavWriter 열기 실패");
  }

  // r8brain-free 리샘플링 테스트 (실제 oboe.wav 파일 사용)
  SPDLOG_INFO("r8brain-free 리샘플링 테스트 시작 (oboe.wav 파일 사용)");
  try {
    // WAV 파일에서 읽은 데이터를 사용
    const double srcSampleRate = static_cast<double>(sampleRate); // WAV 파일의 실제 샘플레이트
    const double dstSampleRate = 48000.0; // 목표 샘플레이트 (48kHz로 업샘플링)

    SPDLOG_INFO("WAV 파일 리샘플링: {}Hz -> {}Hz", srcSampleRate, dstSampleRate);

    // PCM 데이터를 mono로 변환하고 double 타입으로 변환
    std::vector<double> inputData(totalFrames);
    for (size_t i = 0; i < totalFrames; ++i) {
      double sample = 0.0;
      for (size_t c = 0; c < channels; ++c) {
        sample += pcm[i * channels + c];
      }
      inputData[i] = sample / static_cast<double>(channels);
    }

    SPDLOG_INFO("r8b::CDSPResampler 생성 시도");
    // r8b::CDSPResampler 생성 (원본 샘플레이트, 목표 샘플레이트, 최대 입력 길이)
    r8b::CDSPResampler resampler(srcSampleRate, dstSampleRate, static_cast<int>(totalFrames));

    SPDLOG_INFO("r8b::CDSPResampler 생성 성공");

    // 출력 버퍼 크기 계산
    int maxOutputSize = static_cast<int>(std::ceil(totalFrames * dstSampleRate / srcSampleRate) + 1000);
    std::vector<double> outputData(maxOutputSize, 0.0);

    SPDLOG_INFO("리샘플링 실행: 입력 {} 샘플 -> 예상 출력 {} 샘플", totalFrames, maxOutputSize);

    // 스트리밍 방식으로 리샘플링 테스트
    int totalOutputSamples = 0;
    const int chunkSize = 4096; // 청크 단위로 처리

    for (size_t offset = 0; offset < totalFrames; offset += chunkSize) {
      size_t currentChunkSize = std::min(static_cast<size_t>(chunkSize), totalFrames - offset);
      double* outputPtr;

      int chunkOutputSamples = resampler.process(
        inputData.data() + offset, static_cast<int>(currentChunkSize), outputPtr);

      if (chunkOutputSamples > 0 && totalOutputSamples + chunkOutputSamples <= maxOutputSize) {
        // 출력 데이터를 복사
        std::copy(outputPtr, outputPtr + chunkOutputSamples,
                 outputData.data() + totalOutputSamples);
        totalOutputSamples += chunkOutputSamples;
      }
    }

    SPDLOG_INFO("리샘플링 완료, 총 출력 샘플 수: {}", totalOutputSamples);

    // 결과 검증
    if (totalOutputSamples > 0) {
      SPDLOG_INFO("r8brain-free 리샘플링 성공!");
      SPDLOG_INFO("원본 오디오: {}Hz -> 리샘플링 후: {}Hz",
                  srcSampleRate, dstSampleRate);
      SPDLOG_INFO("원본 길이: {:.2f}초 -> 리샘플링 후 길이: {:.2f}초",
                  static_cast<double>(totalFrames) / srcSampleRate,
                  static_cast<double>(totalOutputSamples) / dstSampleRate);

      // 첫 몇 개 샘플 값 출력
      SPDLOG_INFO("처음 5개 입력 샘플 (원본 WAV 데이터):");
      for (size_t i = 0; i < std::min<size_t>(5, totalFrames); ++i) {
        SPDLOG_INFO("  입력[{}]: {:.6f}", i, inputData[i]);
      }

      SPDLOG_INFO("처음 5개 출력 샘플 (리샘플링 후):");
      for (int i = 0; i < std::min(5, totalOutputSamples); ++i) {
        SPDLOG_INFO("  출력[{}]: {:.6f}", i, outputData[i]);
      }

      // 리샘플링 비율 검증
      double expectedRatio = dstSampleRate / srcSampleRate;
      double actualRatio = static_cast<double>(totalOutputSamples) / totalFrames;
      SPDLOG_INFO("예상 리샘플링 비율: {:.6f}, 실제 비율: {:.6f}", expectedRatio, actualRatio);

      // 성공 지표
      double ratioDiff = std::abs(expectedRatio - actualRatio);
      if (ratioDiff < 0.01) {
        SPDLOG_INFO("리샘플링 비율이 정확합니다! (차이: {:.6f})", ratioDiff);
      } else {
        SPDLOG_WARN("리샘플링 비율이 약간 다릅니다 (차이: {:.6f})", ratioDiff);
      }

      // 리샘플링된 오디오를 파일로 저장 (테스트용)
      SPDLOG_INFO("리샘플링된 오디오를 파일로 저장");
      WavWriter resampledWriter;
      if (resampledWriter.open("resampled_oboe.wav", 1, static_cast<int>(dstSampleRate), 16)) {
        // double 데이터를 float으로 변환
        std::vector<float> resampledPcm(totalOutputSamples);
        for (int i = 0; i < totalOutputSamples; ++i) {
          resampledPcm[i] = static_cast<float>(outputData[i]);
        }

        size_t framesWritten;
        if (resampledWriter.write(resampledPcm.data(), totalOutputSamples, &framesWritten)) {
          SPDLOG_INFO("리샘플링된 오디오 저장 성공: {} 프레임 (resampled_oboe.wav)", framesWritten);
        } else {
          SPDLOG_ERROR("리샘플링된 오디오 저장 실패");
        }
        resampledWriter.close();
      } else {
        SPDLOG_ERROR("리샘플링된 오디오 파일 열기 실패");
      }

    } else {
      SPDLOG_ERROR("r8brain-free 리샘플링 실패 - 출력 샘플이 생성되지 않았습니다");
    }

  } catch (const std::exception& e) {
    SPDLOG_ERROR("r8brain-free 테스트 중 예외 발생: {}", e.what());
  } catch (...) {
    SPDLOG_ERROR("r8brain-free 테스트 중 알 수 없는 예외 발생");
  }

  // Google Highway SIMD 예시: pcm 데이터를 벡터화된 방식으로 스케일링
  SPDLOG_INFO("Google Highway SIMD 예시 시작");
  try {
    // 예시 실행
    std::vector<float> scaled_pcm(pcm.size());
    const float scale_factor = 0.5f;  // 0.5배 스케일링

    SPDLOG_INFO("Highway를 사용한 벡터화된 스케일링 실행 (스케일 팩터: {:.2f})", scale_factor);
    hwy_example::HWY_NAMESPACE::ScalePcmData(pcm.data(), scaled_pcm.data(), pcm.size(), scale_factor);

    SPDLOG_INFO("Highway SIMD 스케일링 완료, 처리된 샘플 수: {}", scaled_pcm.size());

    // 결과 검증: 처음 몇 개 샘플 출력
    SPDLOG_INFO("원본 pcm 샘플 (처음 5개):");
    for (size_t i = 0; i < std::min<size_t>(5, pcm.size()); ++i) {
      SPDLOG_INFO("  원본[{}]: {:.6f}", i, pcm[i]);
    }

    SPDLOG_INFO("스케일링된 pcm 샘플 (처음 5개):");
    for (size_t i = 0; i < std::min<size_t>(5, scaled_pcm.size()); ++i) {
      SPDLOG_INFO("  스케일링[{}]: {:.6f}", i, scaled_pcm[i]);
    }

    SPDLOG_INFO("Google Highway SIMD 예시 성공!");

  } catch (const std::exception& e) {
    SPDLOG_ERROR("Google Highway 예시 중 예외 발생: {}", e.what());
  } catch (...) {
    SPDLOG_ERROR("Google Highway 예시 중 알 수 없는 예외 발생");
  }

  // Eigen 예시: 간단한 행렬 연산
  SPDLOG_INFO("Eigen 예시 시작");
  try {
    // 2x2 행렬 생성 및 초기화
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    std::stringstream ss_m;
    ss_m << m;
    SPDLOG_INFO("행렬 m:\n{}", ss_m.str());

    // 다른 행렬 생성
    Eigen::MatrixXd n(2, 2);
    n << 1, 2,
         3, 4;
    std::stringstream ss_n;
    ss_n << n;
    SPDLOG_INFO("행렬 n:\n{}", ss_n.str());

    // 행렬 곱셈
    Eigen::MatrixXd p = m * n;
    std::stringstream ss_p;
    ss_p << p;
    SPDLOG_INFO("m * n:\n{}", ss_p.str());

    // 역행렬 계산
    Eigen::MatrixXd inv_m = m.inverse();
    std::stringstream ss_inv;
    ss_inv << inv_m;
    SPDLOG_INFO("m의 역행렬:\n{}", ss_inv.str());

    SPDLOG_INFO("Eigen 예시 성공!");
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Eigen 예시 중 예외 발생: {}", e.what());
  } catch (...) {
    SPDLOG_ERROR("Eigen 예시 중 알 수 없는 예외 발생");
  }

  SPDLOG_INFO("프로그램 종료");
  return 0;
}