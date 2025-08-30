#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/base.h>
#include <fmt/chrono.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#include "io/wav.h"
#include "kiss_fft.h"

int main() {
  // spdlog 로거 설정
  auto console = spdlog::stdout_color_mt("console");
  spdlog::set_default_logger(console);
  spdlog::set_level(spdlog::level::info);

  SPDLOG_INFO("프로그램 시작");

  SPDLOG_INFO("Hello, world!");

  auto now = std::chrono::system_clock::now();
  SPDLOG_INFO("Date and time: {}", now);
  SPDLOG_INFO("Time: {:%H:%M}", now);

  // WavReader를 사용해서 WAV 파일 읽기
  SPDLOG_INFO("WAV 파일 읽기 시작: assets/oboe.wav");
  WavReader reader;
  if (!reader.open("assets/oboe.wav")) {
    SPDLOG_ERROR("WAV 파일 열기 실패: assets/oboe.wav");
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

  // Prepare real->complex input with a Hann window (mix down to mono)
  SPDLOG_INFO("FFT를 위한 메모리 할당 시작");
  kiss_fft_cpx* in = (kiss_fft_cpx*)std::malloc(sizeof(kiss_fft_cpx) * N);
  kiss_fft_cpx* out = (kiss_fft_cpx*)std::malloc(sizeof(kiss_fft_cpx) * N);
  if (!in || !out) {
    SPDLOG_ERROR("메모리 할당 실패");
    std::free(in); std::free(out);
    return -1;
  }
  SPDLOG_INFO("메모리 할당 완료");
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
    in[i].r = sample * w;
    in[i].i = 0.0f;
  }

  // FFT
  SPDLOG_INFO("FFT 설정 및 실행 시작");
  kiss_fft_cfg cfg = kiss_fft_alloc(static_cast<int>(N), 0, nullptr, nullptr);
  if (!cfg) {
    SPDLOG_ERROR("kiss_fft_alloc 실패");
    std::free(in); std::free(out);
    return -1;
  }
  SPDLOG_INFO("FFT 설정 완료, FFT 실행 중...");
  kiss_fft(cfg, in, out);
  SPDLOG_INFO("FFT 실행 완료");

  // Compute magnitudes for first N/2 bins and find top peaks
  size_t half = N / 2;
  std::vector<std::pair<float, size_t>> mags;
  mags.reserve(half);
  for (size_t k = 0; k < half; ++k) {
    float m = std::sqrt(out[k].r * out[k].r + out[k].i * out[k].i);
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

  // Cleanup
  SPDLOG_INFO("메모리 정리 중...");
  std::free(in);
  std::free(out);
  std::free(cfg); // kiss_fft_alloc uses malloc internally
  SPDLOG_INFO("메모리 정리 완료");

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

  SPDLOG_INFO("프로그램 종료");
  return 0;
}