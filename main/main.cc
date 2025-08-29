#include <fmt/base.h>
#include <fmt/chrono.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#include "io/wav.h"
#include <kissfft/kiss_fft.h>

int main() {
  fmt::print("Hello, world!\n");

  auto now = std::chrono::system_clock::now();
  fmt::print("Date and time: {}\n", now);
  fmt::print("Time: {:%H:%M}\n", now);

  // WavReader를 사용해서 WAV 파일 읽기
  WavReader reader;
  if (!reader.open("assets/oboe.wav")) {
    fmt::print("Failed to open WAV file with WavReader.\n");
    return -1;
  } else {
    fmt::print("WAV file opened successfully with WavReader.\n");
  }

  // 오디오 정보 출력
  fmt::print("Channels: {}\n", reader.get_channels());
  fmt::print("Sample Rate: {} Hz\n", reader.get_sample_rate());
  fmt::print("Total Frames: {}\n", reader.get_total_frames());
  fmt::print("Bits per Sample: {}\n", reader.get_bits_per_sample());

  // 전체 오디오 데이터 읽기
  std::vector<float> pcm = reader.read_all();
  if (pcm.empty()) {
    fmt::print("Failed to read PCM data.\n");
    reader.close();
    return -1;
  }

  size_t totalFrames = reader.get_total_frames();
  size_t channels = reader.get_channels();
  unsigned int sampleRate = reader.get_sample_rate();

  reader.close();

  // Choose FFT size: 4096 or the largest power-of-two <= totalFrames
  size_t N = 4096;
  if (totalFrames < N) {
    size_t p = 1;
    while (p * 2 <= totalFrames) p *= 2;
    N = std::max<size_t>(1, p);
  }
  if (N < 2) {
    fmt::print("Not enough samples for FFT.\n");
    return -1;
  }
  fmt::print("FFT size: {}\n", N);

  // Prepare real->complex input with a Hann window (mix down to mono)
  kiss_fft_cpx* in = (kiss_fft_cpx*)std::malloc(sizeof(kiss_fft_cpx) * N);
  kiss_fft_cpx* out = (kiss_fft_cpx*)std::malloc(sizeof(kiss_fft_cpx) * N);
  if (!in || !out) {
    fmt::print("Allocation failed.\n");
    std::free(in); std::free(out);
    return -1;
  }
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
  kiss_fft_cfg cfg = kiss_fft_alloc(static_cast<int>(N), 0, nullptr, nullptr);
  if (!cfg) {
    fmt::print("kiss_fft_alloc failed.\n");
    std::free(in); std::free(out);
    return -1;
  }
  kiss_fft(cfg, in, out);

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
  fmt::print("Top {} frequency bins:\n", topN);
  for (size_t i = 0; i < topN; ++i) {
    float magnitude = mags[i].first;
    size_t bin = mags[i].second;
    double freq = static_cast<double>(bin) * static_cast<double>(sampleRate) / static_cast<double>(N);
    fmt::print("{:2}: mag={:.6f}, bin={}, freq={:.2f} Hz\n", (int)i + 1, magnitude, bin, freq);
  }

  // Cleanup
  std::free(in);
  std::free(out);
  std::free(cfg); // kiss_fft_alloc uses malloc internally

  // WavWriter를 사용해서 간단한 톤 생성 및 저장 테스트
  WavWriter writer;
  if (writer.open("test_output.wav", 1, 44100, 16)) {
    fmt::print("WavWriter opened successfully.\n");

    // 간단한 사인파 생성 (440Hz, 1초)
    const size_t tone_frames = 44100;
    std::vector<float> tone_data(tone_frames);
    for (size_t i = 0; i < tone_frames; ++i) {
      tone_data[i] = 0.5f * std::sin(2.0f * PI * 440.0f * static_cast<float>(i) / 44100.0f);
    }

    size_t frames_written;
    if (writer.write(tone_data.data(), tone_frames, &frames_written)) {
      fmt::print("Wrote {} frames to test_output.wav\n", frames_written);
    } else {
      fmt::print("Failed to write to WAV file.\n");
    }

    writer.close();
  } else {
    fmt::print("Failed to open WavWriter.\n");
  }

  return 0;
}