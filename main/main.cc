#include <fmt/base.h>
#include <fmt/chrono.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#include <kissfft/kiss_fft.h>

int main() {
  fmt::print("Hello, world!\n");

  auto now = std::chrono::system_clock::now();
  fmt::print("Date and time: {}\n", now);
  fmt::print("Time: {:%H:%M}\n", now);

  drwav wav;
  if (!drwav_init_file(&wav, "assets/oboe.wav", nullptr)) {
    fmt::print("Failed to open WAV file.\n");
    return -1;
  } else {
    fmt::print("WAV file opened successfully.\n");
  }

  // Read all frames as 32-bit floats
  drwav_uint64 totalFrames = wav.totalPCMFrameCount;
  if (totalFrames == 0) {
    fmt::print("No audio frames in WAV.\n");
    drwav_uninit(&wav);
    return -1;
  }
  size_t channels = wav.channels;
  size_t framesToRead = static_cast<size_t>(totalFrames);

  std::vector<float> pcm(framesToRead * channels);
  drwav_uint64 framesRead = drwav_read_pcm_frames_f32(&wav, framesToRead, pcm.data());
  if (framesRead == 0) {
    fmt::print("Failed to read PCM frames.\n");
    drwav_uninit(&wav);
    return -1;
  }
  size_t framesReadS = static_cast<size_t>(framesRead);

  // Choose FFT size: 4096 or the largest power-of-two <= framesRead
  size_t N = 4096;
  if (framesReadS < N) {
    size_t p = 1;
    while (p * 2 <= framesReadS) p *= 2;
    N = std::max<size_t>(1, p);
  }
  if (N < 2) {
    fmt::print("Not enough samples for FFT.\n");
    drwav_uninit(&wav);
    return -1;
  }
  fmt::print("FFT size: {}\n", N);

  // Prepare real->complex input with a Hann window (mix down to mono)
  kiss_fft_cpx* in = (kiss_fft_cpx*)std::malloc(sizeof(kiss_fft_cpx) * N);
  kiss_fft_cpx* out = (kiss_fft_cpx*)std::malloc(sizeof(kiss_fft_cpx) * N);
  if (!in || !out) {
    fmt::print("Allocation failed.\n");
    std::free(in); std::free(out);
    drwav_uninit(&wav);
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
    drwav_uninit(&wav);
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
  unsigned int sampleRate = wav.sampleRate;
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
  drwav_uninit(&wav);


  return 0;
}