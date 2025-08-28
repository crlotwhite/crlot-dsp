#include <fmt/base.h>
#include <fmt/chrono.h>

#define DR_WAV_IMPLEMENTATION
#include "dr_libs/dr_wav.h"

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

  return 0;
}