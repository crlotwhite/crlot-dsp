#include <fmt/base.h>
#include <fmt/chrono.h>

int main() {
  fmt::print("Hello, world!\n");

  auto now = std::chrono::system_clock::now();
  fmt::print("Date and time: {}\n", now);
  fmt::print("Time: {:%H:%M}\n", now);

  return 0;
}