#include <benchmark/benchmark.h>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <array>
#include <sstream>

// 리그래션 가드: 벤치마크 결과를 기준선과 비교
class RegressionGuard {
public:
    struct BenchmarkResult {
        std::string name;
        double time_us;
        double tolerance_percent = 10.0; // ±10% 허용 밴드
    };

    // 기준선 로드 (JSON 또는 텍스트 파일)
    bool loadBaseline(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open baseline file: " << filename << std::endl;
            return false;
        }

        baseline_.clear();
        std::string line;
        while (std::getline(file, line)) {
            // 간단한 파싱: "name: time_us" 형식
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string name = line.substr(0, colon_pos);
                double time_us = std::stod(line.substr(colon_pos + 1));
                baseline_[name] = time_us;
            }
        }

        return true;
    }

    // 기준선 저장
    bool saveBaseline(const std::string& filename, const std::vector<BenchmarkResult>& results) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to save baseline file: " << filename << std::endl;
            return false;
        }

        for (const auto& result : results) {
            file << result.name << ": " << result.time_us << std::endl;
        }

        return true;
    }

    // 리그래션 체크
    bool checkRegression(const std::vector<BenchmarkResult>& current_results) {
        bool all_passed = true;

        std::cout << "\n=== Regression Check Results ===\n";

        for (const auto& current : current_results) {
            auto it = baseline_.find(current.name);
            if (it == baseline_.end()) {
                std::cout << "⚠️  New benchmark: " << current.name << " (" << current.time_us << " us)\n";
                continue;
            }

            double baseline_time = it->second;
            double diff_percent = ((current.time_us - baseline_time) / baseline_time) * 100.0;

            if (std::abs(diff_percent) <= current.tolerance_percent) {
                std::cout << "✅ " << current.name << ": "
                         << current.time_us << " us (vs " << baseline_time << " us, "
                         << (diff_percent >= 0 ? "+" : "") << diff_percent << "%)\n";
            } else {
                std::cout << "❌ " << current.name << ": "
                         << current.time_us << " us (vs " << baseline_time << " us, "
                         << (diff_percent >= 0 ? "+" : "") << diff_percent << "%) - REGRESSION!\n";
                all_passed = false;
            }
        }

        std::cout << "=================================\n";
        return all_passed;
    }

private:
    std::unordered_map<std::string, double> baseline_;
};

// Subprocess로 벤치마크 실행하고 결과 파싱
std::vector<RegressionGuard::BenchmarkResult> runBenchmarkSubprocess(const std::string& benchmark_name) {
    std::vector<RegressionGuard::BenchmarkResult> results;

    // bazel run 명령어 구성 (JSON 대신 console 출력 사용)
    std::string command = "bazel run //bench:" + benchmark_name + " --copt=-O3 --copt=-DNDEBUG";

    // Subprocess 실행 (stderr도 캡처)
    std::array<char, 128> buffer;
    std::string result;
    std::string command_with_stderr = command + " 2>&1";  // stderr를 stdout으로 리다이렉트
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command_with_stderr.c_str(), "r"), pclose);

    if (!pipe) {
        std::cerr << "Failed to run command: " << command << std::endl;
        return results;
    }

    // 출력 읽기
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    // Google Benchmark console 출력 파싱
    // 두 가지 형식 처리:
    // 1. "FFTMicroBenchmark/SingleFFT512/iterations:10000       1.36 us"
    // 2. "FFTMicroBenchmark/SingleFFT512/iterations:10000: 1.36 us"
    std::istringstream iss(result);
    std::string line;

    while (std::getline(iss, line)) {
        // 벤치마크 라인 찾기 (시간 정보가 있는 라인)
        if (line.find(" us") != std::string::npos) {
            std::string bench_name;
            std::string time_str;

            // 형식 1: "FFTMicroBenchmark/SingleFFT512/iterations:10000       1.36 us"
            if (line.find(": ") == std::string::npos) {
                size_t us_pos = line.find(" us");
                if (us_pos != std::string::npos) {
                    std::string before_us = line.substr(0, us_pos);
                    // 마지막 공백부터 벤치마크 이름과 시간 분리
                    size_t last_space = before_us.find_last_of(" \t");
                    if (last_space != std::string::npos) {
                        bench_name = before_us.substr(0, last_space);
                        time_str = before_us.substr(last_space + 1);
                    }
                }
            }
            // 형식 2: "FFTMicroBenchmark/SingleFFT512/iterations:10000: 1.36 us"
            else {
                size_t colon_space_pos = line.find(": ");
                if (colon_space_pos != std::string::npos) {
                    bench_name = line.substr(0, colon_space_pos);
                    size_t us_pos = line.find(" us", colon_space_pos);
                    if (us_pos != std::string::npos) {
                        time_str = line.substr(colon_space_pos + 2, us_pos - colon_space_pos - 2);
                    }
                }
            }

            // iterations:XXXX 부분 제거
            size_t iter_pos = bench_name.find("/iterations:");
            if (iter_pos != std::string::npos) {
                bench_name = bench_name.substr(0, iter_pos);
            }

            if (!bench_name.empty() && !time_str.empty()) {
                try {
                    double time_us = std::stod(time_str);

                    RegressionGuard::BenchmarkResult res;
                    res.name = bench_name;
                    res.time_us = time_us;

                    // 특정 벤치마크에 대한 허용 오차 설정
                    if (bench_name.find("FFT1024") != std::string::npos) {
                        res.tolerance_percent = 5.0; // FFT는 더 엄격한 허용 오차
                    }

                    results.push_back(res);
                } catch (const std::exception&) {
                    // 파싱 실패 시 무시
                }
            }
        }
    }

    return results;
}

// 리그래션 테스트용 메인 함수
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <baseline_file> [--update-baseline]\n";
        return 1;
    }

    std::string baseline_file = argv[1];
    bool update_baseline = (argc >= 3 && std::string(argv[2]) == "--update-baseline");

    RegressionGuard guard;

    if (!update_baseline) {
        if (!guard.loadBaseline(baseline_file)) {
            std::cerr << "Failed to load baseline. Run with --update-baseline to create new baseline.\n";
            return 1;
        }
    }

    // 여러 벤치마크 실행
    std::vector<std::string> benchmark_names = {
        "micro_fft_benchmark",
        "micro_kernels_benchmark",
        "e2e_benchmark"
    };

    std::vector<RegressionGuard::BenchmarkResult> all_results;

    std::cout << "Running benchmarks for regression testing...\n";

    for (const auto& bench_name : benchmark_names) {
        std::cout << "Running " << bench_name << "...\n";
        auto results = runBenchmarkSubprocess(bench_name);
        all_results.insert(all_results.end(), results.begin(), results.end());
    }

    if (update_baseline) {
        // 기준선 업데이트
        if (guard.saveBaseline(baseline_file, all_results)) {
            std::cout << "Baseline updated successfully: " << baseline_file << std::endl;
        } else {
            std::cerr << "Failed to update baseline\n";
            return 1;
        }
    } else {
        // 리그래션 체크
        if (!guard.checkRegression(all_results)) {
            std::cerr << "Performance regression detected!\n";
            return 1;
        } else {
            std::cout << "All benchmarks passed regression check.\n";
        }
    }

    return 0;
}
