#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>
#include "dsp/ola/kernels.h"

using namespace dsp;

/**
 * 벤치마크용 테스트 데이터 생성 클래스
 */
class BenchmarkData {
public:
    static BenchmarkData& getInstance() {
        static BenchmarkData instance;
        return instance;
    }

    const std::vector<float>& getSrc(size_t size) {
        if (src_data_.find(size) == src_data_.end()) {
            generateData(size);
        }
        return src_data_[size];
    }

    const std::vector<float>& getWindow(size_t size) {
        if (window_data_.find(size) == window_data_.end()) {
            generateData(size);
        }
        return window_data_[size];
    }

    const std::vector<float>& getNorm(size_t size) {
        if (norm_data_.find(size) == norm_data_.end()) {
            generateData(size);
        }
        return norm_data_[size];
    }

    std::vector<float> getDst(size_t size) {
        return std::vector<float>(size, 0.1f);
    }

    std::vector<float> getAcc(size_t size) {
        if (acc_data_.find(size) == acc_data_.end()) {
            generateData(size);
        }
        return acc_data_[size]; // 복사본 반환 (수정되기 때문)
    }

private:
    BenchmarkData() = default;

    void generateData(size_t size) {
        std::random_device rd;
        std::mt19937 gen(12345); // 재현 가능한 결과를 위한 고정 시드
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        src_data_[size] = std::vector<float>(size);
        window_data_[size] = std::vector<float>(size);
        norm_data_[size] = std::vector<float>(size);
        acc_data_[size] = std::vector<float>(size);

        for (size_t i = 0; i < size; ++i) {
            src_data_[size][i] = dis(gen);
            window_data_[size][i] = 0.5f + 0.5f * std::cos(2.0f * M_PI * i / size);
            norm_data_[size][i] = 1.0f + std::abs(dis(gen));
            acc_data_[size][i] = dis(gen) * 10.0f;
        }
    }

    std::map<size_t, std::vector<float>> src_data_;
    std::map<size_t, std::vector<float>> window_data_;
    std::map<size_t, std::vector<float>> norm_data_;
    std::map<size_t, std::vector<float>> acc_data_;
};

//==============================================================================
// AXPY 벤치마크
//==============================================================================

static void BM_AXPY_Scalar(benchmark::State& state) {
    const size_t n = state.range(0);
    auto& data = BenchmarkData::getInstance();

    const auto& src = data.getSrc(n);
    const float g = 0.75f;

    for (auto _ : state) {
        auto dst = data.getDst(n);
        axpy_scalar(dst.data(), src.data(), g, n);
        benchmark::DoNotOptimize(dst.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2); // read src + read/write dst
}

static void BM_AXPY_Highway(benchmark::State& state) {
    const size_t n = state.range(0);
    auto& data = BenchmarkData::getInstance();

    const auto& src = data.getSrc(n);
    const float g = 0.75f;

    for (auto _ : state) {
        auto dst = data.getDst(n);
        axpy_hwy(dst.data(), src.data(), g, n);
        benchmark::DoNotOptimize(dst.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
}

static void BM_AXPY_Default(benchmark::State& state) {
    const size_t n = state.range(0);
    auto& data = BenchmarkData::getInstance();

    const auto& src = data.getSrc(n);
    const float g = 0.75f;

    for (auto _ : state) {
        auto dst = data.getDst(n);
        axpy(dst.data(), src.data(), g, n); // 런타임 디스패치
        benchmark::DoNotOptimize(dst.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
}

//==============================================================================
// AXPY Windowed 벤치마크
//==============================================================================

static void BM_AXPY_Windowed_Scalar(benchmark::State& state) {
    const size_t n = state.range(0);
    auto& data = BenchmarkData::getInstance();

    const auto& src = data.getSrc(n);
    const auto& win = data.getWindow(n);
    const float g = 1.25f;

    for (auto _ : state) {
        auto dst = data.getDst(n);
        axpy_windowed_scalar(dst.data(), src.data(), win.data(), g, n);
        benchmark::DoNotOptimize(dst.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 3); // read src, win + read/write dst
}

static void BM_AXPY_Windowed_Highway(benchmark::State& state) {
    const size_t n = state.range(0);
    auto& data = BenchmarkData::getInstance();

    const auto& src = data.getSrc(n);
    const auto& win = data.getWindow(n);
    const float g = 1.25f;

    for (auto _ : state) {
        auto dst = data.getDst(n);
        axpy_windowed_hwy(dst.data(), src.data(), win.data(), g, n);
        benchmark::DoNotOptimize(dst.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 3);
}

static void BM_AXPY_Windowed_Default(benchmark::State& state) {
    const size_t n = state.range(0);
    auto& data = BenchmarkData::getInstance();

    const auto& src = data.getSrc(n);
    const auto& win = data.getWindow(n);
    const float g = 1.25f;

    for (auto _ : state) {
        auto dst = data.getDst(n);
        axpy_windowed(dst.data(), src.data(), win.data(), g, n); // 런타임 디스패치
        benchmark::DoNotOptimize(dst.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 3);
}

//==============================================================================
// Normalize and Clear 벤치마크
//==============================================================================

static void BM_NormalizeAndClear_Scalar(benchmark::State& state) {
    const size_t n = state.range(0);
    auto& data = BenchmarkData::getInstance();

    const auto& norm = data.getNorm(n);
    const float eps = 1e-8f;

    for (auto _ : state) {
        auto acc = data.getAcc(n);
        std::vector<float> out(n);
        normalize_and_clear_scalar(out.data(), acc.data(), norm.data(), eps, n);
        benchmark::DoNotOptimize(out.data());
        benchmark::DoNotOptimize(acc.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 3); // read acc, norm + write out
}

static void BM_NormalizeAndClear_Highway(benchmark::State& state) {
    const size_t n = state.range(0);
    auto& data = BenchmarkData::getInstance();

    const auto& norm = data.getNorm(n);
    const float eps = 1e-8f;

    for (auto _ : state) {
        auto acc = data.getAcc(n);
        std::vector<float> out(n);
        normalize_and_clear_hwy(out.data(), acc.data(), norm.data(), eps, n);
        benchmark::DoNotOptimize(out.data());
        benchmark::DoNotOptimize(acc.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 3);
}

static void BM_NormalizeAndClear_Default(benchmark::State& state) {
    const size_t n = state.range(0);
    auto& data = BenchmarkData::getInstance();

    const auto& norm = data.getNorm(n);
    const float eps = 1e-8f;

    for (auto _ : state) {
        auto acc = data.getAcc(n);
        std::vector<float> out(n);
        normalize_and_clear(out.data(), acc.data(), norm.data(), eps, n); // 런타임 디스패치
        benchmark::DoNotOptimize(out.data());
        benchmark::DoNotOptimize(acc.data());
    }

    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 3);
}

//==============================================================================
// 벤치마크 등록
//==============================================================================

// 다양한 크기에서 테스트: 작은 크기부터 큰 크기까지
// SIMD lanes 경계 근처와 캐시 친화적인 크기들을 포함
static void CustomArguments(benchmark::internal::Benchmark* b) {
    // 작은 크기 (L1 캐시 내)
    b->Arg(16)->Arg(32)->Arg(64)->Arg(128);
    // 중간 크기 (L2 캐시 내)
    b->Arg(256)->Arg(512)->Arg(1024)->Arg(2048);
    // 큰 크기 (L3 캐시 / 메모리)
    b->Arg(4096)->Arg(8192)->Arg(16384)->Arg(32768);
}

// AXPY 벤치마크 등록
BENCHMARK(BM_AXPY_Scalar)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_AXPY_Highway)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_AXPY_Default)->Apply(CustomArguments)->UseRealTime();

// AXPY Windowed 벤치마크 등록
BENCHMARK(BM_AXPY_Windowed_Scalar)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_AXPY_Windowed_Highway)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_AXPY_Windowed_Default)->Apply(CustomArguments)->UseRealTime();

// Normalize and Clear 벤치마크 등록
BENCHMARK(BM_NormalizeAndClear_Scalar)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_NormalizeAndClear_Highway)->Apply(CustomArguments)->UseRealTime();
BENCHMARK(BM_NormalizeAndClear_Default)->Apply(CustomArguments)->UseRealTime();

BENCHMARK_MAIN();