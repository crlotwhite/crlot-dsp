#include "WindowLUT.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace dsp {

// WindowData 구현
WindowData::~WindowData() {
    if (data) {
        std::free(data);
        data = nullptr;
    }
}

WindowData::WindowData(WindowData&& other) noexcept
    : data(other.data), size(other.size), type(other.type),
      periodic(other.periodic), norm(other.norm) {
    other.data = nullptr;
    other.size = 0;
}

WindowData& WindowData::operator=(WindowData&& other) noexcept {
    if (this != &other) {
        if (data) {
            std::free(data);
        }
        data = other.data;
        size = other.size;
        type = other.type;
        periodic = other.periodic;
        norm = other.norm;

        other.data = nullptr;
        other.size = 0;
    }
    return *this;
}

// 정적 멤버 초기화
std::mutex WindowLUT::cache_mutex_;
std::unordered_map<uint64_t, std::unique_ptr<WindowData>> WindowLUT::cache_;

// WindowLUT 구현
WindowLUT::WindowLUT(size_t nfft, WindowType type, bool periodic, NormalizationType norm) {
    if (nfft == 0) {
        throw std::invalid_argument("Window size must be greater than 0");
    }

    window_data_ = createWindow(type, nfft, periodic, norm);
}

WindowLUT::WindowLUT() : window_data_(nullptr) {
    // 캐시 전용 생성자
}

const float* WindowLUT::data() const {
    if (!window_data_) {
        throw std::runtime_error("Window data not initialized");
    }
    return window_data_->data;
}

WindowLUT& WindowLUT::getInstance() {
    static WindowLUT instance;
    return instance;
}

const float* WindowLUT::GetWindow(WindowType type, size_t N, bool periodic,
                                 NormalizationType norm) {
    if (N == 0) {
        throw std::invalid_argument("Window size must be greater than 0");
    }

    uint64_t key = makeCacheKeyExtended(type, N, periodic, norm);

    // 전체 과정을 하나의 락으로 보호하여 경쟁 조건 방지
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // 캐시에서 검색
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second->data;
    }

    // 캐시에 없으면 새로 생성
    auto window_data = createWindow(type, N, periodic, norm);
    const float* result = window_data->data;

    // 캐시에 저장
    cache_[key] = std::move(window_data);

    return result;
}

size_t WindowLUT::getCacheSize() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.size();
}

void WindowLUT::clearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
}

double WindowLUT::calculateSum(const float* window, size_t N) {
    if (!window || N == 0) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum += static_cast<double>(window[i]);
    }
    return sum;
}

double WindowLUT::calculateSumOfSquares(const float* window, size_t N) {
    if (!window || N == 0) {
        return 0.0;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double val = static_cast<double>(window[i]);
        sum_sq += val * val;
    }
    return sum_sq;
}

double WindowLUT::calculateRMSError(const float* window1, const float* window2, size_t N) {
    if (!window1 || !window2 || N == 0) {
        return 0.0;
    }

    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double diff = static_cast<double>(window1[i]) - static_cast<double>(window2[i]);
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / static_cast<double>(N));
}

uint64_t WindowLUT::makeCacheKey(WindowType type, size_t N) const {
    // 상위 32비트: 윈도우 타입, 하위 32비트: 크기
    uint64_t type_bits = static_cast<uint64_t>(type) << 32;
    uint64_t size_bits = static_cast<uint64_t>(N) & 0xFFFFFFFF;
    return type_bits | size_bits;
}

std::unique_ptr<WindowData> WindowLUT::createWindow(WindowType type, size_t N,
                                                   bool periodic, NormalizationType norm) const {
    auto window_data = std::make_unique<WindowData>();
    window_data->data = allocateAlignedMemory(N);
    window_data->size = N;
    window_data->type = type;
    window_data->periodic = periodic;
    window_data->norm = norm;

    if (!window_data->data) {
        throw std::bad_alloc();
    }

    switch (type) {
        case WindowType::HANN:
            generateHannWindow(window_data->data, N, periodic);
            break;
        case WindowType::HAMMING:
            generateHammingWindow(window_data->data, N, periodic);
            break;
        case WindowType::BLACKMAN:
            generateBlackmanWindow(window_data->data, N, periodic);
            break;
        case WindowType::RECT:
            generateRectWindow(window_data->data, N);
            break;
        case WindowType::BLACKMAN_HARRIS:
            throw std::invalid_argument("Blackman-Harris window not yet implemented");
        default:
            freeAlignedMemory(window_data->data);
            throw std::invalid_argument("Unknown window type");
    }

    // 정규화 적용
    if (norm != NormalizationType::NONE) {
        applyNormalization(window_data->data, N, norm);
    }

    return window_data;
}

void WindowLUT::generateHannWindow(float* data, size_t N, bool periodic) const {
    if (N == 1) {
        data[0] = 1.0f;
        return;
    }

    const double pi = M_PI;
    const double denominator = periodic ? static_cast<double>(N) : static_cast<double>(N - 1);
    const double factor = 2.0 * pi / denominator;

    for (size_t i = 0; i < N; ++i) {
        double angle = factor * static_cast<double>(i);
        data[i] = static_cast<float>(0.5 * (1.0 - std::cos(angle)));
    }
}

void WindowLUT::generateHammingWindow(float* data, size_t N, bool periodic) const {
    if (N == 1) {
        data[0] = 1.0f;
        return;
    }

    const double pi = M_PI;
    const double denominator = periodic ? static_cast<double>(N) : static_cast<double>(N - 1);
    const double factor = 2.0 * pi / denominator;
    const double alpha = 0.54;
    const double beta = 0.46;

    for (size_t i = 0; i < N; ++i) {
        double angle = factor * static_cast<double>(i);
        data[i] = static_cast<float>(alpha - beta * std::cos(angle));
    }
}

void WindowLUT::generateBlackmanWindow(float* data, size_t N, bool periodic) const {
    if (N == 1) {
        data[0] = 1.0f;
        return;
    }

    const double pi = M_PI;
    const double denominator = periodic ? static_cast<double>(N) : static_cast<double>(N - 1);
    const double factor = 2.0 * pi / denominator;
    const double a0 = 0.42;
    const double a1 = 0.5;
    const double a2 = 0.08;

    for (size_t i = 0; i < N; ++i) {
        double angle = factor * static_cast<double>(i);
        double cos1 = std::cos(angle);
        double cos2 = std::cos(2.0 * angle);
        data[i] = static_cast<float>(a0 - a1 * cos1 + a2 * cos2);
    }
}

void WindowLUT::generateRectWindow(float* data, size_t N) const {
    for (size_t i = 0; i < N; ++i) {
        data[i] = 1.0f;
    }
}

void WindowLUT::applyNormalization(float* data, size_t N, NormalizationType norm,
                                  size_t hop_size) const {
    switch (norm) {
        case NormalizationType::NONE:
            break;

        case NormalizationType::SUM_TO_ONE: {
            double sum = calculateSum(data, N);
            if (sum > 0.0) {
                float scale = static_cast<float>(1.0 / sum);
                for (size_t i = 0; i < N; ++i) {
                    data[i] *= scale;
                }
            }
            break;
        }

        case NormalizationType::L2_NORM: {
            double sum_sq = calculateSumOfSquares(data, N);
            if (sum_sq > 0.0) {
                float scale = static_cast<float>(1.0 / std::sqrt(sum_sq));
                for (size_t i = 0; i < N; ++i) {
                    data[i] *= scale;
                }
            }
            break;
        }

        case NormalizationType::OLA_UNITY_GAIN: {
            // OLA 정규화는 hop_size 정보가 필요하므로 기본적으로 L2 정규화 적용
            // 실제 OLA 정규화는 사용 시점에서 hop_size와 함께 적용해야 함
            double sum_sq = calculateSumOfSquares(data, N);
            if (sum_sq > 0.0) {
                float scale = static_cast<float>(1.0 / std::sqrt(sum_sq));
                for (size_t i = 0; i < N; ++i) {
                    data[i] *= scale;
                }
            }
            break;
        }

        case NormalizationType::OLA_SUM_WSQ: {
            // 분석창이 이미 곱해진 프레임용 정규화
            // ∑ w_s(t-kH)·w_a(t-kH) 형태의 정규화
            // hop_size가 제공되지 않으면 L2 정규화로 폴백
            if (hop_size > 0) {
                // hop 기반 중첩 정규화 (향후 확장 가능)
                // 현재는 제곱합 기반으로 구현
                double sum_sq = calculateSumOfSquares(data, N);
                if (sum_sq > 0.0) {
                    // hop 비율을 고려한 스케일링
                    double hop_factor = static_cast<double>(N) / static_cast<double>(hop_size);
                    double scale = 1.0 / (std::sqrt(sum_sq) * std::sqrt(hop_factor));
                    float scale_f = static_cast<float>(scale);
                    for (size_t i = 0; i < N; ++i) {
                        data[i] *= scale_f;
                    }
                }
            } else {
                // hop_size가 없으면 L2 정규화로 폴백
                double sum_sq = calculateSumOfSquares(data, N);
                if (sum_sq > 0.0) {
                    float scale = static_cast<float>(1.0 / std::sqrt(sum_sq));
                    for (size_t i = 0; i < N; ++i) {
                        data[i] *= scale;
                    }
                }
            }
            break;
        }
    }
}

double WindowLUT::calculateOLAGain(const float* window, size_t N, size_t hop_size) const {
    // OLA 이득 계산: 윈도우 제곱의 중첩 합
    double gain = 0.0;

    // 중첩되는 윈도우 개수 계산
    size_t num_overlaps = (N + hop_size - 1) / hop_size;

    for (size_t i = 0; i < N; ++i) {
        double overlap_sum = 0.0;

        // 현재 샘플에서 중첩되는 모든 윈도우의 기여도 합산
        for (size_t j = 0; j < num_overlaps; ++j) {
            int sample_idx = static_cast<int>(i) - static_cast<int>(j * hop_size);
            if (sample_idx >= 0 && sample_idx < static_cast<int>(N)) {
                double w = static_cast<double>(window[sample_idx]);
                overlap_sum += w * w;
            }
        }

        gain = std::max(gain, overlap_sum);
    }

    return gain;
}

float* WindowLUT::allocateAlignedMemory(size_t N) const {
    // 32-byte 정렬된 메모리 할당 (AVX2/NEON 벡터화 지원)
    const size_t alignment = 32;
    const size_t size = N * sizeof(float);

    void* ptr = nullptr;
    int result = posix_memalign(&ptr, alignment, size);

    if (result != 0 || ptr == nullptr) {
        return nullptr;
    }

    // 메모리 초기화
    std::memset(ptr, 0, size);

    return static_cast<float*>(ptr);
}

uint64_t WindowLUT::makeCacheKeyExtended(WindowType type, size_t N, bool periodic,
                                        NormalizationType norm) const {
    // 64비트 키: [타입:8][주기:1][정규화:3][크기:52]
    uint64_t type_bits = static_cast<uint64_t>(type) << 56;
    uint64_t periodic_bits = (periodic ? 1ULL : 0ULL) << 55;
    uint64_t norm_bits = static_cast<uint64_t>(norm) << 52;
    uint64_t size_bits = static_cast<uint64_t>(N) & 0xFFFFFFFFFFFFFULL;

    return type_bits | periodic_bits | norm_bits | size_bits;
}

void WindowLUT::freeAlignedMemory(float* ptr) const {
    if (ptr) {
        std::free(ptr);
    }
}

} // namespace dsp