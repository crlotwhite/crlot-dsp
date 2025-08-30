#pragma once

#include <cstddef>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>

namespace dsp {

/**
 * 윈도우 함수 타입
 */
enum class WindowType {
    HANN,           // Hann window
    HAMMING,        // Hamming window
    BLACKMAN,       // Blackman window
    RECT,           // Rectangular window
    BLACKMAN_HARRIS // Blackman-Harris window (추후 구현)
};

/**
 * 윈도우 정규화 타입
 */
enum class NormalizationType {
    NONE,           // 정규화 없음
    SUM_TO_ONE,     // 합이 1이 되도록 정규화
    L2_NORM,        // L2 노름이 1이 되도록 정규화 (분석용)
    OLA_UNITY_GAIN, // OLA 합성 시 이득이 1이 되도록 정규화 (합성용)
    OLA_SUM_WSQ     // 분석창이 이미 곱해진 프레임용 정규화 (∑w²)
};

/**
 * 캐시 정렬된 윈도우 데이터 구조
 * AVX2/NEON 벡터화를 위한 32-byte 정렬
 */
struct alignas(32) WindowData {
    alignas(32) float* data;    // 윈도우 데이터 (32-byte 정렬)
    size_t size;                // 윈도우 크기
    WindowType type;            // 윈도우 타입
    bool periodic;              // 주기적 윈도우 여부 (FFT용)
    NormalizationType norm;     // 정규화 타입

    WindowData() : data(nullptr), size(0), type(WindowType::HANN),
                   periodic(false), norm(NormalizationType::NONE) {}
    ~WindowData();

    // 복사 생성자와 대입 연산자 삭제 (RAII)
    WindowData(const WindowData&) = delete;
    WindowData& operator=(const WindowData&) = delete;

    // 이동 생성자와 이동 대입 연산자
    WindowData(WindowData&& other) noexcept;
    WindowData& operator=(WindowData&& other) noexcept;
};

/**
 * WindowLUT - 윈도우 함수 Look-Up Table 클래스
 *
 * 주요 기능:
 * - Hann, Hamming, Blackman, Rect 윈도우 함수 지원
 * - periodic 옵션: FFT용(DFT 주기성) vs 분석창(실신호)
 * - 정규화 옵션: sum=1, L2=1, OLA 보정 지원
 * - 길이별 캐시를 통한 성능 최적화
 * - 메모리 정렬 최적화 (32-byte AVX2/NEON 지원)
 * - 스레드 안전성 보장 (읽기 작업)
 * - 임의 길이 N 지원 (2^k 및 비2^k)
 *
 * 정규화 규칙:
 * - 분석만 쓸 때: L2=1도 OK
 * - 합성 고려 시: ∑ w²[n]·hop에 맞춘 보정 필수
 * - OLA 합성 시 이득 1이 되도록 창의 제곱합/중첩 보정
 *
 * 멀티스레드 사용 가이드:
 * - GetWindow(): 스레드 안전 (mutex 보호)
 * - clearCache(): 테스트용, 운영 환경에서 주의 필요
 * - 캐시된 포인터는 clearCache() 호출 전까지 유효
 * - 운영 환경에서는 캐시 정리를 피하거나 shared_ptr 핸들 사용 권장
 */
class WindowLUT {
public:
    /**
     * WindowLUT 생성자
     *
     * @param nfft 윈도우 크기 (FFT 크기)
     * @param type 윈도우 함수 타입
     * @param periodic FFT용(true) vs 분석창용(false)
     * @param norm 정규화 타입
     */
    WindowLUT(size_t nfft, WindowType type, bool periodic = false,
              NormalizationType norm = NormalizationType::NONE);

    /**
     * 기본 생성자 (캐시 전용)
     */
    WindowLUT();

    /**
     * 소멸자
     */
    ~WindowLUT() = default;

    /**
     * 윈도우 함수 데이터 반환
     *
     * @return 윈도우 함수 데이터 포인터 (크기 nfft)
     */
    const float* data() const;

    /**
     * 윈도우 크기 반환
     */
    size_t size() const { return window_data_ ? window_data_->size : 0; }

    /**
     * 윈도우 타입 반환
     */
    WindowType type() const { return window_data_ ? window_data_->type : WindowType::HANN; }

    /**
     * periodic 여부 반환
     */
    bool periodic() const { return window_data_ ? window_data_->periodic : false; }

    /**
     * 정규화 타입 반환
     */
    NormalizationType normalization() const {
        return window_data_ ? window_data_->norm : NormalizationType::NONE;
    }

    /**
     * 지정된 타입과 크기의 윈도우 함수 반환 (캐시 기능)
     *
     * ⚠️  스레드 안전성 개선:
     * - shared_ptr 기반으로 메모리 안전성 보장
     * - clearCache() 호출 시에도 기존 참조는 유효 유지
     * - Generation 기반 캐시 무효화 지원
     *
     * @param type 윈도우 함수 타입
     * @param N 윈도우 크기
     * @param periodic FFT용(true) vs 분석창용(false)
     * @param norm 정규화 타입
     * @return 윈도우 함수 데이터의 안전한 참조 (크기 N)
     */
    std::shared_ptr<const float> GetWindowSafe(WindowType type, size_t N, bool periodic = false,
                                              NormalizationType norm = NormalizationType::NONE);

    /**
     * 하위 호환성을 위한 기존 API (deprecated)
     *
     * ⚠️  주의: 멀티스레드 환경에서 clearCache() 호출 시 UAF 위험
     * 새로운 코드에서는 GetWindowSafe() 사용 권장
     */
    [[deprecated("Use GetWindowSafe() for thread safety")]]
    const float* GetWindow(WindowType type, size_t N, bool periodic = false,
                          NormalizationType norm = NormalizationType::NONE);

    /**
     * 싱글톤 인스턴스 반환 (하위 호환성)
     */
    static WindowLUT& getInstance();

    /**
     * 캐시된 윈도우 개수 반환
     */
    size_t getCacheSize() const;

    /**
     * 캐시 초기화 (개선된 안전성)
     *
     * ✅ 개선된 안전성:
     * - Generation 기반 무효화로 기존 참조는 계속 유효
     * - shared_ptr 기반으로 메모리 안전성 보장
     * - 멀티스레드 환경에서 안전하게 사용 가능
     *
     * @param force_immediate true시 즉시 메모리 해제 (테스트용)
     */
    void clearCache(bool force_immediate = false);

    /**
     * 현재 캐시 generation 반환
     */
    uint64_t getCurrentGeneration() const;

    /**
     * 윈도우 함수의 합 계산 (정규화 검증용)
     */
    static double calculateSum(const float* window, size_t N);

    /**
     * 윈도우 함수의 제곱합 계산 (정규화 검증용)
     */
    static double calculateSumOfSquares(const float* window, size_t N);

    /**
     * 두 윈도우 함수 간의 RMS 오차 계산
     */
    static double calculateRMSError(const float* window1, const float* window2, size_t N);

private:
    // 복사 생성자와 대입 연산자 삭제
    WindowLUT(const WindowLUT&) = delete;
    WindowLUT& operator=(const WindowLUT&) = delete;

    /**
     * 캐시 키 생성 (하위 호환성)
     */
    uint64_t makeCacheKey(WindowType type, size_t N) const;

    /**
     * 윈도우 함수 생성 (확장된 버전)
     */
    std::unique_ptr<WindowData> createWindow(WindowType type, size_t N,
                                           bool periodic = false,
                                           NormalizationType norm = NormalizationType::NONE) const;

    /**
     * Hann 윈도우 생성
     */
    void generateHannWindow(float* data, size_t N, bool periodic = false) const;

    /**
     * Hamming 윈도우 생성
     */
    void generateHammingWindow(float* data, size_t N, bool periodic = false) const;

    /**
     * Blackman 윈도우 생성
     */
    void generateBlackmanWindow(float* data, size_t N, bool periodic = false) const;

    /**
     * Rectangular 윈도우 생성
     */
    void generateRectWindow(float* data, size_t N) const;

    /**
     * 윈도우 정규화 적용
     */
    void applyNormalization(float* data, size_t N, NormalizationType norm,
                           size_t hop_size = 0) const;

    /**
     * OLA 이득 보정 계수 계산
     */
    double calculateOLAGain(const float* window, size_t N, size_t hop_size) const;

    /**
     * 정렬된 메모리 할당
     */
    float* allocateAlignedMemory(size_t N) const;

    /**
     * 정렬된 메모리 해제
     */
    void freeAlignedMemory(float* ptr) const;

    /**
     * 캐시 키 생성 (확장된 버전)
     */
    uint64_t makeCacheKeyExtended(WindowType type, size_t N, bool periodic,
                                 NormalizationType norm) const;

private:
    std::unique_ptr<WindowData> window_data_;  // 인스턴스 윈도우 데이터

    // 안전한 캐시 엔트리 구조
    struct SafeCacheEntry {
        std::shared_ptr<WindowData> data;
        uint64_t generation;

        // 기본 생성자 (std::unordered_map 호환성)
        SafeCacheEntry() : data(nullptr), generation(0) {}

        SafeCacheEntry(std::shared_ptr<WindowData> d, uint64_t gen)
            : data(std::move(d)), generation(gen) {}
    };

    // 정적 캐시 (개선된 안전성)
    static std::mutex cache_mutex_;
    static std::unordered_map<uint64_t, SafeCacheEntry> safe_cache_;
    static std::atomic<uint64_t> current_generation_;

    // 하위 호환성을 위한 기존 캐시 (deprecated)
    static std::unordered_map<uint64_t, std::unique_ptr<WindowData>> legacy_cache_;
};

} // namespace dsp