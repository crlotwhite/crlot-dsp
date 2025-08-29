#ifndef WAV_H
#define WAV_H

#include <string>
#include <vector>
#include <cstdint>

// dr_wav 라이브러리 헤더 포함
#include <dr_wav.h>

class WavReader {
public:
    WavReader();
    ~WavReader();

    // 파일 열기
    bool open(const std::string& filename);

    // 파일 닫기
    void close();

    // 오디오 데이터 읽기 (float32)
    bool read(float* buffer, size_t frames_to_read, size_t* frames_read = nullptr);

    // 전체 오디오 데이터를 한 번에 읽기
    std::vector<float> read_all();

    // 오디오 정보 얻기
    uint32_t get_channels() const;
    uint32_t get_sample_rate() const;
    uint64_t get_total_frames() const;
    uint32_t get_bits_per_sample() const;

    // 파일이 열려있는지 확인
    bool is_open() const;

private:
    drwav* wav_;
    bool is_open_;
};

class WavWriter {
public:
    WavWriter();
    ~WavWriter();

    // 파일 생성 (다양한 포맷 지원)
    bool open(const std::string& filename,
              uint32_t channels,
              uint32_t sample_rate,
              uint32_t bits_per_sample = 16,
              bool float_format = false);

    // 파일 닫기
    void close();

    // 오디오 데이터 쓰기 (float32 입력)
    bool write(const float* buffer, size_t frames_to_write, size_t* frames_written = nullptr);

    // 파일이 열려있는지 확인
    bool is_open() const;

private:
    drwav* wav_;
    bool is_open_;
    uint32_t bits_per_sample_;
    bool is_float_;
    
    // 다양한 비트 심도에 따른 데이터 변환
    template<typename T>
    void convert_float_to_int(const float* input, T* output, size_t sample_count, float scale);
};

#endif // WAV_H
