// dr_wav 라이브러리 구현 포함
#define DR_WAV_IMPLEMENTATION
#include "wav.h"

#include <algorithm>
#include <cstring>
#include <cmath>

WavReader::WavReader() : wav_(nullptr), is_open_(false) {}

WavReader::~WavReader() {
    close();
}

bool WavReader::open(const std::string& filename) {
    close();

    wav_ = new drwav();
    if (!drwav_init_file(wav_, filename.c_str(), nullptr)) {
        delete wav_;
        wav_ = nullptr;
        return false;
    }

    // 포맷 가드: 채널(1/2), 비트 심도(16/24/32), 포맷(PCM 또는 IEEE float 32)
    const bool channels_ok = (wav_->channels == 1 || wav_->channels == 2);
    const bool is_pcm = (wav_->translatedFormatTag == DR_WAVE_FORMAT_PCM);
    const bool is_f32 = (wav_->translatedFormatTag == DR_WAVE_FORMAT_IEEE_FLOAT && wav_->bitsPerSample == 32);
    const bool bps_ok = (wav_->bitsPerSample == 16 || wav_->bitsPerSample == 24 || wav_->bitsPerSample == 32);

    if (!channels_ok || !bps_ok || !(is_pcm || is_f32)) {
        drwav_uninit(wav_);
        delete wav_;
        wav_ = nullptr;
        return false;
    }

    is_open_ = true;
    return true;
}

void WavReader::close() {
    if (is_open_ && wav_) {
        drwav_uninit(wav_);
        delete wav_;
        wav_ = nullptr;
        is_open_ = false;
    }
}

bool WavReader::read(float* buffer, size_t frames_to_read, size_t* frames_read) {
    if (!is_open_ || !wav_) {
        return false;
    }

    drwav_uint64 frames_read_actual = drwav_read_pcm_frames_f32(wav_, frames_to_read, buffer);

    if (frames_read) {
        *frames_read = static_cast<size_t>(frames_read_actual);
    }

    return frames_read_actual > 0 || frames_to_read == 0;
}

std::vector<float> WavReader::read_all() {
    if (!is_open_ || !wav_) {
        return {};
    }

    uint64_t total_frames = get_total_frames();
    uint32_t channels = get_channels();
    size_t total_samples = static_cast<size_t>(total_frames * channels);

    if (total_samples == 0) {
        return {};
    }

    std::vector<float> data(total_samples);

    // 파일이 이미 열려있으므로 처음부터 읽음
    drwav_uint64 frames_read_actual = drwav_read_pcm_frames_f32(wav_, total_frames, data.data());
    size_t frames_read = static_cast<size_t>(frames_read_actual);

    if (frames_read < total_frames) {
        // 읽은 프레임 수만큼 데이터 크기 조정
        data.resize(frames_read * channels);
    }

    return data;
}

uint32_t WavReader::get_channels() const {
    return is_open_ && wav_ ? wav_->channels : 0;
}

uint32_t WavReader::get_sample_rate() const {
    return is_open_ && wav_ ? wav_->sampleRate : 0;
}

uint64_t WavReader::get_total_frames() const {
    return is_open_ && wav_ ? wav_->totalPCMFrameCount : 0;
}

uint32_t WavReader::get_bits_per_sample() const {
    return is_open_ && wav_ ? wav_->bitsPerSample : 0;
}

bool WavReader::is_open() const {
    return is_open_;
}

WavWriter::WavWriter() : wav_(nullptr), is_open_(false), bits_per_sample_(16), is_float_(false) {}

WavWriter::~WavWriter() {
    close();
}

bool WavWriter::open(const std::string& filename,
                     uint32_t channels,
                     uint32_t sample_rate,
                     uint32_t bits_per_sample,
                     bool float_format) {
    close();

    // 채널 가드
    if (channels == 0 || (channels != 1 && channels != 2)) {
        return false;
    }

    // 지원되는 비트 심도 확인
    if (bits_per_sample != 16 && bits_per_sample != 24 && bits_per_sample != 32) {
        return false;
    }

    drwav_data_format format;
    format.container = drwav_container_riff;
    // float32는 IEEE_FLOAT로, 그 외는 PCM
    format.format = (float_format && bits_per_sample == 32) ? DR_WAVE_FORMAT_IEEE_FLOAT : DR_WAVE_FORMAT_PCM;
    format.channels = channels;
    format.sampleRate = sample_rate;
    format.bitsPerSample = bits_per_sample;

    wav_ = new drwav();
    if (drwav_init_file_write(wav_, filename.c_str(), &format, nullptr)) {
        is_open_ = true;
        bits_per_sample_ = bits_per_sample;
        is_float_ = (format.format == DR_WAVE_FORMAT_IEEE_FLOAT);
        return true;
    } else {
        delete wav_;
        wav_ = nullptr;
        return false;
    }
}

void WavWriter::close() {
    if (is_open_ && wav_) {
        drwav_uninit(wav_);
        delete wav_;
        wav_ = nullptr;
        is_open_ = false;
    }
}

bool WavWriter::write(const float* buffer, size_t frames_to_write, size_t* frames_written) {
    if (!is_open_ || !wav_) {
        return false;
    }

    drwav_uint64 frames_written_actual = 0;

    const uint32_t channels = wav_->channels;
    const size_t total_samples = frames_to_write * channels;

    if (is_float_) {
        // 포맷이 IEEE float32인 경우 그대로 기록
        frames_written_actual = drwav_write_pcm_frames(wav_, frames_to_write, buffer);
    } else {
        // PCM 정수 포맷: 비트 심도에 따른 변환
        switch (bits_per_sample_) {
            case 16: {
                std::vector<drwav_int16> tmp(total_samples);
                drwav_f32_to_s16(tmp.data(), buffer, total_samples);
                frames_written_actual = drwav_write_pcm_frames(wav_, frames_to_write, tmp.data());
                break;
            }
            case 24: {
                // 24비트는 3바이트 패킹 필요
                std::vector<drwav_uint8> tmp(total_samples * 3);
                // float -> signed 24-bit little-endian
                for (size_t i = 0; i < total_samples; ++i) {
                    float x = std::max(-1.0f, std::min(1.0f, buffer[i]));
                    // 스케일 및 라운딩
                    int32_t s = static_cast<int32_t>(std::lrintf(x * 8388607.0f));
                    if (s >  8388607)  s =  8388607;
                    if (s < -8388608) s = -8388608;
                    // 리틀엔디안 3바이트 패킹
                    tmp[i * 3 + 0] = static_cast<drwav_uint8>(s & 0xFF);
                    tmp[i * 3 + 1] = static_cast<drwav_uint8>((s >> 8) & 0xFF);
                    tmp[i * 3 + 2] = static_cast<drwav_uint8>((s >> 16) & 0xFF);
                }
                frames_written_actual = drwav_write_pcm_frames(wav_, frames_to_write, tmp.data());
                break;
            }
            case 32: {
                std::vector<drwav_int32> tmp(total_samples);
                drwav_f32_to_s32(tmp.data(), buffer, total_samples);
                frames_written_actual = drwav_write_pcm_frames(wav_, frames_to_write, tmp.data());
                break;
            }
            default:
                return false;
        }
    }

    if (frames_written) {
        *frames_written = static_cast<size_t>(frames_written_actual);
    }

    return frames_written_actual == frames_to_write;
}

bool WavWriter::is_open() const {
    return is_open_;
}

template<typename T>
void WavWriter::convert_float_to_int(const float* input, T* output, size_t sample_count, float scale) {
    for (size_t i = 0; i < sample_count; ++i) {
        float clamped = std::max(-1.0f, std::min(1.0f, input[i]));
        output[i] = static_cast<T>(clamped * scale);
    }
}
