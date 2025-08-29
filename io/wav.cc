// dr_wav 라이브러리 구현 포함
#define DR_WAV_IMPLEMENTATION
#include "wav.h"

#include <algorithm>
#include <cstring>

WavReader::WavReader() : wav_(nullptr), is_open_(false) {}

WavReader::~WavReader() {
    close();
}

bool WavReader::open(const std::string& filename) {
    close();

    wav_ = new drwav();
    if (drwav_init_file(wav_, filename.c_str(), nullptr)) {
        is_open_ = true;
        return true;
    } else {
        delete wav_;
        wav_ = nullptr;
        return false;
    }
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

    std::vector<float> data(total_samples);

    // 파일 포인터를 처음으로 되돌림
    drwav_seek_to_pcm_frame(wav_, 0);

    size_t frames_read;
    if (read(data.data(), total_frames, &frames_read)) {
        data.resize(frames_read * channels);
        return data;
    }

    return {};
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

WavWriter::WavWriter() : wav_(nullptr), is_open_(false) {}

WavWriter::~WavWriter() {
    close();
}

bool WavWriter::open(const std::string& filename,
                     uint32_t channels,
                     uint32_t sample_rate,
                     uint32_t bits_per_sample) {
    close();

    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = channels;
    format.sampleRate = sample_rate;
    format.bitsPerSample = bits_per_sample;

    wav_ = new drwav();
    if (drwav_init_file_write(wav_, filename.c_str(), &format, nullptr)) {
        is_open_ = true;
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

    // float 데이터를 int16으로 변환 (dr_wav는 기본적으로 16비트 정수로 쓰기 지원)
    uint32_t channels = wav_->channels;
    size_t total_samples = frames_to_write * channels;

    std::vector<drwav_int16> int16_buffer(total_samples);

    // float (-1.0 ~ 1.0) to int16 변환
    for (size_t i = 0; i < total_samples; ++i) {
        float sample = buffer[i];
        // 클리핑 적용
        sample = std::max(-1.0f, std::min(1.0f, sample));
        int16_buffer[i] = static_cast<drwav_int16>(sample * 32767.0f);
    }

    drwav_uint64 frames_written_actual = drwav_write_pcm_frames(wav_, frames_to_write, int16_buffer.data());

    if (frames_written) {
        *frames_written = static_cast<size_t>(frames_written_actual);
    }

    return frames_written_actual == frames_to_write;
}

bool WavWriter::is_open() const {
    return is_open_;
}
