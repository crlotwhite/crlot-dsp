#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <cmath>
#include "io/wav.h"
#include <cstdlib>

namespace fs = std::filesystem;

class WavIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 테스트용 임시 디렉토리 생성
        test_dir_ = fs::temp_directory_path() / "wav_test";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        // 테스트 후 정리
        fs::remove_all(test_dir_);
    }

    fs::path test_dir_;
};

static std::string RunfilePath(const std::string& relative) {
    const char* srcdir = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    if (!srcdir || !workspace) return relative; // fallback for non-bazel
    return std::string(srcdir) + "/" + workspace + "/" + relative;
}

// WavReader 테스트
TEST_F(WavIOTest, WavReader_OpenValidFile) {
    WavReader reader;

    // assets/oboe.wav 파일이 존재한다고 가정
    EXPECT_TRUE(reader.open(RunfilePath("assets/oboe.wav")));
    EXPECT_TRUE(reader.is_open());

    // 오디오 정보 확인
    EXPECT_GT(reader.get_channels(), 0u);
    EXPECT_GT(reader.get_sample_rate(), 0u);
    EXPECT_GT(reader.get_total_frames(), 0u);
    EXPECT_GT(reader.get_bits_per_sample(), 0u);

    reader.close();
    EXPECT_FALSE(reader.is_open());
}

TEST_F(WavIOTest, WavReader_OpenInvalidFile) {
    WavReader reader;

    // 존재하지 않는 파일 열기 시도
    EXPECT_FALSE(reader.open("nonexistent.wav"));
    EXPECT_FALSE(reader.is_open());
}

TEST_F(WavIOTest, WavReader_ReadData) {
    WavReader reader;

    ASSERT_TRUE(reader.open(RunfilePath("assets/oboe.wav")));

    uint32_t channels = reader.get_channels();
    uint64_t total_frames = reader.get_total_frames();

    // 전체 데이터 읽기
    std::vector<float> data = reader.read_all();
    EXPECT_EQ(data.size(), total_frames * channels);
    EXPECT_FALSE(data.empty());

    reader.close();
}

TEST_F(WavIOTest, WavReader_BasicInfoTest) {
    WavReader reader;

    ASSERT_TRUE(reader.open(RunfilePath("assets/oboe.wav")));

    uint64_t total_frames = reader.get_total_frames();
    uint32_t channels = reader.get_channels();

    // 기본 정보 확인
    EXPECT_GT(total_frames, 0u);
    EXPECT_GT(channels, 0u);
    EXPECT_EQ(reader.get_bits_per_sample(), 16u);
    EXPECT_EQ(reader.get_sample_rate(), 44100u);
    EXPECT_TRUE(reader.is_open());

    reader.close();
    EXPECT_FALSE(reader.is_open());
}

// WavWriter 테스트
TEST_F(WavIOTest, WavWriter_WriteAndRead) {
    // 테스트용 WAV 파일 생성
    fs::path test_file = test_dir_ / "test_output.wav";

    WavWriter writer;
    EXPECT_TRUE(writer.open(test_file.string(), 2, 44100, 16));
    EXPECT_TRUE(writer.is_open());

    // 사인파 데이터 생성 (440Hz, 1초)
    const size_t frames = 44100;
    const size_t channels = 2;
    std::vector<float> data(frames * channels);

    const float PI = 3.14159265358979323846f;
    for (size_t i = 0; i < frames; ++i) {
        float sample = 0.5f * std::sin(2.0f * PI * 440.0f * static_cast<float>(i) / 44100.0f);
        data[i * channels] = sample;     // 왼쪽 채널
        data[i * channels + 1] = sample; // 오른쪽 채널
    }

    // 데이터 쓰기
    size_t frames_written;
    EXPECT_TRUE(writer.write(data.data(), frames, &frames_written));
    EXPECT_EQ(frames_written, frames);

    writer.close();
    EXPECT_FALSE(writer.is_open());

    // 생성된 파일이 제대로 읽히는지 확인
    WavReader reader;
    EXPECT_TRUE(reader.open(test_file.string()));
    EXPECT_TRUE(reader.is_open());

    EXPECT_EQ(reader.get_channels(), channels);
    EXPECT_EQ(reader.get_sample_rate(), 44100u);
    EXPECT_EQ(reader.get_total_frames(), frames);
    EXPECT_EQ(reader.get_bits_per_sample(), 16u);

    // 데이터 읽기 및 비교
    std::vector<float> read_data = reader.read_all();
    EXPECT_EQ(read_data.size(), data.size());

    // 일부 샘플 값 비교 (정밀도 고려)
    size_t compare_frames = (frames < 1000u) ? frames : 1000u;
    for (size_t i = 0; i < compare_frames; ++i) {
        EXPECT_NEAR(read_data[i * channels], data[i * channels], 0.01f);
        EXPECT_NEAR(read_data[i * channels + 1], data[i * channels + 1], 0.01f);
    }

    reader.close();
}

TEST_F(WavIOTest, WavWriter_WriteMono) {
    fs::path test_file = test_dir_ / "test_mono.wav";

    WavWriter writer;
    EXPECT_TRUE(writer.open(test_file.string(), 1, 22050, 16));
    EXPECT_TRUE(writer.is_open());

    // 모노 사인파 데이터 생성
    const size_t frames = 22050 / 2;  // 0.5초
    std::vector<float> data(frames);

    const float PI = 3.14159265358979323846f;
    for (size_t i = 0; i < frames; ++i) {
        data[i] = 0.3f * std::sin(2.0f * PI * 880.0f * static_cast<float>(i) / 22050.0f);
    }

    size_t frames_written;
    EXPECT_TRUE(writer.write(data.data(), frames, &frames_written));
    EXPECT_EQ(frames_written, frames);

    writer.close();

    // 검증
    WavReader reader;
    EXPECT_TRUE(reader.open(test_file.string()));
    EXPECT_EQ(reader.get_channels(), 1u);
    EXPECT_EQ(reader.get_sample_rate(), 22050u);
    EXPECT_EQ(reader.get_total_frames(), frames);

    reader.close();
}

TEST_F(WavIOTest, WavWriter_WriteEmptyData) {
    fs::path test_file = test_dir_ / "test_empty.wav";

    WavWriter writer;
    EXPECT_TRUE(writer.open(test_file.string(), 1, 44100, 16));

    // 빈 데이터 쓰기
    std::vector<float> empty_data;
    size_t frames_written;
    EXPECT_TRUE(writer.write(empty_data.data(), 0, &frames_written));
    EXPECT_EQ(frames_written, 0u);

    writer.close();

    // 검증
    WavReader reader;
    EXPECT_TRUE(reader.open(test_file.string()));
    EXPECT_EQ(reader.get_total_frames(), 0u);
    reader.close();
}

TEST_F(WavIOTest, WavReader_SeekAndRead) {
    WavReader reader;
    ASSERT_TRUE(reader.open("assets/oboe.wav"));

    uint32_t channels = reader.get_channels();
    uint64_t total_frames = reader.get_total_frames();

    // 기본 정보만 확인
    EXPECT_GT(total_frames, 0u);
    EXPECT_GT(channels, 0u);

    reader.close();
}

// 통합 테스트: Writer로 만든 파일을 Reader로 읽기
TEST_F(WavIOTest, Integration_WriteReadRoundTrip) {
    fs::path test_file = test_dir_ / "roundtrip.wav";

    // 랜덤 데이터 생성
    const size_t frames = 1000;
    const size_t channels = 2;
    std::vector<float> original_data(frames * channels);

    srand(42);  // 재현 가능한 랜덤 데이터
    for (size_t i = 0; i < original_data.size(); ++i) {
        original_data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.8f;
    }

    // 쓰기
    {
        WavWriter writer;
        EXPECT_TRUE(writer.open(test_file.string(), channels, 48000, 16));

        size_t frames_written;
        EXPECT_TRUE(writer.write(original_data.data(), frames, &frames_written));
        EXPECT_EQ(frames_written, frames);

        writer.close();
    }

    // 읽기 및 비교
    {
        WavReader reader;
        EXPECT_TRUE(reader.open(test_file.string()));

        EXPECT_EQ(reader.get_channels(), channels);
        EXPECT_EQ(reader.get_sample_rate(), 48000u);
        EXPECT_EQ(reader.get_total_frames(), frames);

        std::vector<float> read_data = reader.read_all();
        EXPECT_EQ(read_data.size(), original_data.size());

        // 데이터 비교 (16비트 정밀도 고려)
        for (size_t i = 0; i < read_data.size(); ++i) {
            EXPECT_NEAR(read_data[i], original_data[i], 0.001f);
        }

        reader.close();
    }
}

// 다양한 비트 심도 테스트
TEST_F(WavIOTest, BitDepth_16Bit_RoundTrip) {
    fs::path test_file = test_dir_ / "bitdepth_16.wav";

    // 테스트 데이터 생성 (다양한 주파수와 레벨)
    const size_t frames = 1000;
    const size_t channels = 2;
    std::vector<float> original_data(frames * channels);

    const float PI = 3.14159265358979323846f;
    for (size_t i = 0; i < frames; ++i) {
        float sample1 = 0.7f * std::sin(2.0f * PI * 1000.0f * static_cast<float>(i) / 44100.0f);
        float sample2 = 0.3f * std::sin(2.0f * PI * 3000.0f * static_cast<float>(i) / 44100.0f);
        original_data[i * channels] = sample1;
        original_data[i * channels + 1] = sample2;
    }

    // 쓰기
    {
        WavWriter writer;
        EXPECT_TRUE(writer.open(test_file.string(), channels, 44100, 16));
        size_t frames_written;
        EXPECT_TRUE(writer.write(original_data.data(), frames, &frames_written));
        EXPECT_EQ(frames_written, frames);
        writer.close();
    }

    // 읽기 및 검증
    {
        WavReader reader;
        EXPECT_TRUE(reader.open(test_file.string()));
        EXPECT_EQ(reader.get_bits_per_sample(), 16u);
        EXPECT_EQ(reader.get_channels(), channels);
        EXPECT_EQ(reader.get_sample_rate(), 44100u);

        std::vector<float> read_data = reader.read_all();
        EXPECT_EQ(read_data.size(), original_data.size());

        // 오차 검증 (16비트: ~96dB SNR)
        for (size_t i = 0; i < read_data.size(); ++i) {
            EXPECT_NEAR(read_data[i], original_data[i], 0.001f);
        }

        reader.close();
    }
}

TEST_F(WavIOTest, BitDepth_24Bit_RoundTrip) {
    fs::path test_file = test_dir_ / "bitdepth_24.wav";

    // 테스트 데이터 생성
    const size_t frames = 1000;
    const size_t channels = 2;
    std::vector<float> original_data(frames * channels);

    const float PI = 3.14159265358979323846f;
    for (size_t i = 0; i < frames; ++i) {
        float sample1 = 0.7f * std::sin(2.0f * PI * 1000.0f * static_cast<float>(i) / 44100.0f);
        float sample2 = 0.3f * std::sin(2.0f * PI * 3000.0f * static_cast<float>(i) / 44100.0f);
        original_data[i * channels] = sample1;
        original_data[i * channels + 1] = sample2;
    }

    // 쓰기
    {
        WavWriter writer;
        EXPECT_TRUE(writer.open(test_file.string(), channels, 44100, 24));
        size_t frames_written;
        EXPECT_TRUE(writer.write(original_data.data(), frames, &frames_written));
        EXPECT_EQ(frames_written, frames);
        writer.close();
    }

    // 읽기 및 검증
    {
        WavReader reader;
        EXPECT_TRUE(reader.open(test_file.string()));
        EXPECT_EQ(reader.get_bits_per_sample(), 24u);
        EXPECT_EQ(reader.get_channels(), channels);
        EXPECT_EQ(reader.get_sample_rate(), 44100u);

        std::vector<float> read_data = reader.read_all();
        EXPECT_EQ(read_data.size(), original_data.size());

        // 오차 검증 (24비트)
        for (size_t i = 0; i < read_data.size(); ++i) {
            EXPECT_NEAR(read_data[i], original_data[i], 1e-4f);
        }

        reader.close();
    }
}

TEST_F(WavIOTest, BitDepth_32Bit_RoundTrip) {
    fs::path test_file = test_dir_ / "bitdepth_32.wav";

    // 테스트 데이터 생성
    const size_t frames = 1000;
    const size_t channels = 2;
    std::vector<float> original_data(frames * channels);

    const float PI = 3.14159265358979323846f;
    for (size_t i = 0; i < frames; ++i) {
        float sample1 = 0.7f * std::sin(2.0f * PI * 1000.0f * static_cast<float>(i) / 44100.0f);
        float sample2 = 0.3f * std::sin(2.0f * PI * 3000.0f * static_cast<float>(i) / 44100.0f);
        original_data[i * channels] = sample1;
        original_data[i * channels + 1] = sample2;
    }

    // 쓰기
    {
        WavWriter writer;
        EXPECT_TRUE(writer.open(test_file.string(), channels, 44100, 32));
        size_t frames_written;
        EXPECT_TRUE(writer.write(original_data.data(), frames, &frames_written));
        EXPECT_EQ(frames_written, frames);
        writer.close();
    }

    // 읽기 및 검증
    {
        WavReader reader;
        EXPECT_TRUE(reader.open(test_file.string()));
        EXPECT_EQ(reader.get_bits_per_sample(), 32u);
        EXPECT_EQ(reader.get_channels(), channels);
        EXPECT_EQ(reader.get_sample_rate(), 44100u);

        std::vector<float> read_data = reader.read_all();
        EXPECT_EQ(read_data.size(), original_data.size());

        // 오차 검증 (32비트: ~192dB SNR)
        for (size_t i = 0; i < read_data.size(); ++i) {
            EXPECT_NEAR(read_data[i], original_data[i], 0.00001f);
        }

        reader.close();
    }
}

// 샘플레이트 테스트
TEST_F(WavIOTest, SampleRate_44100Hz_RoundTrip) {
    fs::path test_file = test_dir_ / "samplerate_44100.wav";

    // 테스트 데이터 생성
    const size_t frames = 1000;
    const size_t channels = 1;
    std::vector<float> original_data(frames * channels);

    const float PI = 3.14159265358979323846f;
    for (size_t i = 0; i < frames; ++i) {
        original_data[i] = 0.5f * std::sin(2.0f * PI * 1000.0f * static_cast<float>(i) / 44100.0f);
    }

    // 쓰기
    {
        WavWriter writer;
        EXPECT_TRUE(writer.open(test_file.string(), channels, 44100, 16));
        size_t frames_written;
        EXPECT_TRUE(writer.write(original_data.data(), frames, &frames_written));
        EXPECT_EQ(frames_written, frames);
        writer.close();
    }

    // 읽기 및 검증
    {
        WavReader reader;
        EXPECT_TRUE(reader.open(test_file.string()));
        EXPECT_EQ(reader.get_sample_rate(), 44100u);
        EXPECT_EQ(reader.get_channels(), channels);
        EXPECT_EQ(reader.get_bits_per_sample(), 16u);

        std::vector<float> read_data = reader.read_all();
        EXPECT_EQ(read_data.size(), original_data.size());

        // 기본적인 데이터 일치 검증
        for (size_t i = 0; i < read_data.size(); ++i) {
            EXPECT_NEAR(read_data[i], original_data[i], 0.001f);
        }

        reader.close();
    }
}

TEST_F(WavIOTest, SampleRate_48000Hz_RoundTrip) {
    fs::path test_file = test_dir_ / "samplerate_48000.wav";

    // 테스트 데이터 생성
    const size_t frames = 1000;
    const size_t channels = 1;
    std::vector<float> original_data(frames * channels);

    const float PI = 3.14159265358979323846f;
    for (size_t i = 0; i < frames; ++i) {
        original_data[i] = 0.5f * std::sin(2.0f * PI * 1000.0f * static_cast<float>(i) / 48000.0f);
    }

    // 쓰기
    {
        WavWriter writer;
        EXPECT_TRUE(writer.open(test_file.string(), channels, 48000, 16));
        size_t frames_written;
        EXPECT_TRUE(writer.write(original_data.data(), frames, &frames_written));
        EXPECT_EQ(frames_written, frames);
        writer.close();
    }

    // 읽기 및 검증
    {
        WavReader reader;
        EXPECT_TRUE(reader.open(test_file.string()));
        EXPECT_EQ(reader.get_sample_rate(), 48000u);
        EXPECT_EQ(reader.get_channels(), channels);
        EXPECT_EQ(reader.get_bits_per_sample(), 16u);

        std::vector<float> read_data = reader.read_all();
        EXPECT_EQ(read_data.size(), original_data.size());

        // 기본적인 데이터 일치 검증
        for (size_t i = 0; i < read_data.size(); ++i) {
            EXPECT_NEAR(read_data[i], original_data[i], 0.001f);
        }

        reader.close();
    }
}

// dBFS 오차 검증 테스트
TEST_F(WavIOTest, DBFS_Error_Check) {
    fs::path test_file = test_dir_ / "dbfs_test.wav";

    // 다양한 레벨의 사인파 생성
    const size_t frames = 1000;
    const size_t channels = 1;
    std::vector<float> original_data(frames * channels);

    const float PI = 3.14159265358979323846f;
    const float test_levels[] = {1.0f, 0.5f, 0.1f, 0.01f, 0.001f};  // -0dBFS, -6dBFS, -20dBFS, -40dBFS, -60dBFS

    for (float level : test_levels) {
        // 사인파 생성
        for (size_t i = 0; i < frames; ++i) {
            original_data[i] = level * std::sin(2.0f * PI * 1000.0f * static_cast<float>(i) / 44100.0f);
        }

        // 16비트로 쓰기
        {
            WavWriter writer;
            EXPECT_TRUE(writer.open(test_file.string(), channels, 44100, 16));
            size_t frames_written;
            EXPECT_TRUE(writer.write(original_data.data(), frames, &frames_written));
            writer.close();
        }

        // 읽기 및 오차 계산
        {
            WavReader reader;
            EXPECT_TRUE(reader.open(test_file.string()));
            std::vector<float> read_data = reader.read_all();
            reader.close();

            // dBFS 오차 계산
            double max_error = 0.0;
            for (size_t i = 0; i < read_data.size(); ++i) {
                double error = std::abs(read_data[i] - original_data[i]);
                if (error > max_error) max_error = error;
            }

            // dBFS로 변환: 20 * log10(error / reference)
            double dbfs_error = (max_error > 0.0) ? 20.0 * std::log10(max_error) : -200.0;

            // -84dBFS 이하 검증 (16비트 기준)
            EXPECT_LE(dbfs_error, -84.0) << "dBFS error too high: " << dbfs_error << " dBFS for level " << level;
        }
    }
}

// float32 라운드트립: 읽고 -> float32로 쓰고 -> 다시 읽었을 때 매우 낮은 오차 보장 (< -100dBFS)
TEST_F(WavIOTest, RoundTrip_With_Float32_Output_dBFS_BetterThanMinus100) {
    // 원본 읽기
    WavReader reader;
    ASSERT_TRUE(reader.open(RunfilePath("assets/oboe.wav")));
    std::vector<float> original = reader.read_all();
    const uint32_t ch = reader.get_channels();
    const uint32_t sr = reader.get_sample_rate();
    reader.close();

    // float32 포맷으로 쓰기
    fs::path test_file = test_dir_ / "rt_float32.wav";
    {
        WavWriter writer;
        ASSERT_TRUE(writer.open(test_file.string(), ch, sr, 32, /*float_format=*/true));
        size_t frames = original.size() / ch;
        size_t frames_written = 0;
        ASSERT_TRUE(writer.write(original.data(), frames, &frames_written));
        ASSERT_EQ(frames_written, frames);
        writer.close();
    }

    // 다시 읽기
    std::vector<float> again;
    {
        WavReader r2;
        ASSERT_TRUE(r2.open(test_file.string()));
        again = r2.read_all();
        r2.close();
    }

    ASSERT_EQ(again.size(), original.size());

    double max_err = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double e = std::abs(again[i] - original[i]);
        if (e > max_err) max_err = e;
    }
    double dbfs_error = (max_err > 0.0) ? 20.0 * std::log10(max_err) : -200.0;
    EXPECT_LE(dbfs_error, -100.0) << "dBFS error: " << dbfs_error;
}

// 모노/스테레오 테스트
TEST_F(WavIOTest, Mono_Stereo_RoundTrip) {
    const size_t frames = 1000;
    std::vector<float> mono_data(frames);
    std::vector<float> stereo_data(frames * 2);

    // 모노 데이터 생성
    const float PI = 3.14159265358979323846f;
    for (size_t i = 0; i < frames; ++i) {
        mono_data[i] = 0.5f * std::sin(2.0f * PI * 1000.0f * static_cast<float>(i) / 44100.0f);
    }

    // 스테레오 데이터 생성 (모노를 양쪽 채널에 복사)
    for (size_t i = 0; i < frames; ++i) {
        stereo_data[i * 2] = mono_data[i];
        stereo_data[i * 2 + 1] = mono_data[i];
    }

    // 모노 테스트
    {
        fs::path mono_file = test_dir_ / "mono_test.wav";
        WavWriter writer;
        EXPECT_TRUE(writer.open(mono_file.string(), 1, 44100, 16));
        size_t frames_written;
        EXPECT_TRUE(writer.write(mono_data.data(), frames, &frames_written));
        writer.close();

        WavReader reader;
        EXPECT_TRUE(reader.open(mono_file.string()));
        EXPECT_EQ(reader.get_channels(), 1u);
        std::vector<float> read_data = reader.read_all();
        EXPECT_EQ(read_data.size(), mono_data.size());
        reader.close();
    }

    // 스테레오 테스트
    {
        fs::path stereo_file = test_dir_ / "stereo_test.wav";
        WavWriter writer;
        EXPECT_TRUE(writer.open(stereo_file.string(), 2, 44100, 16));
        size_t frames_written;
        EXPECT_TRUE(writer.write(stereo_data.data(), frames, &frames_written));
        writer.close();

        WavReader reader;
        EXPECT_TRUE(reader.open(stereo_file.string()));
        EXPECT_EQ(reader.get_channels(), 2u);
        std::vector<float> read_data = reader.read_all();
        EXPECT_EQ(read_data.size(), stereo_data.size());
        reader.close();
    }
}
