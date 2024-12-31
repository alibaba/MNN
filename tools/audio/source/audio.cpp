//
//  audio.cpp
//  MNN
//
//  Created by MNN on 2024/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "audio/audio.hpp"
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <cmath>
#include <algorithm>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#ifndef M_PI
#define M_PI 3.141592654
#endif
#ifdef _MSC_VER
#define NOMINMAX
#include <intrin.h>
#include <windows.h>
#endif


namespace MNN {
namespace AUDIO {
#ifdef _MSC_VER
inline uint32_t mnn_clz( uint32_t value ) {
    DWORD leading_zero = 0;
    if (_BitScanReverse(&leading_zero, value)) {
        return 31 - leading_zero;
    }else {
         // Same remarks as above
         return 32;
    }
}
#else
inline uint32_t mnn_clz( uint32_t value ) {
    return __builtin_clz(value);
}
#endif
struct WaveHeader {
    void SeekToDataChunk(std::istream &is) {
        //                              a t a d
        while (is && subchunk2_id != 0x61746164) {
            is.seekg(subchunk2_size, std::istream::cur);
            is.read(reinterpret_cast<char *>(&subchunk2_id), sizeof(int32_t));
            is.read(reinterpret_cast<char *>(&subchunk2_size), sizeof(int32_t));
        }
    }
    int32_t chunk_id = 0x46464952; // "RIFF"
    int32_t chunk_size;
    int32_t format         = 0x45564157; // "WAVE"
    int32_t subchunk1_id   = 0x20746d66; // "fmt "
    int32_t subchunk1_size = 16;         // PCM
    int16_t audio_format   = 1;          // PCM = 1
    int16_t num_channels   = 1;          // Mono
    int32_t sample_rate;
    int32_t byte_rate;
    int16_t block_align;
    int16_t bits_per_sample = 16;
    int32_t subchunk2_id    = 0x61746164; // "data"
    int32_t subchunk2_size;
};

std::pair<VARP, int> load(const std::string &filename, int sr, int frame_offset, int num_frames) {
    std::ifstream is(filename, std::ifstream::binary);
    auto ret = std::make_pair<VARP, int>(nullptr, 0);
    if (!is) {
        MNN_ERROR("Failed to open file: %s\n", filename.c_str());
        return ret;
    }
    WaveHeader header{};
    is.read(reinterpret_cast<char *>(&header.chunk_id), sizeof(header.chunk_id));
    if (header.chunk_id != 0x46464952) { // "RIFF"
        MNN_ERROR("Expected chunk_id RIFF. Given: 0x%08x\n", header.chunk_id);
        return ret;
    }

    is.read(reinterpret_cast<char *>(&header.chunk_size), sizeof(header.chunk_size));
    is.read(reinterpret_cast<char *>(&header.format), sizeof(header.format));
    if (header.format != 0x45564157) { // "WAVE"
        MNN_ERROR("Expected format WAVE. Given: 0x%08x\n", header.format);
        return ret;
    }

    is.read(reinterpret_cast<char *>(&header.subchunk1_id), sizeof(header.subchunk1_id));
    is.read(reinterpret_cast<char *>(&header.subchunk1_size), sizeof(header.subchunk1_size));

    if (header.subchunk1_id == 0x4b4e554a) { // "JUNK"
        is.seekg(header.subchunk1_size, std::istream::cur);
        is.read(reinterpret_cast<char *>(&header.subchunk1_id), sizeof(header.subchunk1_id));
        is.read(reinterpret_cast<char *>(&header.subchunk1_size), sizeof(header.subchunk1_size));
    }

    if (header.subchunk1_id != 0x20746d66) { // "fmt "
        MNN_ERROR("Expected subchunk1_id 'fmt '. Given: 0x%08x\n", header.subchunk1_id);
        return ret;
    }

    if (header.subchunk1_size != 16 && header.subchunk1_size != 18) {
        MNN_ERROR("Expected subchunk1_size 16 or 18. Given: %d\n", header.subchunk1_size);
        return ret;
    }

    is.read(reinterpret_cast<char *>(&header.audio_format), sizeof(header.audio_format));
    if (header.audio_format != 1 && header.audio_format != 3) {
        MNN_ERROR("Unsupported audio_format: %d. Only PCM(1) and IEEE Float(3) supported.\n", header.audio_format);
        return ret;
    }

    is.read(reinterpret_cast<char *>(&header.num_channels), sizeof(header.num_channels));
    if (header.num_channels != 1) {
        MNN_ERROR("Warning: %d channels found. Only the first channel will be used.\n", header.num_channels);
    }

    is.read(reinterpret_cast<char *>(&header.sample_rate), sizeof(header.sample_rate));
    is.read(reinterpret_cast<char *>(&header.byte_rate), sizeof(header.byte_rate));
    is.read(reinterpret_cast<char *>(&header.block_align), sizeof(header.block_align));
    is.read(reinterpret_cast<char *>(&header.bits_per_sample), sizeof(header.bits_per_sample));

    if (header.byte_rate != (header.sample_rate * header.num_channels * header.bits_per_sample / 8)) {
        MNN_ERROR("Incorrect byte rate: %d. Expected: %d\n", header.byte_rate,
                  header.sample_rate * header.num_channels * header.bits_per_sample / 8);
        return ret;
    }

    if (header.block_align != (header.num_channels * header.bits_per_sample / 8)) {
        MNN_ERROR("Incorrect block align: %d. Expected: %d\n", header.block_align,
                  header.num_channels * header.bits_per_sample / 8);
        return ret;
    }

    if (header.bits_per_sample != 8 && header.bits_per_sample != 16 && header.bits_per_sample != 32) {
        MNN_ERROR("Unsupported bits_per_sample: %d. Only 8, 16, or 32 bits per sample supported.\n",
                  header.bits_per_sample);
        return ret;
    }

    if (header.subchunk1_size == 18) {
        int16_t extra_size;
        is.read(reinterpret_cast<char *>(&extra_size), sizeof(int16_t));
        if (extra_size != 0) {
            MNN_ERROR("Unexpected extra size: %d. Expected 0.\n", extra_size);
            return ret;
        }
    }

    is.read(reinterpret_cast<char *>(&header.subchunk2_id), sizeof(header.subchunk2_id));
    is.read(reinterpret_cast<char *>(&header.subchunk2_size), sizeof(header.subchunk2_size));
    header.SeekToDataChunk(is);

    if (!is) {
        MNN_ERROR("Could not locate data chunk.\n");
        return ret;
    }

    int total_frames = header.subchunk2_size / header.block_align;
    if (frame_offset < 0 || frame_offset >= total_frames) {
        MNN_ERROR("Frame offset out of range.\n");
        return ret;
    }

    if (num_frames <= 0 || frame_offset + num_frames > total_frames) {
        num_frames = total_frames - frame_offset;
    }

    is.seekg(frame_offset * header.block_align, std::istream::cur);

    ret.first    = _Input({num_frames}, NHWC);
    ret.second   = header.sample_rate;
    auto ans_ptr = ret.first->writeMap<float>();
    if (header.bits_per_sample == 16 && header.audio_format == 1) {
        std::vector<int16_t> samples(num_frames * header.num_channels);
        is.read(reinterpret_cast<char *>(samples.data()), num_frames * header.block_align);
        if (!is) {
            MNN_ERROR("Failed to read audio data.\n");
            return ret;
        }
        for (int i = 0; i < num_frames; ++i) {
            ans_ptr[i] = samples[i * header.num_channels] / 32768.f;
        }
    } else if (header.bits_per_sample == 8 && header.audio_format == 1) {
        std::vector<uint8_t> samples(num_frames * header.num_channels);
        is.read(reinterpret_cast<char *>(samples.data()), num_frames * header.block_align);
        if (!is) {
            MNN_ERROR("Failed to read audio data.\n");
            return ret;
        }
        for (int i = 0; i < num_frames; ++i) {
            ans_ptr[i] = static_cast<float>(samples[i * header.num_channels]) / 128.f - 1.f;
        }
    } else if (header.bits_per_sample == 32 && header.audio_format == 1) {
        std::vector<int32_t> samples(num_frames * header.num_channels);
        is.read(reinterpret_cast<char *>(samples.data()), num_frames * header.block_align);
        if (!is) {
            MNN_ERROR("Failed to read audio data.\n");
            return ret;
        }
        for (int i = 0; i < num_frames; ++i) {
            ans_ptr[i] = static_cast<float>(samples[i * header.num_channels]) / static_cast<float>(INT32_MAX);
        }
    } else if (header.bits_per_sample == 32 && header.audio_format == 3) {
        std::vector<float> samples(num_frames * header.num_channels);
        is.read(reinterpret_cast<char *>(samples.data()), num_frames * header.block_align);
        if (!is) {
            MNN_ERROR("Failed to read audio data.\n");
            return ret;
        }
        for (int i = 0; i < num_frames; ++i) {
            ans_ptr[i] = samples[i * header.num_channels];
        }
    } else {
        MNN_ERROR("Unsupported bits per sample: %d or audio format: %d.\n", header.bits_per_sample,
                  header.audio_format);
        return ret;
    }

    if (sr > 0 && sr != ret.second) {
        // resample
        float resample_ratio    = static_cast<float>(sr) / header.sample_rate;
        int resample_num_frames = static_cast<int>(num_frames * resample_ratio);
        auto resampled_data     = _Input({resample_num_frames}, NHWC);
        auto src                = ret.first->readMap<float>();
        auto dst                = resampled_data->writeMap<float>();
        for (int i = 0; i < resample_num_frames; ++i) {
            float interp_index = i / resample_ratio;
            int low_index      = static_cast<int>(interp_index);
            int high_index     = std::min(low_index + 1, num_frames - 1);
            float frac         = interp_index - low_index;
            dst[i]             = (1 - frac) * src[low_index] + frac * src[high_index];
        }
        ret.first  = resampled_data;
        ret.second = sr;
    }
    return ret;
}

bool save(const std::string &filename, VARP audio, int sample_rate) {
    std::ofstream os(filename, std::ios::binary);
    if (!os) {
        MNN_ERROR("Failed to open file for writing: %s\n", filename.c_str());
        return false;
    }

    auto audio_size = audio->getInfo()->size;
    auto audio_ptr  = audio->readMap<float>();
    WaveHeader header;
    header.num_channels   = 1;
    header.sample_rate    = sample_rate;
    header.byte_rate      = sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align    = header.num_channels * (header.bits_per_sample / 8);
    header.subchunk2_size = audio_size * (header.bits_per_sample / 8);
    header.chunk_size     = 36 + header.subchunk2_size;

    os.write(reinterpret_cast<const char *>(&header), sizeof(WaveHeader));

    // Convert float samples to int16 and write to file
    for (int i = 0; i < audio_size; i++) {
        float sample       = audio_ptr[i];
        int16_t int_sample = static_cast<int16_t>(std::max(-1.0f, std::min(1.0f, sample)) * 32767);
        os.write(reinterpret_cast<const char *>(&int_sample), sizeof(int16_t));
    }

    if (!os) {
        MNN_ERROR("Failed to write audio data to file.\n");
        return false;
    }

    os.close();
    return true;
}

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

unsigned int next_power_of_2(unsigned int x) {
    if (x == 0)
        return 1;
    if ((x & (x - 1)) == 0)
        return x;
    return 1U << (32 - mnn_clz(x));
}

VARP hamming_window(int n_fft, bool periodic, float alpha, float beta) {
    auto window     = _Input({n_fft}, NHWC);
    auto window_ptr = window->writeMap<float>();
    int N           = periodic ? n_fft : n_fft - 1;
    for (int n = 0; n < n_fft; ++n) {
        window_ptr[n] = alpha - beta * std::cos(2.0 * M_PI * n / N);
    }
    return window;
}

VARP hann_window(int n_fft, bool periodic) {
    auto window     = _Input({n_fft}, NHWC);
    auto window_ptr = window->writeMap<float>();
    int N           = periodic ? n_fft : n_fft - 1;
    for (int n = 0; n < n_fft; ++n) {
        window_ptr[n] = 0.5 * (1 - std::cos(2 * M_PI * n / N));
    }
    return window;
}

float hz_to_mel(float freq, bool htk) {
    if (htk) {
        return 2595 * std::log10(1 + freq / 700);
    } else {
        constexpr float f_min = 0.0, f_sp = 200.0 / 3.0, min_log_hz = 1000.0;
        constexpr float logstep     = 0.06875177742094912;
        constexpr float min_log_mel = (min_log_hz - f_min) / f_sp;
        float mels                  = (freq - f_min) / f_sp;
        if (freq >= min_log_hz) {
            mels = min_log_mel + std::log(freq / min_log_hz) / logstep;
        }
        return mels;
    }
}

float mel_to_hz(float mel, bool htk) {
    if (htk) {
        return 700 * (std::pow(10, mel / 2595.0) - 1);
    } else {
        constexpr float f_min = 0.0f, f_sp = 200.0f / 3, min_log_hz = 1000.0f;
        constexpr float logstep     = 0.06875177742094912;
        constexpr float min_log_mel = (min_log_hz - f_min) / f_sp;
        float freq                  = f_min + f_sp * mel;
        if (mel >= min_log_mel) {
            freq = min_log_hz * std::exp(logstep * (mel - min_log_mel));
        }
        return freq;
    }
}

VARP melscale_fbanks(const MelscaleParams *params) {
    int n_mels = 128, n_fft = 400, sample_rate = 16000;
    bool htk = true, norm = false;
    float f_min = 0.0, f_max = 0.0;
    if (params != nullptr) {
        n_mels      = params->n_mels;
        n_fft       = params->n_fft;
        sample_rate = params->sample_rate;
        htk         = params->htk;
        norm        = params->norm;
        f_min       = params->f_min;
        f_max       = params->f_max;
    }
    int n_freqs   = n_fft / 2 + 1;
    float nyquist = 0.5 * sample_rate;
    std::vector<float> all_freqs(n_freqs);
    for (int i = 0; i < n_freqs; ++i) {
        all_freqs[i] = i * nyquist / (n_freqs - 1);
    }
    f_max         = f_max <= 0.0 ? nyquist : f_max;
    float m_min   = hz_to_mel(f_min, htk);
    float m_max   = hz_to_mel(f_max, htk);
    float m_delta = (m_max - m_min) / (n_mels + 1);

    auto bins     = _Input({n_mels, n_freqs}, NHWC);
    auto bins_ptr = bins->writeMap<float>();
    for (int n = 0; n < n_mels; ++n) {
        float left  = mel_to_hz(m_min + m_delta * (n + 0), htk);
        float curr  = mel_to_hz(m_min + m_delta * (n + 1), htk);
        float right = mel_to_hz(m_min + m_delta * (n + 2), htk);
        float enorm = (htk && norm) ? 1.0 : 2.0 / (right - left);
        for (int k = 0; k < n_freqs; ++k) {
            float val = 0.f, f_k = all_freqs[k];
            if (f_k >= left && f_k <= curr) {
                val = (f_k - left) / (curr - left);
            } else if (f_k > curr && f_k <= right) {
                val = (right - f_k) / (right - curr);
            }
            bins_ptr[n * n_freqs + k] = val * enorm;
        }
    }
    return bins;
}

VARP spectrogram(VARP waveform, const SpectrogramParams *params) {
    int pad_left = 0, pad_right = 0, pad_mode = REFLECT;
    int n_fft = 400, hop_length = 0, win_length = 0, window_type = HANNING;
    bool center = false, normalized = false;
    float power = 2.0;
    if (params) {
        pad_left    = params->pad_left;
        pad_right   = params->pad_right;
        center      = params->center;
        pad_mode    = params->pad_mode;
        n_fft       = params->n_fft;
        hop_length  = params->hop_length;
        win_length  = params->win_length;
        window_type = params->window_type;
        normalized  = params->normalized;
        power       = params->power;
    }
    if (pad_left > 1 || pad_right > 1) {
        waveform = _Pad(waveform, _var<int>({pad_left, pad_right}, {2}), CONSTANT);
    }
    if (center) {
        waveform = _Pad(waveform, _var<int>({n_fft / 2, n_fft / 2}, {2}), static_cast<PadValueMode>(pad_mode));
    }
    hop_length = hop_length ? hop_length : n_fft / 2;
    win_length = win_length ? win_length : n_fft;
    VARP window;
    switch (window_type) {
        case HANNING:
            window = hann_window(win_length);
            break;
        case HAMMING:
            window = hamming_window(win_length);
            break;
        default:
            window = hann_window(win_length);
            break;
    }
    auto specgram = _Stft(waveform, window, n_fft, hop_length);
    if (normalized) {
        float window_norm = std::sqrt(_ReduceSum(_Square(window))->readMap<float>()[0]);
        specgram          = specgram / _Scalar<float>(window_norm);
    }
    if (power == 2.0) {
        specgram = _Square(specgram);
    } else if (power > 2.0) {
        specgram = _Pow(specgram, _Scalar<float>(power));
    }
    return specgram;
}

VARP mel_spectrogram(VARP waveform, const MelscaleParams *mel_params, const SpectrogramParams *spec_params) {
    auto banks        = melscale_fbanks(mel_params);
    auto specgram     = spectrogram(waveform, spec_params);
    auto mel_specgram = _MatMul(specgram, banks, false, true);
    return mel_specgram;
}

VARP fbank(VARP waveform, int sampling_rate, int n_mels, int n_fft, int hop_length, float dither, float preemphasis) {
    int wav_len      = waveform->getInfo()->size;
    int frame_num    = (wav_len - n_fft) / hop_length + 1;
    if (frame_num <= 0 || wav_len < n_fft) {
        return nullptr; // frame_num is zero
    }
    // get_strided: sizes: [m, n_fft], strides: [windows_shift, 1]
    int m                           = 1 + (wav_len - n_fft) / hop_length;
    std::vector<int> strided_region = {
        0, // src offset
        wav_len,
        hop_length,
        1, // src strides
        0, // dst offset
        m * n_fft,
        n_fft,
        1, // dst strides
        1,
        m,
        n_fft // dst sizes
    };
    auto strided_wav = _Raster({waveform}, strided_region, {m, n_fft});
    auto wav_dim     = strided_wav->getInfo()->dim;
    // add_dither
    if (dither > 0.f) {
        auto rand_dither = _RandomUnifom(_var<int>(wav_dim, {static_cast<int>(wav_dim.size())}),
                                         halide_type_of<float>(), -dither, dither);
        strided_wav      = strided_wav + rand_dither;
    }
    // subtract each row/frame by its mean
    {
        auto row_means   = _ReduceMean(strided_wav, {-1}, true);
        strided_wav      = strided_wav - row_means;
    }
    if (preemphasis != 0.f) {
        std::vector<int> offset_region          = {
            // region 0
            0,                               // src offset
            m * n_fft, n_fft, 1, // src strides
            0,                               // dst offset
            m * n_fft, n_fft, 1, // dst strides
            1, m, 1,                         // dst sizes
            // region 1
            0,                               // src offset
            m * n_fft, n_fft, 1, // src strides
            1,                               // dst offset
            m * n_fft, n_fft, 1, // dst strides
            1, m, n_fft - 1            // dst sizes
        };
        auto offset_strided_wav = _Raster({strided_wav, strided_wav}, offset_region, {m, n_fft});
        strided_wav             = strided_wav - _Scalar<float>(preemphasis) * offset_strided_wav;
    }
    int padded_n_fft = next_power_of_2(n_fft);
    MelscaleParams mel_params;
    mel_params.n_mels      = n_mels;
    mel_params.n_fft       = padded_n_fft;
    mel_params.sample_rate = sampling_rate;
    mel_params.f_min       = 20.0;
    SpectrogramParams spec_params;
    spec_params.n_fft      = padded_n_fft;
    spec_params.hop_length = n_fft;
    auto mel_energies      = mel_spectrogram(strided_wav, &mel_params, &spec_params);
    mel_energies           = _Log(mel_energies);
    return mel_energies;
}

VARP whisper_fbank(VARP waveform, int sample_rate, int n_mels, int n_fft, int hop_length, int chunk_len) {
    int n_samples = chunk_len * sample_rate;
    int pad_right = n_samples - waveform->getInfo()->size;
    pad_right     = pad_right > 0 ? pad_right : 0;
    MelscaleParams mel_params;
    mel_params.n_mels      = n_mels;
    mel_params.n_fft       = n_fft;
    mel_params.sample_rate = sample_rate;
    mel_params.htk         = false;
    mel_params.norm        = true;
    SpectrogramParams spec_params;
    spec_params.pad_right  = pad_right;
    spec_params.n_fft      = n_fft;
    spec_params.hop_length = hop_length;
    spec_params.center     = true;
    auto mel_specgram      = mel_spectrogram(waveform, &mel_params, &spec_params);
    mel_specgram =
        _Slice(mel_specgram, _var<int>({0, 0}, {2}), _var<int>({mel_specgram->getInfo()->dim[0] - 1, -1}, {2}));
    auto log_specgram = _Log(mel_specgram) / _Log(_Scalar<float>(10.0));
    log_specgram      = _Maximum(log_specgram, _ReduceMax(log_specgram) - _Scalar<float>(8.0));
    log_specgram      = (log_specgram + _Scalar<float>(4.0)) / _Scalar<float>(4.0);
    // NHWC -> NCHW
    log_specgram = _Unsqueeze(log_specgram, {0, 1});
    log_specgram = _Convert(log_specgram, NCHW);
    log_specgram = _Squeeze(log_specgram, {2});
    return log_specgram;
}

} // namespace AUDIO
} // namespace MNN
