//
//  audio.hpp
//  MNN
//
//  Created by MNN on 2024/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_AUDIO_HPP
#define MNN_AUDIO_HPP

#include <MNN/MNNDefine.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

namespace MNN {
namespace AUDIO {

using namespace Express;

enum WINDOW_TYPE { HAMMING = 0, HANNING = 1, POVEY = 2, RECTANGULAR = 3, BLACKMAN = 4 };

/**
 * Structure to store parameters for the `melscale_fbanks`.
 */
struct MelscaleParams {
    /** Number of mel filterbanks, default is 128. */
    int n_mels = 128;
    /** Number of FFT bins, default is 400. */
    int n_fft = 400;
    /** Sample rate, default is 16000. */
    int sample_rate = 16000;
    /** Scale to use `htk` or `slaney`, default is true mean `htk`. */
    bool htk = true;
    /** Divide the triangular mel weights by the width of the mel band, default is false. */
    bool norm = false;
    /** Minimum frequency, default is 0. */
    float f_min = 0.0;
    /** Maximum frequency, default is 0.(equal to `sample_rate / 2`). */
    float f_max = 0.0;
};

/**
 * Structure to store parameters for the `spectrogram`.
 */
struct SpectrogramParams {
    /** Size of the FFT window, default is 400. */
    int n_fft = 400;

    /** Hop length between frames, default is 0 (equal to `n_fft / 2`). */
    int hop_length = 0;

    /** Window length, default is 0 (equal to `n_fft`). */
    int win_length = 0;

    /** Type of window function, default is Hann window (HANNING). */
    int window_type = HANNING;

    /** Constant padding value on the left side of the input audio, default is 0. */
    int pad_left = 0;

    /** Constant padding value on the right side of the input audio, default is 0. */
    int pad_right = 0;

    /** Whether to apply center padding to the STFT input, default is false. */
    bool center = false;

    /** Whether to normalize the output, default is false. */
    bool normalized = false;

    /** Padding mode of `center = true`, default is reflect (REFLECT). */
    int pad_mode = REFLECT;

    /** Power scaling factor, default is 2.0. */
    float power = 2.0;
};

/**
 * @brief load audio from file
 * @param filename audio file path
 * @param frame_offset start frame
 * @param num_frames number of frames
 * @return pair<audio tensor, sample rate>
 */
MNN_PUBLIC std::pair<VARP, int> load(const std::string& filename, int sr = 0, int frame_offset = 0,
                                     int num_frames = -1);

/**
 * @brief save audio to file
 * @param filename audio file path
 * @param audio audio tensor
 * @param sample_rate sample rate
 */
MNN_PUBLIC bool save(const std::string& filename, VARP audio, int sample_rate);

/**
 * @brief compute hamming window
 * @param window_size window size
 * @param periodic periodic
 * @param alpha alpha
 * @param beta beta
 * @return hamming window tensor
 */
MNN_PUBLIC VARP hamming_window(int window_size, bool periodic = false, float alpha = 0.54, float beta = 0.46);

/**
 * @brief compute hann window
 * @param window_size window size
 * @param periodic periodic
 * @return hann window tensor
 */
MNN_PUBLIC VARP hann_window(int window_size, bool periodic = false);

/**
 * @brief compute melscale fbanks
 * @param params melscale fbanks params
 * @return melscale fbanks var
 */
MNN_PUBLIC VARP melscale_fbanks(const MelscaleParams* params = nullptr);

/**
 * @brief compute spectrogram from audio
 * @param waveform waveform tensor
 * @param params spectrogram params
 * @return spectrogram tensor
 */
MNN_PUBLIC VARP spectrogram(VARP waveform, const SpectrogramParams* params = nullptr);

/**
 * @brief compute mel spectrogram from audio
 * @param waveform waveform of audio signal.
 * @param params mel spectrogram params
 * @param params spectrogram params
 * @return mel spectrogram tensor
 */
MNN_PUBLIC VARP mel_spectrogram(VARP waveform, const MelscaleParams* mel_params = nullptr,
                                const SpectrogramParams* spec_params = nullptr);

/**
 * @brief compute fbank from audio
 * @param waveform waveform tensor
 * @param sampling_rate sampling rate
 * @param n_mels number of mel bins
 * @param n_fft number of fft bins
 * @param hop_length hop length
 * @param dither dither
 * @addindex preemphasis preemphasis
 * @return fbank tensor
 */
MNN_PUBLIC VARP fbank(VARP waveform, int sampling_rate = 16000, int n_mels = 80, int n_fft = 400,
                      int hop_length = 160, float dither = 0.f, float preemphasis = 0.97);

/**
 * @brief compute whisper fbank from audio
 * @param waveform waveform tensor
 * @param sample_rate sample rate
 * @param n_mels number of mel bins
 * @param n_fft number of fft bins
 * @param hop_length hop length
 * @param chunk_len chunk length
 * @return fbank tensor
 */
MNN_PUBLIC VARP whisper_fbank(VARP waveform, int sample_rate = 16000, int n_mels = 128, int n_fft = 400,
                              int hop_length = 160, int chunk_len = 0);

} // namespace AUDIO
} // namespace MNN

#endif // MNN_AUDIO_HPP