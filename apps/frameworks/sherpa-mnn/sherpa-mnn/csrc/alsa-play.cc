// sherpa-mnn/csrc/alsa-play.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifdef SHERPA_ONNX_ENABLE_ALSA

#include "sherpa-mnn/csrc/alsa-play.h"

#include <algorithm>

namespace sherpa_mnn {

AlsaPlay::AlsaPlay(const char *device_name, int32_t sample_rate) {
  int32_t err = snd_pcm_open(&handle_, device_name, SND_PCM_STREAM_PLAYBACK, 0);

  if (err) {
    fprintf(stderr, "Unable to open: %s. %s\n", device_name, snd_strerror(err));
    exit(-1);
  }

  SetParameters(sample_rate);
}

AlsaPlay::~AlsaPlay() {
  if (handle_) {
    int32_t err = snd_pcm_close(handle_);
    if (err < 0) {
      printf("Failed to close pcm: %s\n", snd_strerror(err));
    }
  }
}

void AlsaPlay::SetParameters(int32_t sample_rate) {
  // set the following parameters
  // 1. sample_rate
  // 2. sample format: int16_t
  // 3. num_channels: 1
  snd_pcm_hw_params_t *params;
  snd_pcm_hw_params_alloca(&params);
  snd_pcm_hw_params_any(handle_, params);

  int32_t err = snd_pcm_hw_params_set_access(handle_, params,
                                             SND_PCM_ACCESS_RW_INTERLEAVED);
  if (err < 0) {
    printf("SND_PCM_ACCESS_RW_INTERLEAVED is not supported: %s\n",
           snd_strerror(err));
    exit(-1);
  }

  err = snd_pcm_hw_params_set_format(handle_, params, SND_PCM_FORMAT_S16_LE);

  if (err < 0) {
    printf("Can't set format to 16-bit: %s\n", snd_strerror(err));
    exit(-1);
  }

  err = snd_pcm_hw_params_set_channels(handle_, params, 1);

  if (err < 0) {
    printf("Can't set channel number to 1: %s\n", snd_strerror(err));
  }

  uint32_t rate = sample_rate;
  err = snd_pcm_hw_params_set_rate_near(handle_, params, &rate, 0);
  if (err < 0) {
    printf("Can't set rate to %d. %s\n", rate, snd_strerror(err));
  }

  err = snd_pcm_hw_params(handle_, params);
  if (err < 0) {
    printf("Can't set hardware parameters. %s\n", snd_strerror(err));
    exit(-1);
  }

  uint32_t tmp;
  snd_pcm_hw_params_get_rate(params, &tmp, 0);
  int32_t actual_sample_rate = tmp;
  if (actual_sample_rate != sample_rate) {
    fprintf(stderr,
            "Creating a resampler:\n"
            "   in_sample_rate: %d\n"
            "   output_sample_rate: %d\n",
            sample_rate, actual_sample_rate);

    float min_freq = std::min(actual_sample_rate, sample_rate);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler_ = std::make_unique<LinearResample>(
        sample_rate, actual_sample_rate, lowpass_cutoff, lowpass_filter_width);
  }

  snd_pcm_uframes_t frames;
  snd_pcm_hw_params_get_period_size(params, &frames, 0);
  buf_.resize(frames);
}

void AlsaPlay::Play(const std::vector<float> &samples) {
  std::vector<float> tmp;
  const float *p = samples.data();
  int32_t num_samples = samples.size();
  if (resampler_) {
    resampler_->Resample(samples.data(), samples.size(), false, &tmp);
    p = tmp.data();
    num_samples = tmp.size();
  }

  int32_t frames = buf_.size();
  int32_t i = 0;
  for (; i + frames < num_samples; i += frames) {
    for (int32_t k = 0; k != frames; ++k) {
      buf_[k] = p[i + k] * 32767;
    }

    int32_t err = snd_pcm_writei(handle_, buf_.data(), frames);
    if (err == -EPIPE) {
      printf("XRUN.\n");
      snd_pcm_prepare(handle_);
    } else if (err < 0) {
      printf("Can't write to PCM device: %s\n", snd_strerror(err));
      exit(-1);
    }
  }

  if (i < num_samples) {
    for (int32_t k = 0; k + i < num_samples; ++k) {
      buf_[k] = p[i + k] * 32767;
    }

    int32_t err = snd_pcm_writei(handle_, buf_.data(), num_samples - i);
    if (err == -EPIPE) {
      printf("XRUN.\n");
      snd_pcm_prepare(handle_);
    } else if (err < 0) {
      printf("Can't write to PCM device: %s\n", snd_strerror(err));
      exit(-1);
    }
  }
}

void AlsaPlay::Drain() {
  int32_t err = snd_pcm_drain(handle_);
  if (err < 0) {
    printf("Failed to drain pcm. %s\n", snd_strerror(err));
  }
}

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_ENABLE_ALSA
