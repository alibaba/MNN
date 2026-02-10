// sherpa-mnn/csrc/sherpa-alsa.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifdef SHERPA_ONNX_ENABLE_ALSA

#include "sherpa-mnn/csrc/alsa.h"

#include <algorithm>

#include "alsa/asoundlib.h"

namespace sherpa_mnn {

void ToFloat(const std::vector<int16_t> &in, int32_t num_channels,
             std::vector<float> *out) {
  out->resize(in.size() / num_channels);

  int32_t n = in.size();
  for (int32_t i = 0, k = 0; i < n; i += num_channels, ++k) {
    (*out)[k] = in[i] / 32768.;
  }
}

Alsa::Alsa(const char *device_name) {
  const char *kDeviceHelp = R"(
Please use the command:

  arecord -l

to list all available devices. For instance, if the output is:

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and device 0 on that card, please use:

  plughw:3,0

  )";

  int32_t err =
      snd_pcm_open(&capture_handle_, device_name, SND_PCM_STREAM_CAPTURE, 0);
  if (err) {
    fprintf(stderr, "Unable to open: %s. %s\n", device_name, snd_strerror(err));
    fprintf(stderr, "%s\n", kDeviceHelp);
    exit(-1);
  }

  snd_pcm_hw_params_t *hw_params;
  snd_pcm_hw_params_alloca(&hw_params);

  err = snd_pcm_hw_params_any(capture_handle_, hw_params);
  if (err) {
    fprintf(stderr, "Failed to initialize hw_params: %s\n", snd_strerror(err));
    exit(-1);
  }

  err = snd_pcm_hw_params_set_access(capture_handle_, hw_params,
                                     SND_PCM_ACCESS_RW_INTERLEAVED);
  if (err) {
    fprintf(stderr, "Failed to set access type: %s\n", snd_strerror(err));
    exit(-1);
  }

  err = snd_pcm_hw_params_set_format(capture_handle_, hw_params,
                                     SND_PCM_FORMAT_S16_LE);
  if (err) {
    fprintf(stderr, "Failed to set format: %s\n", snd_strerror(err));
    exit(-1);
  }

  // mono
  err = snd_pcm_hw_params_set_channels(capture_handle_, hw_params, 1);
  if (err) {
    fprintf(stderr, "Failed to set number of channels to 1. %s\n",
            snd_strerror(err));

    err = snd_pcm_hw_params_set_channels(capture_handle_, hw_params, 2);
    if (err) {
      fprintf(stderr, "Failed to set number of channels to 2. %s\n",
              snd_strerror(err));

      exit(-1);
    }
    actual_channel_count_ = 2;
    fprintf(stderr,
            "Channel count is set to 2. Will use only 1 channel of it.\n");
  }

  uint32_t actual_sample_rate = expected_sample_rate_;

  int32_t dir = 0;
  err = snd_pcm_hw_params_set_rate_near(capture_handle_, hw_params,
                                        &actual_sample_rate, &dir);
  if (err) {
    fprintf(stderr, "Failed to set sample rate to, %d: %s\n",
            expected_sample_rate_, snd_strerror(err));
    exit(-1);
  }
  actual_sample_rate_ = actual_sample_rate;

  if (actual_sample_rate_ != expected_sample_rate_) {
    fprintf(stderr, "Failed to set sample rate to %d\n", expected_sample_rate_);
    fprintf(stderr, "Current sample rate is %d\n", actual_sample_rate_);
    fprintf(stderr,
            "Creating a resampler:\n"
            "   in_sample_rate: %d\n"
            "   output_sample_rate: %d\n",
            actual_sample_rate_, expected_sample_rate_);

    float min_freq = std::min(actual_sample_rate_, expected_sample_rate_);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler_ = std::make_unique<LinearResample>(
        actual_sample_rate_, expected_sample_rate_, lowpass_cutoff,
        lowpass_filter_width);
  } else {
    fprintf(stderr, "Current sample rate: %d\n", actual_sample_rate_);
  }

  err = snd_pcm_hw_params(capture_handle_, hw_params);
  if (err) {
    fprintf(stderr, "Failed to set hw params: %s\n", snd_strerror(err));
    exit(-1);
  }

  err = snd_pcm_prepare(capture_handle_);
  if (err) {
    fprintf(stderr, "Failed to prepare for recording: %s\n", snd_strerror(err));
    exit(-1);
  }

  fprintf(stderr, "Recording started!\n");
}

Alsa::~Alsa() { snd_pcm_close(capture_handle_); }

const std::vector<float> &Alsa::Read(int32_t num_samples) {
  samples_.resize(num_samples * actual_channel_count_);

  // count is in frames. Each frame contains actual_channel_count_ samples
  int32_t count = snd_pcm_readi(capture_handle_, samples_.data(), num_samples);
  if (count == -EPIPE) {
    static int32_t n = 0;
    if (++n > 5) {
      fprintf(
          stderr,
          "Too many overruns. It is very likely that the RTF on your board is "
          "larger than 1. Please use ./bin/sherpa-mnn to compute the RTF.\n");
      exit(-1);
    }
    fprintf(stderr, "XRUN.\n");
    snd_pcm_prepare(capture_handle_);

    static std::vector<float> tmp;
    return tmp;
  } else if (count < 0) {
    fprintf(stderr, "Can't read PCM device: %s\n", snd_strerror(count));
    exit(-1);
  }

  samples_.resize(count * actual_channel_count_);

  ToFloat(samples_, actual_channel_count_, &samples1_);

  if (!resampler_) {
    return samples1_;
  }

  resampler_->Resample(samples1_.data(), samples_.size(), false, &samples2_);
  return samples2_;
}

}  // namespace sherpa_mnn

#endif
