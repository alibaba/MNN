// sherpa-mnn/csrc/offline-speech-denoiser-gtcrn-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_IMPL_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/istft.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "sherpa-mnn/csrc/offline-speech-denoiser-gtcrn-model.h"
#include "sherpa-mnn/csrc/offline-speech-denoiser-impl.h"
#include "sherpa-mnn/csrc/offline-speech-denoiser.h"
#include "sherpa-mnn/csrc/resample.h"

namespace sherpa_mnn {

class OfflineSpeechDenoiserGtcrnImpl : public OfflineSpeechDenoiserImpl {
 public:
  explicit OfflineSpeechDenoiserGtcrnImpl(
      const OfflineSpeechDenoiserConfig &config)
      : model_(config.model) {}

  template <typename Manager>
  OfflineSpeechDenoiserGtcrnImpl(Manager *mgr,
                                 const OfflineSpeechDenoiserConfig &config)
      : model_(mgr, config.model) {}

  DenoisedAudio Run(const float *samples, int32_t n,
                    int32_t sample_rate) const override {
    const auto &meta = model_.GetMetaData();

    std::vector<float> tmp;
    auto p = samples;

    if (sample_rate != meta.sample_rate) {
      SHERPA_ONNX_LOGE(
          "Creating a resampler:\n"
          "   in_sample_rate: %d\n"
          "   output_sample_rate: %d\n",
          sample_rate, meta.sample_rate);

      float min_freq = std::min<int32_t>(sample_rate, meta.sample_rate);
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;

      int32_t lowpass_filter_width = 6;
      auto resampler = std::make_unique<LinearResample>(
          sample_rate, meta.sample_rate, lowpass_cutoff, lowpass_filter_width);
      resampler->Resample(samples, n, true, &tmp);
      p = tmp.data();
      n = tmp.size();
    }

    knf::StftConfig stft_config;
    stft_config.n_fft = meta.n_fft;
    stft_config.hop_length = meta.hop_length;
    stft_config.win_length = meta.window_length;
    stft_config.window_type = meta.window_type;
    if (stft_config.window_type == "hann_sqrt") {
      auto window = knf::GetWindow("hann", stft_config.win_length);
      for (auto &w : window) {
        w = std::sqrt(w);
      }
      stft_config.window = std::move(window);
    }

    knf::Stft stft(stft_config);
    knf::StftResult stft_result = stft.Compute(p, n);

    auto states = model_.GetInitStates();
    OfflineSpeechDenoiserGtcrnModel::States next_states;

    knf::StftResult enhanced_stft_result;
    enhanced_stft_result.num_frames = stft_result.num_frames;
    for (int32_t i = 0; i < stft_result.num_frames; ++i) {
      auto p = Process(stft_result, i, std::move(states), &next_states);
      states = std::move(next_states);

      enhanced_stft_result.real.insert(enhanced_stft_result.real.end(),
                                       p.first.begin(), p.first.end());
      enhanced_stft_result.imag.insert(enhanced_stft_result.imag.end(),
                                       p.second.begin(), p.second.end());
    }

    knf::IStft istft(stft_config);

    DenoisedAudio denoised_audio;
    denoised_audio.sample_rate = meta.sample_rate;
    denoised_audio.samples = istft.Compute(enhanced_stft_result);
    return denoised_audio;
  }

  int32_t GetSampleRate() const override {
    return model_.GetMetaData().sample_rate;
  }

 private:
  std::pair<std::vector<float>, std::vector<float>> Process(
      const knf::StftResult &stft_result, int32_t frame_index,
      OfflineSpeechDenoiserGtcrnModel::States states,
      OfflineSpeechDenoiserGtcrnModel::States *next_states) const {
    const auto &meta = model_.GetMetaData();
    int32_t n_fft = meta.n_fft;
    std::vector<float> x((n_fft / 2 + 1) * 2);

    const float *p_real =
        stft_result.real.data() + frame_index * (n_fft / 2 + 1);
    const float *p_imag =
        stft_result.imag.data() + frame_index * (n_fft / 2 + 1);

    for (int32_t i = 0; i < n_fft / 2 + 1; ++i) {
      x[2 * i] = p_real[i];
      x[2 * i + 1] = p_imag[i];
    }
    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 4> x_shape{1, n_fft / 2 + 1, 1, 2};
    MNN::Express::VARP x_tensor = MNNUtilsCreateTensor(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());

    MNN::Express::VARP output{nullptr};
    std::tie(output, *next_states) =
        model_.Run(std::move(x_tensor), std::move(states));

    std::vector<float> real(n_fft / 2 + 1);
    std::vector<float> imag(n_fft / 2 + 1);
    const auto *p = output->readMap<float>();
    for (int32_t i = 0; i < n_fft / 2 + 1; ++i) {
      real[i] = p[2 * i];
      imag[i] = p[2 * i + 1];
    }

    return {std::move(real), std::move(imag)};
  }

 private:
  OfflineSpeechDenoiserGtcrnModel model_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_IMPL_H_
