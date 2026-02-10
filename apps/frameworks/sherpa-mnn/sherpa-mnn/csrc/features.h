// sherpa-mnn/csrc/features.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FEATURES_H_
#define SHERPA_ONNX_CSRC_FEATURES_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct FeatureExtractorConfig {
  // Sampling rate used by the feature extractor. If it is different from
  // the sampling rate of the input waveform, we will do resampling inside.
  int32_t sampling_rate = 16000;

  // num_mel_bins
  //
  // Note: for mfcc, this value is also for num_mel_bins.
  // The actual feature dimension is actuall num_ceps
  int32_t feature_dim = 80;

  // minimal frequency for Mel-filterbank, in Hz
  float low_freq = 20.0f;

  // maximal frequency of Mel-filterbank
  // in Hz; negative value is subtracted from Nyquist freq.:
  // i.e. for sampling_rate 16000 / 2 - 400 = 7600Hz
  //
  // Please see
  // https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/fbank.py#L27
  // and
  // https://github.com/k2-fsa/sherpa-mnn/issues/514
  float high_freq = -400.0f;

  // dithering constant, useful for signals with hard-zeroes in non-speech parts
  // this prevents large negative values in log-mel filterbanks
  //
  // In k2, audio samples are in range [-1..+1], in kaldi the range was
  // [-32k..+32k], so the value 0.00003 is equivalent to kaldi default 1.0
  //
  float dither = 0.0f;  // dithering disabled by default

  // Set internally by some models, e.g., paraformer sets it to false.
  // This parameter is not exposed to users from the commandline
  // If true, the feature extractor expects inputs to be normalized to
  // the range [-1, 1].
  // If false, we will multiply the inputs by 32768
  bool normalize_samples = true;

  bool snip_edges = false;
  float frame_shift_ms = 10.0f;   // in milliseconds.
  float frame_length_ms = 25.0f;  // in milliseconds.
  bool is_librosa = false;
  bool remove_dc_offset = true;       // Subtract mean of wave before FFT.
  float preemph_coeff = 0.97f;        // Preemphasis coefficient.
  std::string window_type = "povey";  // e.g. Hamming window

  // For models from NeMo
  // This option is not exposed and is set internally when loading models.
  // Possible values:
  // - per_feature
  // - all_features (not implemented yet)
  // - fixed_mean (not implemented)
  // - fixed_std (not implemented)
  // - or just leave it to empty
  // See
  // https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L59
  // for details
  std::string nemo_normalize_type;

  // for MFCC
  int32_t num_ceps = 13;
  bool use_energy = true;

  bool is_mfcc = false;

  std::string ToString() const;

  void Register(ParseOptions *po);
};

class FeatureExtractor {
 public:
  explicit FeatureExtractor(const FeatureExtractorConfig &config = {});
  ~FeatureExtractor();

  /**
     @param sampling_rate The sampling_rate of the input waveform. If it does
                          not equal to  config.sampling_rate, we will do
                          resampling inside.
     @param waveform Pointer to a 1-D array of size n. It must be normalized to
                     the range [-1, 1].
     @param n Number of entries in waveform
   */
  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const;

  /**
   * InputFinished() tells the class you won't be providing any
   * more waveform.  This will help flush out the last frame or two
   * of features, in the case where snip-edges == false; it also
   * affects the return value of IsLastFrame().
   */
  void InputFinished() const;

  int32_t NumFramesReady() const;

  /** Note: IsLastFrame() will only ever return true if you have called
   * InputFinished() (and this frame is the last frame).
   */
  bool IsLastFrame(int32_t frame) const;

  /** Get n frames starting from the given frame index.
   *
   * @param frame_index  The starting frame index
   * @param n  Number of frames to get.
   * @return Return a 2-D tensor of shape (n, feature_dim).
   *         which is flattened into a 1-D vector (flattened in row major)
   */
  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const;

  /// Return feature dim of this extractor
  int32_t FeatureDim() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_FEATURES_H_
