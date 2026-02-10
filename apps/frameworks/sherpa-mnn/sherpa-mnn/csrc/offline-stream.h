// sherpa-mnn/csrc/offline-stream.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
#define SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/context-graph.h"
#include "sherpa-mnn/csrc/features.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct OfflineRecognitionResult {
  // Recognition results.
  // For English, it consists of space separated words.
  // For Chinese, it consists of Chinese words without spaces.
  std::string text;

  // Decoded results at the token level.
  // For instance, for BPE-based models it consists of a list of BPE tokens.
  std::vector<std::string> tokens;

  std::string lang;

  // emotion target of the audio.
  std::string emotion;

  // event target of the audio.
  std::string event;

  /// timestamps.size() == tokens.size()
  /// timestamps[i] records the time in seconds when tokens[i] is decoded.
  std::vector<float> timestamps;

  std::vector<int32_t> words;

  std::string AsJsonString() const;
};

struct WhisperTag {
  int32_t dim = 80;
};

struct CEDTag {};

// It uses a neural network model, a preprocessor, to convert
// audio samples to features
struct MoonshineTag {};

class OfflineStream {
 public:
  explicit OfflineStream(const FeatureExtractorConfig &config = {},
                         ContextGraphPtr context_graph = {});

  explicit OfflineStream(WhisperTag tag);
  explicit OfflineStream(CEDTag tag);
  explicit OfflineStream(MoonshineTag tag);
  ~OfflineStream();

  /**
     @param sampling_rate The sampling_rate of the input waveform. If it does
                          not equal to  config.sampling_rate, we will do
                          resampling inside.
     @param waveform Pointer to a 1-D array of size n. It must be normalized to
                     the range [-1, 1].
     @param n Number of entries in waveform

     Caution: You can only invoke this function once so you have to input
              all the samples at once
   */
  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const;

  /// Return feature dim of this extractor.
  ///
  /// Note: if it is Moonshine, then it returns the number of audio samples
  /// currently received.
  int32_t FeatureDim() const;

  // Get all the feature frames of this stream in a 1-D array, which is
  // flattened from a 2-D array of shape (num_frames, feat_dim).
  std::vector<float> GetFrames() const;

  /** Set the recognition result for this stream. */
  void SetResult(const OfflineRecognitionResult &r);

  /** Get the recognition result of this stream */
  const OfflineRecognitionResult &GetResult() const;

  /** Get the ContextGraph of this stream */
  const ContextGraphPtr &GetContextGraph() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
