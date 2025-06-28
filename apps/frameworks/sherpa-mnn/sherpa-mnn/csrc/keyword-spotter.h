// sherpa-mnn/csrc/keyword-spotter.h
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_H_
#define SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/features.h"
#include "sherpa-mnn/csrc/online-model-config.h"
#include "sherpa-mnn/csrc/online-stream.h"
#include "sherpa-mnn/csrc/online-transducer-model-config.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct KeywordResult {
  /// The triggered keyword.
  /// For English, it consists of space separated words.
  /// For Chinese, it consists of Chinese words without spaces.
  /// Example 1: "hello world"
  /// Example 2: "你好世界"
  std::string keyword;

  /// Decoded results at the token level.
  /// For instance, for BPE-based models it consists of a list of BPE tokens.
  std::vector<std::string> tokens;

  /// timestamps.size() == tokens.size()
  /// timestamps[i] records the time in seconds when tokens[i] is decoded.
  std::vector<float> timestamps;

  /// Starting time of this segment.
  /// When an endpoint is detected, it will change
  float start_time = 0;

  /** Return a json string.
   *
   * The returned string contains:
   *   {
   *     "keyword": "The triggered keyword",
   *     "tokens": [x, x, x],
   *     "timestamps": [x, x, x],
   *     "start_time": x,
   *   }
   */
  std::string AsJsonString() const;
};

struct KeywordSpotterConfig {
  FeatureExtractorConfig feat_config;
  OnlineModelConfig model_config;

  int32_t max_active_paths = 4;

  int32_t num_trailing_blanks = 1;

  float keywords_score = 1.0;

  float keywords_threshold = 0.25;

  std::string keywords_file;

  /// if keywords_buf is non-empty,
  /// the keywords will be loaded from the buffer instead of from the
  /// "keywrods_file"
  std::string keywords_buf;

  KeywordSpotterConfig() = default;

  KeywordSpotterConfig(const FeatureExtractorConfig &feat_config,
                       const OnlineModelConfig &model_config,
                       int32_t max_active_paths, int32_t num_trailing_blanks,
                       float keywords_score, float keywords_threshold,
                       const std::string &keywords_file)
      : feat_config(feat_config),
        model_config(model_config),
        max_active_paths(max_active_paths),
        num_trailing_blanks(num_trailing_blanks),
        keywords_score(keywords_score),
        keywords_threshold(keywords_threshold),
        keywords_file(keywords_file) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class KeywordSpotterImpl;

class KeywordSpotter {
 public:
  explicit KeywordSpotter(const KeywordSpotterConfig &config);

  template <typename Manager>
  KeywordSpotter(Manager *mgr, const KeywordSpotterConfig &config);

  ~KeywordSpotter();

  /** Create a stream for decoding.
   *
   */
  std::unique_ptr<OnlineStream> CreateStream() const;

  /** Create a stream for decoding.
   *
   *  @param The keywords for this string, it might contain several keywords,
   *         the keywords are separated by "/". In each of the keywords, there
   *         are cjkchars or bpes, the bpe/cjkchar are separated by space (" ").
   *         For example, keywords I LOVE YOU and HELLO WORLD, looks like:
   *
   *         "▁I ▁LOVE ▁YOU/▁HE LL O ▁WORLD"
   */
  std::unique_ptr<OnlineStream> CreateStream(const std::string &keywords) const;

  /**
   * Return true if the given stream has enough frames for decoding.
   * Return false otherwise
   */
  bool IsReady(OnlineStream *s) const;

  // Remember to call it after detecting a keyword
  void Reset(OnlineStream *s) const;

  /** Decode a single stream. */
  void DecodeStream(OnlineStream *s) const {
    OnlineStream *ss[1] = {s};
    DecodeStreams(ss, 1);
  }

  /** Decode multiple streams in parallel
   *
   * @param ss Pointer array containing streams to be decoded.
   * @param n Number of streams in `ss`.
   */
  void DecodeStreams(OnlineStream **ss, int32_t n) const;

  KeywordResult GetResult(OnlineStream *s) const;

 private:
  std::unique_ptr<KeywordSpotterImpl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_H_
