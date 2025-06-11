// sherpa-mnn/csrc/endpoint.h
//
// Copyright (c)  2022  (authors: Pingfeng Luo)
//                2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ENDPOINT_H_
#define SHERPA_ONNX_CSRC_ENDPOINT_H_

#include <string>
#include <vector>

namespace sherpa_mnn {

struct EndpointRule {
  // If True, for this endpointing rule to apply there must
  // be nonsilence in the best-path traceback.
  // For decoding, a non-blank token is considered as non-silence
  bool must_contain_nonsilence = true;
  // This endpointing rule requires duration of trailing silence
  // (in seconds) to be >= this value.
  float min_trailing_silence = 2.0;
  // This endpointing rule requires utterance-length (in seconds)
  // to be >= this value.
  float min_utterance_length = 0.0f;

  EndpointRule() = default;

  EndpointRule(bool must_contain_nonsilence, float min_trailing_silence,
               float min_utterance_length)
      : must_contain_nonsilence(must_contain_nonsilence),
        min_trailing_silence(min_trailing_silence),
        min_utterance_length(min_utterance_length) {}

  std::string ToString() const;
};

class ParseOptions;

struct EndpointConfig {
  // For default setting,
  // rule1 times out after 2.4 seconds of silence, even if we decoded nothing.
  // rule2 times out after 1.2 seconds of silence after decoding something.
  // rule3 times out after the utterance is 20 seconds long, regardless of
  // anything else.
  EndpointRule rule1;
  EndpointRule rule2;
  EndpointRule rule3;

  void Register(ParseOptions *po);

  EndpointConfig()
      : rule1{false, 2.4, 0}, rule2{true, 1.2, 0}, rule3{false, 0, 20} {}

  EndpointConfig(const EndpointRule &rule1, const EndpointRule &rule2,
                 const EndpointRule &rule3)
      : rule1(rule1), rule2(rule2), rule3(rule3) {}

  std::string ToString() const;
};

class Endpoint {
 public:
  explicit Endpoint(const EndpointConfig &config) : config_(config) {}

  /// This function returns true if this set of endpointing rules thinks we
  /// should terminate decoding.
  bool IsEndpoint(int32_t num_frames_decoded, int32_t trailing_silence_frames,
                  float frame_shift_in_seconds) const;

 private:
  EndpointConfig config_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ENDPOINT_H_
