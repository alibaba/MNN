// sherpa-mnn/csrc/endpoint.cc
//
// Copyright (c)  2022  (authors: Pingfeng Luo)
//                2022-2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/endpoint.h"

#include <string>

#include "sherpa-mnn/csrc/log.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

static bool RuleActivated(const EndpointRule &rule,
                          const std::string &rule_name, float trailing_silence,
                          float utterance_length) {
  bool contain_nonsilence = utterance_length > trailing_silence;
  bool ans = (contain_nonsilence || !rule.must_contain_nonsilence) &&
             trailing_silence >= rule.min_trailing_silence &&
             utterance_length >= rule.min_utterance_length;
  if (ans) {
    SHERPA_ONNX_LOG(DEBUG) << "Endpointing rule " << rule_name << " activated: "
                           << (contain_nonsilence ? "true" : "false") << ','
                           << trailing_silence << ',' << utterance_length;
  }
  return ans;
}

static void RegisterEndpointRule(ParseOptions *po, EndpointRule *rule,
                                 const std::string &rule_name) {
  po->Register(
      rule_name + "-must-contain-nonsilence", &rule->must_contain_nonsilence,
      "If True, for this endpointing " + rule_name +
          " to apply there must be nonsilence in the best-path traceback. "
          "For decoding, a non-blank token is considered as non-silence");
  po->Register(rule_name + "-min-trailing-silence", &rule->min_trailing_silence,
               "This endpointing " + rule_name +
                   " requires duration of trailing silence in seconds) to "
                   "be >= this value.");
  po->Register(rule_name + "-min-utterance-length", &rule->min_utterance_length,
               "This endpointing " + rule_name +
                   " requires utterance-length (in seconds) to be >= this "
                   "value.");
}

std::string EndpointRule::ToString() const {
  std::ostringstream os;

  os << "EndpointRule(";
  os << "must_contain_nonsilence="
     << (must_contain_nonsilence ? "True" : "False") << ", ";
  os << "min_trailing_silence=" << min_trailing_silence << ", ";
  os << "min_utterance_length=" << min_utterance_length << ")";

  return os.str();
}

void EndpointConfig::Register(ParseOptions *po) {
  RegisterEndpointRule(po, &rule1, "rule1");
  RegisterEndpointRule(po, &rule2, "rule2");
  RegisterEndpointRule(po, &rule3, "rule3");
}

std::string EndpointConfig::ToString() const {
  std::ostringstream os;

  os << "EndpointConfig(";
  os << "rule1=" << rule1.ToString() << ", ";
  os << "rule2=" << rule2.ToString() << ", ";
  os << "rule3=" << rule3.ToString() << ")";

  return os.str();
}

bool Endpoint::IsEndpoint(int32_t num_frames_decoded,
                          int32_t trailing_silence_frames,
                          float frame_shift_in_seconds) const {
  float utterance_length =
      static_cast<float>(num_frames_decoded) * frame_shift_in_seconds;

  float trailing_silence =
      static_cast<float>(trailing_silence_frames) * frame_shift_in_seconds;

  if (RuleActivated(config_.rule1, "rule1", trailing_silence,
                    utterance_length) ||
      RuleActivated(config_.rule2, "rule2", trailing_silence,
                    utterance_length) ||
      RuleActivated(config_.rule3, "rule3", trailing_silence,
                    utterance_length)) {
    return true;
  }
  return false;
}

}  // namespace sherpa_mnn
