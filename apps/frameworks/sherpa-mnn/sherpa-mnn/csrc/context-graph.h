// sherpa-mnn/csrc/context-graph.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_CONTEXT_GRAPH_H_
#define SHERPA_ONNX_CSRC_CONTEXT_GRAPH_H_

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/log.h"

namespace sherpa_mnn {

class ContextGraph;
using ContextGraphPtr = std::shared_ptr<ContextGraph>;

struct ContextState {
  int32_t token;
  float token_score;
  float node_score;
  float output_score;
  int32_t level;
  float ac_threshold;
  bool is_end;
  std::string phrase;
  std::unordered_map<int32_t, std::unique_ptr<ContextState>> next;
  const ContextState *fail = nullptr;
  const ContextState *output = nullptr;

  ContextState() = default;
  ContextState(int32_t token, float token_score, float node_score,
               float output_score, int32_t level = 0, float ac_threshold = 0.0f,
               bool is_end = false, const std::string &phrase = {})
      : token(token),
        token_score(token_score),
        node_score(node_score),
        output_score(output_score),
        level(level),
        ac_threshold(ac_threshold),
        is_end(is_end),
        phrase(phrase) {}
};

class ContextGraph {
 public:
  ContextGraph() = default;
  ContextGraph(const std::vector<std::vector<int32_t>> &token_ids,
               float context_score, float ac_threshold,
               const std::vector<float> &scores = {},
               const std::vector<std::string> &phrases = {},
               const std::vector<float> &ac_thresholds = {})
      : context_score_(context_score), ac_threshold_(ac_threshold) {
    root_ = std::make_unique<ContextState>(-1, 0, 0, 0);
    root_->fail = root_.get();
    Build(token_ids, scores, phrases, ac_thresholds);
  }

  ContextGraph(const std::vector<std::vector<int32_t>> &token_ids,
               float context_score, const std::vector<float> &scores = {})
      : ContextGraph(token_ids, context_score, 0.0f, scores,
                     std::vector<std::string>(), std::vector<float>()) {}

  std::tuple<float, const ContextState *, const ContextState *> ForwardOneStep(
      const ContextState *state, int32_t token_id,
      bool strict_mode = true) const;

  std::pair<bool, const ContextState *> IsMatched(
      const ContextState *state) const;

  std::pair<float, const ContextState *> Finalize(
      const ContextState *state) const;

  const ContextState *Root() const { return root_.get(); }

 private:
  float context_score_;
  float ac_threshold_;
  std::unique_ptr<ContextState> root_;
  void Build(const std::vector<std::vector<int32_t>> &token_ids,
             const std::vector<float> &scores,
             const std::vector<std::string> &phrases,
             const std::vector<float> &ac_thresholds) const;
  void FillFailOutput() const;
};

}  // namespace sherpa_mnn
#endif  // SHERPA_ONNX_CSRC_CONTEXT_GRAPH_H_
