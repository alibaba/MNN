// sherpa-mnn/csrc/context-graph.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/context-graph.h"

#include <algorithm>
#include <cassert>
#include <queue>
#include <string>
#include <tuple>
#include <utility>

#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {
void ContextGraph::Build(const std::vector<std::vector<int32_t>> &token_ids,
                         const std::vector<float> &scores,
                         const std::vector<std::string> &phrases,
                         const std::vector<float> &ac_thresholds) const {
  if (!scores.empty()) {
    SHERPA_ONNX_CHECK_EQ(token_ids.size(), scores.size());
  }
  if (!phrases.empty()) {
    SHERPA_ONNX_CHECK_EQ(token_ids.size(), phrases.size());
  }
  if (!ac_thresholds.empty()) {
    SHERPA_ONNX_CHECK_EQ(token_ids.size(), ac_thresholds.size());
  }
  for (int32_t i = 0; i < static_cast<int32_t>(token_ids.size()); ++i) {
    auto node = root_.get();
    float score = scores.empty() ? 0.0f : scores[i];
    score = score == 0.0f ? context_score_ : score;
    float ac_threshold = ac_thresholds.empty() ? 0.0f : ac_thresholds[i];
    ac_threshold = ac_threshold == 0.0f ? ac_threshold_ : ac_threshold;
    std::string phrase = phrases.empty() ? std::string() : phrases[i];

    for (int32_t j = 0; j < static_cast<int32_t>(token_ids[i].size()); ++j) {
      int32_t token = token_ids[i][j];
      if (0 == node->next.count(token)) {
        bool is_end = j == (static_cast<int32_t>(token_ids[i].size()) - 1);
        node->next[token] = std::make_unique<ContextState>(
            token, score, node->node_score + score,
            is_end ? node->node_score + score : 0, j + 1,
            is_end ? ac_threshold : 0.0f, is_end,
            is_end ? phrase : std::string());
      } else {
        float token_score = std::max(score, node->next[token]->token_score);
        node->next[token]->token_score = token_score;
        float node_score = node->node_score + token_score;
        node->next[token]->node_score = node_score;
        bool is_end = (j == static_cast<int32_t>(token_ids[i].size()) - 1) ||
                      node->next[token]->is_end;
        node->next[token]->output_score = is_end ? node_score : 0.0f;
        node->next[token]->is_end = is_end;
        if (j == static_cast<int32_t>(token_ids[i].size()) - 1) {
          node->next[token]->phrase = phrase;
          node->next[token]->ac_threshold = ac_threshold;
        }
      }
      node = node->next[token].get();
    }
  }
  FillFailOutput();
}

std::tuple<float, const ContextState *, const ContextState *>
ContextGraph::ForwardOneStep(const ContextState *state, int32_t token,
                             bool strict_mode /*= true*/) const {
  const ContextState *node = nullptr;
  float score = 0;
  if (1 == state->next.count(token)) {
    node = state->next.at(token).get();
    score = node->token_score;
  } else {
    node = state->fail;
    while (0 == node->next.count(token)) {
      node = node->fail;
      if (-1 == node->token) break;  // root
    }
    if (1 == node->next.count(token)) {
      node = node->next.at(token).get();
    }
    score = node->node_score - state->node_score;
  }

  if (!node) {
    SHERPA_ONNX_LOGE("Some bad things happened.");
    exit(-1);
  }

  const ContextState *matched_node =
      node->is_end ? node : (node->output != nullptr ? node->output : nullptr);

  if (!strict_mode && node->output_score != 0) {
    SHERPA_ONNX_CHECK(nullptr != matched_node);
    float output_score =
        node->is_end ? node->node_score
                     : (node->output != nullptr ? node->output->node_score
                                                : node->node_score);
    return std::make_tuple(score + output_score - node->node_score, root_.get(),
                           matched_node);
  }
  return std::make_tuple(score + node->output_score, node, matched_node);
}

std::pair<float, const ContextState *> ContextGraph::Finalize(
    const ContextState *state) const {
  float score = -state->node_score;
  return std::make_pair(score, root_.get());
}

std::pair<bool, const ContextState *> ContextGraph::IsMatched(
    const ContextState *state) const {
  bool status = false;
  const ContextState *node = nullptr;
  if (state->is_end) {
    status = true;
    node = state;
  } else {
    if (state->output != nullptr) {
      status = true;
      node = state->output;
    }
  }
  return std::make_pair(status, node);
}

void ContextGraph::FillFailOutput() const {
  std::queue<const ContextState *> node_queue;
  for (auto &kv : root_->next) {
    kv.second->fail = root_.get();
    node_queue.push(kv.second.get());
  }
  while (!node_queue.empty()) {
    auto current_node = node_queue.front();
    node_queue.pop();
    for (auto &kv : current_node->next) {
      auto fail = current_node->fail;
      if (1 == fail->next.count(kv.first)) {
        fail = fail->next.at(kv.first).get();
      } else {
        fail = fail->fail;
        while (0 == fail->next.count(kv.first)) {
          fail = fail->fail;
          if (-1 == fail->token) break;
        }
        if (1 == fail->next.count(kv.first))
          fail = fail->next.at(kv.first).get();
      }
      kv.second->fail = fail;
      // fill the output arc
      auto output = fail;
      while (!output->is_end) {
        output = output->fail;
        if (-1 == output->token) {
          output = nullptr;
          break;
        }
      }
      kv.second->output = output;
      kv.second->output_score += output == nullptr ? 0 : output->output_score;
      node_queue.push(kv.second.get());
    }
  }
}
}  // namespace sherpa_mnn
