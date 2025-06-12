// sherpa-mnn/csrc/on-rnn-lm.cc
//
// Copyright (c)  2023  Pingfeng Luo
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-rnn-lm.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/session.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

class OnlineRnnLM::Impl {
 public:
  explicit Impl(const OnlineLMConfig &config)
      : config_(config),
        sess_opts_{GetSessionOptions(config)},
        allocator_{} {
    Init(config);
  }

  // shallow fusion scoring function
  void ComputeLMScoreSF(float scale, Hypothesis *hyp) {
    if (hyp->nn_lm_states.empty()) {
      auto init_states = GetInitStatesSF();
      hyp->nn_lm_scores.value = std::move(init_states.first);
      hyp->nn_lm_states = Convert(std::move(init_states.second));
    }

    // get lm score for cur token given the hyp->ys[:-1] and save to lm_log_prob
    const float *nn_lm_scores = hyp->nn_lm_scores.value->readMap<float>();
    hyp->lm_log_prob += nn_lm_scores[hyp->ys.back()] * scale;

    // get lm scores for next tokens given the hyp->ys[:] and save to
    // nn_lm_scores
    std::array<int, 2> x_shape{1, 1};
    MNN::Express::VARP x = MNNUtilsCreateTensor<int>(allocator_, x_shape.data(),
                                                     x_shape.size());
    *x->writeMap<int>() = hyp->ys.back();
    auto lm_out = ScoreToken(std::move(x), Convert(hyp->nn_lm_states));
    hyp->nn_lm_scores.value = std::move(lm_out.first);
    hyp->nn_lm_states = Convert(std::move(lm_out.second));
  }

  // classic rescore function
  void ComputeLMScore(float scale, int32_t context_size,
                      std::vector<Hypotheses> *hyps) {
    MNNAllocator* allocator;

    for (auto &hyp : *hyps) {
      for (auto &h_m : hyp) {
        auto &h = h_m.second;
        auto &ys = h.ys;
        const int32_t token_num_in_chunk =
            ys.size() - context_size - h.cur_scored_pos - 1;

        if (token_num_in_chunk < 1) {
          continue;
        }

        if (h.nn_lm_states.empty()) {
          h.nn_lm_states = Convert(GetInitStates());
        }

        if (token_num_in_chunk >= h.lm_rescore_min_chunk) {
          std::array<int, 2> x_shape{1, token_num_in_chunk};

          MNN::Express::VARP x = MNNUtilsCreateTensor<int>(
              allocator, x_shape.data(), x_shape.size());
          int *p_x = x->writeMap<int>();
          std::copy(ys.begin() + context_size + h.cur_scored_pos, ys.end() - 1,
                    p_x);

          // streaming forward by NN LM
          auto out =
              ScoreToken(std::move(x), Convert(std::move(h.nn_lm_states)));

          // update NN LM score in hyp
          const float *p_nll = out.first->readMap<float>();
          h.lm_log_prob = -scale * (*p_nll);

          // update NN LM states in hyp
          h.nn_lm_states = Convert(std::move(out.second));

          h.cur_scored_pos += token_num_in_chunk;
        }
      }
    }
  }

  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> ScoreToken(
      MNN::Express::VARP x, std::vector<MNN::Express::VARP> states) {
    std::vector<MNN::Express::VARP> inputs = {std::move(x), std::move(states[0]),
                                        std::move(states[1])};

    auto out =
        sess_->onForward(inputs);

    std::vector<MNN::Express::VARP> next_states;
    next_states.reserve(2);
    next_states.push_back(std::move(out[1]));
    next_states.push_back(std::move(out[2]));

    return {std::move(out[0]), std::move(next_states)};
  }

  // get init states for shallow fusion
  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> GetInitStatesSF() {
    std::vector<MNN::Express::VARP> ans;
    ans.reserve(init_states_.size());
    for (auto &s : init_states_) {
      ans.emplace_back(View(s));
    }
    return {View(init_scores_.value), std::move(ans)};
  }

  // get init states for classic rescore
  std::vector<MNN::Express::VARP> GetInitStates() {
    std::vector<MNN::Express::VARP> ans;
    ans.reserve(init_states_.size());

    for (const auto &s : init_states_) {
      ans.emplace_back(Clone(allocator_, s));
    }

    return ans;
  }

 private:
  void Init(const OnlineLMConfig &config) {
    auto buf = ReadFile(config_.model);

    sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {},  (const uint8_t*)buf.data(), buf.size(),
                                           sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    MNNMeta meta_data = sess_->getInfo()->metaData;
    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(rnn_num_layers_, "num_layers");
    SHERPA_ONNX_READ_META_DATA(rnn_hidden_size_, "hidden_size");
    SHERPA_ONNX_READ_META_DATA(sos_id_, "sos_id");

    ComputeInitStates();
  }

  void ComputeInitStates() {
    constexpr int32_t kBatchSize = 1;
    std::array<int, 3> h_shape{rnn_num_layers_, kBatchSize,
                                   rnn_hidden_size_};
    std::array<int, 3> c_shape{rnn_num_layers_, kBatchSize,
                                   rnn_hidden_size_};
    MNN::Express::VARP h = MNNUtilsCreateTensor<float>(allocator_, h_shape.data(),
                                                   h_shape.size());
    MNN::Express::VARP c = MNNUtilsCreateTensor<float>(allocator_, c_shape.data(),
                                                   c_shape.size());
    Fill<float>(h, 0);
    Fill<float>(c, 0);
    std::array<int, 2> x_shape{1, 1};
    MNN::Express::VARP x = MNNUtilsCreateTensor<int>(allocator_, x_shape.data(),
                                                     x_shape.size());
    *x->writeMap<int>() = sos_id_;

    std::vector<MNN::Express::VARP> states;
    states.push_back(std::move(h));
    states.push_back(std::move(c));
    auto pair = ScoreToken(std::move(x), std::move(states));

    init_scores_.value = std::move(pair.first);  // only used during
                                                 // shallow fusion
    init_states_ = std::move(pair.second);
  }

 private:
  OnlineLMConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  CopyableOrtValue init_scores_;
  std::vector<MNN::Express::VARP> init_states_;

  int32_t rnn_num_layers_ = 2;
  int32_t rnn_hidden_size_ = 512;
  int32_t sos_id_ = 1;
};

OnlineRnnLM::OnlineRnnLM(const OnlineLMConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OnlineRnnLM::~OnlineRnnLM() = default;

// classic rescore state init
std::vector<MNN::Express::VARP> OnlineRnnLM::GetInitStates() {
  return impl_->GetInitStates();
}

// shallow fusion state init
std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> OnlineRnnLM::GetInitStatesSF() {
  return impl_->GetInitStatesSF();
}

std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> OnlineRnnLM::ScoreToken(
    MNN::Express::VARP x, std::vector<MNN::Express::VARP> states) {
  return impl_->ScoreToken(std::move(x), std::move(states));
}

// classic rescore scores
void OnlineRnnLM::ComputeLMScore(float scale, int32_t context_size,
                                 std::vector<Hypotheses> *hyps) {
  return impl_->ComputeLMScore(scale, context_size, hyps);
}

// shallow fusion scores
void OnlineRnnLM::ComputeLMScoreSF(float scale, Hypothesis *hyp) {
  return impl_->ComputeLMScoreSF(scale, hyp);
}

}  // namespace sherpa_mnn
