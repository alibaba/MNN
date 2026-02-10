// sherpa-mnn/csrc/online-punctuation-cnn-bilstm-impl.h
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#ifndef SHERPA_ONNX_CSRC_ONLINE_PUNCTUATION_CNN_BILSTM_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_PUNCTUATION_CNN_BILSTM_IMPL_H_

#include <math.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include <chrono>  // NOLINT

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/math.h"
#include "sherpa-mnn/csrc/online-cnn-bilstm-model-meta-data.h"
#include "sherpa-mnn/csrc/online-cnn-bilstm-model.h"
#include "sherpa-mnn/csrc/online-punctuation-impl.h"
#include "sherpa-mnn/csrc/online-punctuation.h"
#include "sherpa-mnn/csrc/text-utils.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_mnn {

static const int32_t kMaxSeqLen = 200;

class OnlinePunctuationCNNBiLSTMImpl : public OnlinePunctuationImpl {
 public:
  explicit OnlinePunctuationCNNBiLSTMImpl(const OnlinePunctuationConfig &config)
      : config_(config), model_(config.model) {
    if (!config_.model.bpe_vocab.empty()) {
      bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(
          config_.model.bpe_vocab);
    }
  }

#if __ANDROID_API__ >= 9
  OnlinePunctuationCNNBiLSTMImpl(AAssetManager *mgr,
                                 const OnlinePunctuationConfig &config)
      : config_(config), model_(mgr, config.model) {
    if (!config_.model.bpe_vocab.empty()) {
      auto buf = ReadFile(mgr, config_.model.bpe_vocab);
      std::istringstream iss(std::string(buf.begin(), buf.end()));
      bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(iss);
    }
  }
#endif

  std::string AddPunctuationWithCase(const std::string &text) const override {
    if (text.empty()) {
      return {};
    }

    std::vector<int32_t> tokens_list;     // N * kMaxSeqLen
    std::vector<int32_t> valids_list;     // N * kMaxSeqLen
    std::vector<int32_t> label_len_list;  // N

    EncodeSentences(text, tokens_list, valids_list, label_len_list);

    const auto &meta_data = model_.metaData();

    auto memory_info =
        (MNNAllocator*)(nullptr);

    int32_t n = label_len_list.size();

    std::array<int, 2> token_ids_shape = {n, kMaxSeqLen};
    MNN::Express::VARP token_ids = MNNUtilsCreateTensor(
        memory_info, tokens_list.data(), tokens_list.size(),
        token_ids_shape.data(), token_ids_shape.size());

    std::array<int, 2> valid_ids_shape = {n, kMaxSeqLen};
    MNN::Express::VARP valid_ids = MNNUtilsCreateTensor(
        memory_info, valids_list.data(), valids_list.size(),
        valid_ids_shape.data(), valid_ids_shape.size());

    std::array<int, 1> label_len_shape = {n};
    MNN::Express::VARP label_len = MNNUtilsCreateTensor(
        memory_info, label_len_list.data(), label_len_list.size(),
        label_len_shape.data(), label_len_shape.size());

    auto pair = model_.Forward(std::move(token_ids), std::move(valid_ids),
                               std::move(label_len));

    std::vector<int32_t> case_pred;
    std::vector<int32_t> punct_pred;
    const float *active_case_logits = pair.first->readMap<float>();
    const float *active_punct_logits = pair.second->readMap<float>();
    std::vector<int> case_logits_shape =
        pair.first->getInfo()->dim;

    for (int32_t i = 0; i < case_logits_shape[0]; ++i) {
      const float *p_cur_case = active_case_logits + i * meta_data.num_cases;
      auto index_case = static_cast<int32_t>(std::distance(
          p_cur_case,
          std::max_element(p_cur_case, p_cur_case + meta_data.num_cases)));
      case_pred.push_back(index_case);

      const float *p_cur_punct =
          active_punct_logits + i * meta_data.num_punctuations;
      auto index_punct = static_cast<int32_t>(std::distance(
          p_cur_punct,
          std::max_element(p_cur_punct,
                           p_cur_punct + meta_data.num_punctuations)));
      punct_pred.push_back(index_punct);
    }

    std::string ans = DecodeSentences(text, case_pred, punct_pred);

    return ans;
  }

 private:
  void EncodeSentences(const std::string &text,
                       std::vector<int32_t> &tokens_list,             // NOLINT
                       std::vector<int32_t> &valids_list,             // NOLINT
                       std::vector<int32_t> &label_len_list) const {  // NOLINT
    std::vector<int32_t> tokens;
    std::vector<int32_t> valids;
    int32_t label_len = 0;

    tokens.push_back(1);  // hardcode 1 now, 1 - <s>
    valids.push_back(1);

    std::stringstream ss(text);
    std::string word;
    while (ss >> word) {
      std::vector<int32_t> word_tokens;
      bpe_encoder_->Encode(word, &word_tokens);

      int32_t seq_len = tokens.size() + word_tokens.size();
      if (seq_len > kMaxSeqLen - 1) {
        tokens.push_back(2);  // hardcode 2 now, 2 - </s>
        valids.push_back(1);

        label_len = std::count(valids.begin(), valids.end(), 1);

        if (tokens.size() < kMaxSeqLen) {
          tokens.resize(kMaxSeqLen, 0);
          valids.resize(kMaxSeqLen, 0);
        }

        assert(tokens.size() == kMaxSeqLen);
        assert(valids.size() == kMaxSeqLen);

        tokens_list.insert(tokens_list.end(), tokens.begin(), tokens.end());
        valids_list.insert(valids_list.end(), valids.begin(), valids.end());
        label_len_list.push_back(label_len);

        std::vector<int32_t>().swap(tokens);
        std::vector<int32_t>().swap(valids);
        label_len = 0;
        tokens.push_back(1);  // hardcode 1 now, 1 - <s>
        valids.push_back(1);
      }

      tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
      valids.push_back(1);  // only the first sub word is valid
      int32_t remaining_size = static_cast<int32_t>(word_tokens.size()) - 1;
      if (remaining_size > 0) {
        int32_t valids_cur_size = static_cast<int32_t>(valids.size());
        valids.resize(valids_cur_size + remaining_size, 0);
      }
    }

    if (tokens.size() > 0) {
      tokens.push_back(2);  // hardcode 2 now, 2 - </s>
      valids.push_back(1);

      label_len = std::count(valids.begin(), valids.end(), 1);

      if (tokens.size() < kMaxSeqLen) {
        tokens.resize(kMaxSeqLen, 0);
        valids.resize(kMaxSeqLen, 0);
      }

      assert(tokens.size() == kMaxSeqLen);
      assert(valids.size() == kMaxSeqLen);

      tokens_list.insert(tokens_list.end(), tokens.begin(), tokens.end());
      valids_list.insert(valids_list.end(), valids.begin(), valids.end());
      label_len_list.push_back(label_len);
    }
  }

  std::string DecodeSentences(const std::string &raw_text,
                              const std::vector<int32_t> &case_pred,
                              const std::vector<int32_t> &punct_pred) const {
    std::string result_text;
    std::istringstream iss(raw_text);
    std::vector<std::string> words;
    std::string word;

    while (iss >> word) {
      words.emplace_back(word);
    }

    assert(words.size() == case_pred.size());
    assert(words.size() == punct_pred.size());

    for (int32_t i = 0; i < words.size(); ++i) {
      std::string prefix = ((i != 0) ? " " : "");
      result_text += prefix;
      switch (case_pred[i]) {
        case 1:  // upper
        {
          std::transform(words[i].begin(), words[i].end(), words[i].begin(),
                         [](auto c) { return std::toupper(c); });
          result_text += words[i];
          break;
        }
        case 2:  // cap
        {
          words[i][0] = std::toupper(words[i][0]);
          result_text += words[i];
          break;
        }
        case 3:  // mix case
        {
          // TODO(frankyoujian):
          // Need to add a map containing supported mix case words so that we
          // can fetch the predicted word from the map e.g. mcdonald's ->
          // McDonald's
          result_text += words[i];
          break;
        }
        default: {
          result_text += words[i];
          break;
        }
      }

      std::string suffix;
      switch (punct_pred[i]) {
        case 1:  // comma
        {
          suffix = ",";
          break;
        }
        case 2:  // period
        {
          suffix = ".";
          break;
        }
        case 3:  // question
        {
          suffix = "?";
          break;
        }
        default:
          break;
      }

      result_text += suffix;
    }

    return result_text;
  }

 private:
  OnlinePunctuationConfig config_;
  OnlineCNNBiLSTMModel model_;
  std::unique_ptr<ssentencepiece::Ssentencepiece> bpe_encoder_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_PUNCTUATION_CNN_BILSTM_IMPL_H_
