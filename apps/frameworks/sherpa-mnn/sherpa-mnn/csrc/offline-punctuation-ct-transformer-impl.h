// sherpa-mnn/csrc/offline-punctuation-ct-transformer-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_CT_TRANSFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_CT_TRANSFORMER_IMPL_H_

#include <math.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/math.h"
#include "sherpa-mnn/csrc/offline-ct-transformer-model.h"
#include "sherpa-mnn/csrc/offline-punctuation-impl.h"
#include "sherpa-mnn/csrc/offline-punctuation.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

class OfflinePunctuationCtTransformerImpl : public OfflinePunctuationImpl {
 public:
  explicit OfflinePunctuationCtTransformerImpl(
      const OfflinePunctuationConfig &config)
      : config_(config), model_(config.model) {}

#if __ANDROID_API__ >= 9
  OfflinePunctuationCtTransformerImpl(AAssetManager *mgr,
                                      const OfflinePunctuationConfig &config)
      : config_(config), model_(mgr, config.model) {}
#endif

  std::string AddPunctuation(const std::string &text) const override {
    if (text.empty()) {
      return {};
    }

    std::vector<std::string> tokens = SplitUtf8(text);
    std::vector<int32_t> token_ids;
    token_ids.reserve(tokens.size());

    const auto &meta_data = model_.metaData();

    for (const auto &t : tokens) {
      std::string token = ToLowerCase(t);
      if (meta_data.token2id.count(token)) {
        token_ids.push_back(meta_data.token2id.at(token));
      } else {
        token_ids.push_back(meta_data.unk_id);
      }
    }

    auto memory_info =
        (MNNAllocator*)(nullptr);

    int32_t segment_size = 20;
    int32_t max_len = 200;
    int32_t num_segments =
        ceil((static_cast<float>(token_ids.size()) + segment_size - 1) /
             segment_size);

    std::vector<int32_t> punctuations;
    int32_t last = -1;
    for (int32_t i = 0; i != num_segments; ++i) {
      int32_t this_start = i * segment_size;         // included
      int32_t this_end = this_start + segment_size;  // not included
      if (this_end > static_cast<int32_t>(token_ids.size())) {
        this_end = token_ids.size();
      }

      if (last != -1) {
        this_start = last;
      }
      // token_ids[this_start:this_end] is sent to the model

      std::array<int, 2> x_shape = {1, this_end - this_start};
      MNN::Express::VARP x =
          MNNUtilsCreateTensor(memory_info, token_ids.data() + this_start,
                                   x_shape[1], x_shape.data(), x_shape.size());

      int len_shape = 1;
      int32_t len = x_shape[1];
      MNN::Express::VARP x_len =
          MNNUtilsCreateTensor(memory_info, &len, 1, &len_shape, 1);

      MNN::Express::VARP out = model_.Forward(std::move(x), std::move(x_len));

      // [N, T, num_punctuations]
      std::vector<int> out_shape =
          out->getInfo()->dim;

      assert(out_shape[0] == 1);
      assert(out_shape[1] == len);
      assert(out_shape[2] == meta_data.num_punctuations);

      std::vector<int32_t> this_punctuations;
      this_punctuations.reserve(len);

      const float *p = out->readMap<float>();
      for (int32_t k = 0; k != len; ++k, p += meta_data.num_punctuations) {
        auto index = static_cast<int32_t>(std::distance(
            p, std::max_element(p, p + meta_data.num_punctuations)));
        this_punctuations.push_back(index);
      }  // for (int32_t k = 0; k != len; ++k, p += meta_data.num_punctuations)

      int32_t dot_index = -1;
      int32_t comma_index = -1;

      for (int32_t m = static_cast<int32_t>(this_punctuations.size()) - 2;
           m >= 1; --m) {
        int32_t punct_id = this_punctuations[m];

        if (punct_id == meta_data.dot_id || punct_id == meta_data.quest_id) {
          dot_index = m;
          break;
        }

        if (comma_index == -1 && punct_id == meta_data.comma_id) {
          comma_index = m;
        }
      }  // for (int32_t k = this_punctuations.size() - 1; k >= 1; --k)

      if (dot_index == -1 && len >= max_len && comma_index != -1) {
        dot_index = comma_index;
        this_punctuations[dot_index] = meta_data.dot_id;
      }

      if (dot_index == -1) {
        if (last == -1) {
          last = this_start;
        }

        if (i == num_segments - 1) {
          dot_index = static_cast<int32_t>(this_punctuations.size()) - 1;
        }
      } else {
        last = this_start + dot_index + 1;
      }

      if (dot_index != -1) {
        punctuations.insert(punctuations.end(), this_punctuations.begin(),
                            this_punctuations.begin() + (dot_index + 1));
      }
    }  // for (int32_t i = 0; i != num_segments; ++i)

    if (punctuations.empty()) {
      return text + meta_data.id2punct[meta_data.dot_id];
    }
    std::vector<std::string> words_punct;

    for (int32_t i = 0; i != static_cast<int32_t>(punctuations.size()); ++i) {
      if (i >= static_cast<int32_t>(tokens.size())) {
        break;
      }
      std::string &w = tokens[i];
      if (i > 0 && !(words_punct.back()[0] & 0x80) && !(w[0] & 0x80)) {
        words_punct.push_back(" ");
      }
      words_punct.push_back(std::move(w));

      if (punctuations[i] != meta_data.underline_id) {
        words_punct.push_back(meta_data.id2punct[punctuations[i]]);
      }
    }

    if (words_punct.back() == meta_data.id2punct[meta_data.comma_id] ||
        words_punct.back() == meta_data.id2punct[meta_data.pause_id]) {
      words_punct.back() = meta_data.id2punct[meta_data.dot_id];
    }

    if (words_punct.back() != meta_data.id2punct[meta_data.dot_id] &&
        words_punct.back() != meta_data.id2punct[meta_data.quest_id]) {
      words_punct.push_back(meta_data.id2punct[meta_data.dot_id]);
    }

    std::string ans;
    for (const auto &w : words_punct) {
      ans.append(w);
    }
    return ans;
  }

 private:
  OfflinePunctuationConfig config_;
  OfflineCtTransformerModel model_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_CT_TRANSFORMER_IMPL_H_
