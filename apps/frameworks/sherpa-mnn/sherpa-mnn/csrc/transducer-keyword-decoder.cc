// sherpa-mnn/csrc/transducer-keywords-decoder.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/transducer-keyword-decoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/log.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace sherpa_mnn {

TransducerKeywordResult TransducerKeywordDecoder::GetEmptyResult() const {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
  TransducerKeywordResult r;
  std::vector<int> blanks(context_size, -1);
  blanks.back() = blank_id;

  Hypotheses blank_hyp({{blanks, 0}});
  r.hyps = std::move(blank_hyp);
  return r;
}

void TransducerKeywordDecoder::Decode(
    MNN::Express::VARP encoder_out, OnlineStream **ss,
    std::vector<TransducerKeywordResult> *result) {
  std::vector<int> encoder_out_shape =
      encoder_out->getInfo()->dim;

  if (encoder_out_shape[0] != result->size()) {
    SHERPA_ONNX_LOGE(
        "Size mismatch! encoder_out.size(0) %d, result.size(0): %d\n",
        static_cast<int32_t>(encoder_out_shape[0]),
        static_cast<int32_t>(result->size()));
    exit(-1);
  }

  int32_t batch_size = static_cast<int32_t>(encoder_out_shape[0]);

  int32_t num_frames = static_cast<int32_t>(encoder_out_shape[1]);
  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();
  std::vector<int> blanks(context_size, -1);
  blanks.back() = 0;  // blank_id is hardcoded to 0

  std::vector<Hypotheses> cur;
  for (auto &r : *result) {
    cur.push_back(std::move(r.hyps));
  }
  std::vector<Hypothesis> prev;

  for (int32_t t = 0; t != num_frames; ++t) {
    // Due to merging paths with identical token sequences,
    // not all utterances have "num_active_paths" paths.
    auto hyps_row_splits = GetHypsRowSplits(cur);
    int32_t num_hyps =
        hyps_row_splits.back();  // total num hyps for all utterance
    prev.clear();
    for (auto &hyps : cur) {
      for (auto &h : hyps) {
        prev.push_back(std::move(h.second));
      }
    }
    cur.clear();
    cur.reserve(batch_size);

    MNN::Express::VARP decoder_input = model_->BuildDecoderInput(prev);
    MNN::Express::VARP decoder_out = model_->RunDecoder(std::move(decoder_input));

    MNN::Express::VARP cur_encoder_out =
        GetEncoderOutFrame(model_->Allocator(), encoder_out, t);
    cur_encoder_out =
        Repeat(model_->Allocator(), cur_encoder_out, hyps_row_splits);
    MNN::Express::VARP logit =
        model_->RunJoiner(std::move(cur_encoder_out), View(decoder_out));

    float *p_logit = logit->writeMap<float>();
    LogSoftmax(p_logit, vocab_size, num_hyps);

    // The acoustic logprobs for current frame
    std::vector<float> logprobs(vocab_size * num_hyps);
    std::memcpy(logprobs.data(), p_logit,
                sizeof(float) * vocab_size * num_hyps);

    // now p_logit contains log_softmax output, we rename it to p_logprob
    // to match what it actually contains
    float *p_logprob = p_logit;

    // add log_prob of each hypothesis to p_logprob before taking top_k
    for (int32_t i = 0; i != num_hyps; ++i) {
      float log_prob = prev[i].log_prob;
      for (int32_t k = 0; k != vocab_size; ++k, ++p_logprob) {
        *p_logprob += log_prob;
      }
    }
    p_logprob = p_logit;  // we changed p_logprob in the above for loop

    for (int32_t b = 0; b != batch_size; ++b) {
      int32_t frame_offset = (*result)[b].frame_offset;
      int32_t start = hyps_row_splits[b];
      int32_t end = hyps_row_splits[b + 1];
      auto topk =
          TopkIndex(p_logprob, vocab_size * (end - start), max_active_paths_);

      Hypotheses hyps;
      for (auto k : topk) {
        int32_t hyp_index = k / vocab_size + start;
        int32_t new_token = k % vocab_size;

        Hypothesis new_hyp = prev[hyp_index];
        float context_score = 0;
        auto context_state = new_hyp.context_state;

        // blank is hardcoded to 0
        // also, it treats unk as blank
        if (new_token != 0 && new_token != unk_id_) {
          new_hyp.ys.push_back(new_token);
          new_hyp.timestamps.push_back(t + frame_offset);
          new_hyp.ys_probs.push_back(
              exp(logprobs[hyp_index * vocab_size + new_token]));

          new_hyp.num_trailing_blanks = 0;
          auto context_res = ss[b]->GetContextGraph()->ForwardOneStep(
              context_state, new_token);
          context_score = std::get<0>(context_res);
          new_hyp.context_state = std::get<1>(context_res);
          // Start matching from the start state, forget the decoder history.
          if (new_hyp.context_state->token == -1) {
            new_hyp.ys = blanks;
            new_hyp.timestamps.clear();
            new_hyp.ys_probs.clear();
          }
        } else {
          ++new_hyp.num_trailing_blanks;
        }
        new_hyp.log_prob = p_logprob[k] + context_score;
        hyps.Add(std::move(new_hyp));
      }  // for (auto k : topk)

      auto best_hyp = hyps.GetMostProbable(false);

      auto status = ss[b]->GetContextGraph()->IsMatched(best_hyp.context_state);
      bool matched = std::get<0>(status);
      const ContextState *matched_state = std::get<1>(status);

      if (matched) {
        float ys_prob = 0.0;
        for (int32_t i = 0; i < matched_state->level; ++i) {
          ys_prob += best_hyp.ys_probs[i];
        }
        ys_prob /= matched_state->level;
        if (best_hyp.num_trailing_blanks > num_trailing_blanks_ &&
            ys_prob >= matched_state->ac_threshold) {
          auto &r = (*result)[b];
          r.tokens = {best_hyp.ys.end() - matched_state->level,
                      best_hyp.ys.end()};
          r.timestamps = {best_hyp.timestamps.end() - matched_state->level,
                          best_hyp.timestamps.end()};
          r.keyword = matched_state->phrase;

          hyps = Hypotheses({{blanks, 0, ss[b]->GetContextGraph()->Root()}});
        }
      }
      cur.push_back(std::move(hyps));
      p_logprob += (end - start) * vocab_size;
    }  // for (int32_t b = 0; b != batch_size; ++b)
  }

  for (int32_t b = 0; b != batch_size; ++b) {
    auto &hyps = cur[b];
    auto best_hyp = hyps.GetMostProbable(false);
    auto &r = (*result)[b];
    r.hyps = std::move(hyps);
    r.num_trailing_blanks = best_hyp.num_trailing_blanks;
    r.frame_offset += num_frames;
  }
}

}  // namespace sherpa_mnn
