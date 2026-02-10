// sherpa-mnn/csrc/offline-transducer-modified-beam-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-transducer-modified-beam-search-decoder.h"

#include <deque>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/context-graph.h"
#include "sherpa-mnn/csrc/hypothesis.h"
#include "sherpa-mnn/csrc/log.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/packed-sequence.h"
#include "sherpa-mnn/csrc/slice.h"

namespace sherpa_mnn {

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerModifiedBeamSearchDecoder::Decode(
    MNN::Express::VARP encoder_out, MNN::Express::VARP encoder_out_length,
    OfflineStream **ss /*=nullptr */, int32_t n /*= 0*/) {
  PackedSequence packed_encoder_out = PackPaddedSequence(
      model_->Allocator(), encoder_out, encoder_out_length);

  int32_t batch_size =
      static_cast<int32_t>(packed_encoder_out.sorted_indexes.size());

  if (ss != nullptr) SHERPA_ONNX_CHECK_EQ(batch_size, n);

  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();

  std::vector<int> blanks(context_size, -1);
  blanks.back() = 0;

  std::deque<Hypotheses> finalized;
  std::vector<Hypotheses> cur;
  std::vector<Hypothesis> prev;

  std::vector<ContextGraphPtr> context_graphs(batch_size, nullptr);

  for (int32_t i = 0; i < batch_size; ++i) {
    const ContextState *context_state = nullptr;
    if (ss != nullptr) {
      context_graphs[i] =
          ss[packed_encoder_out.sorted_indexes[i]]->GetContextGraph();
      if (context_graphs[i] != nullptr)
        context_state = context_graphs[i]->Root();
    }
    Hypotheses blank_hyp({{blanks, 0, context_state}});
    cur.emplace_back(std::move(blank_hyp));
  }

  int32_t start = 0;
  int32_t t = 0;
  for (auto n : packed_encoder_out.batch_sizes) {
    MNN::Express::VARP cur_encoder_out = packed_encoder_out.Get(start, n);
    start += n;

    if (n < static_cast<int32_t>(cur.size())) {
      for (int32_t k = static_cast<int32_t>(cur.size()) - 1; k >= n; --k) {
        finalized.push_front(std::move(cur[k]));
      }

      cur.erase(cur.begin() + n, cur.end());
    }  // if (n < static_cast<int32_t>(cur.size()))

    // Due to merging paths with identical token sequences,
    // not all utterances have "max_active_paths" paths.
    auto hyps_row_splits = GetHypsRowSplits(cur);
    int32_t num_hyps = hyps_row_splits.back();

    prev.clear();
    prev.reserve(num_hyps);

    for (auto &hyps : cur) {
      for (auto &h : hyps) {
        prev.push_back(std::move(h.second));
      }
    }
    cur.clear();
    cur.reserve(n);

    auto decoder_input = model_->BuildDecoderInput(prev, num_hyps);
    // decoder_input shape: (num_hyps, context_size)

    auto decoder_out = model_->RunDecoder(std::move(decoder_input));
    // decoder_out is (num_hyps, joiner_dim)

    cur_encoder_out =
        Repeat(model_->Allocator(), cur_encoder_out, hyps_row_splits);
    // now cur_encoder_out is of shape (num_hyps, joiner_dim)

    MNN::Express::VARP logit =
        model_->RunJoiner(std::move(cur_encoder_out), View(decoder_out));

    float *p_logit = logit->writeMap<float>();
    if (blank_penalty_ > 0.0) {
      // assuming blank id is 0
      SubtractBlank(p_logit, vocab_size, num_hyps, 0, blank_penalty_);
    }
    LogSoftmax(p_logit, vocab_size, num_hyps);

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

    // Now compute top_k for each utterance
    for (int32_t i = 0; i != n; ++i) {
      int32_t start = hyps_row_splits[i];
      int32_t end = hyps_row_splits[i + 1];
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
          new_hyp.timestamps.push_back(t);
          if (context_graphs[i] != nullptr) {
            auto context_res =
                context_graphs[i]->ForwardOneStep(context_state, new_token);
            context_score = std::get<0>(context_res);
            new_hyp.context_state = std::get<1>(context_res);
          }
        }

        new_hyp.log_prob = p_logprob[k] + context_score;
        hyps.Add(std::move(new_hyp));
      }  // for (auto k : topk)
      p_logprob += (end - start) * vocab_size;
      cur.push_back(std::move(hyps));
    }  // for (int32_t i = 0; i != n; ++i)

    ++t;
  }  // for (auto n : packed_encoder_out.batch_sizes)

  for (auto &h : finalized) {
    cur.push_back(std::move(h));
  }

  // Finalize context biasing matching..
  for (int32_t i = 0; i < cur.size(); ++i) {
    for (auto iter = cur[i].begin(); iter != cur[i].end(); ++iter) {
      if (context_graphs[i] != nullptr) {
        auto context_res =
            context_graphs[i]->Finalize(iter->second.context_state);
        iter->second.log_prob += context_res.first;
        iter->second.context_state = context_res.second;
      }
    }
  }

  if (lm_) {
    // use LM for rescoring
    lm_->ComputeLMScore(lm_scale_, context_size, &cur);
  }

  std::vector<OfflineTransducerDecoderResult> unsorted_ans(batch_size);
  for (int32_t i = 0; i != batch_size; ++i) {
    Hypothesis hyp = cur[i].GetMostProbable(true);

    auto &r = unsorted_ans[packed_encoder_out.sorted_indexes[i]];

    // strip leading blanks
    r.tokens = {hyp.ys.begin() + context_size, hyp.ys.end()};
    r.timestamps = std::move(hyp.timestamps);
  }

  return unsorted_ans;
}

}  // namespace sherpa_mnn
