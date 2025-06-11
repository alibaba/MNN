// sherpa-mnn/csrc/online-ctc-fst-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-ctc-fst-decoder.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/decodable-ctc.h"
#include "kaldifst/csrc/fstext-utils.h"
#include "sherpa-mnn/csrc/fst-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/online-stream.h"

namespace sherpa_mnn {

OnlineCtcFstDecoder::OnlineCtcFstDecoder(
    const OnlineCtcFstDecoderConfig &config, int32_t blank_id)
    : config_(config), fst_(ReadGraph(config.graph)), blank_id_(blank_id) {
  options_.max_active = config_.max_active;
}

std::unique_ptr<kaldi_decoder::FasterDecoder>
OnlineCtcFstDecoder::CreateFasterDecoder() const {
  return std::make_unique<kaldi_decoder::FasterDecoder>(*fst_, options_);
}

static void DecodeOne(const float *log_probs, int32_t num_rows,
                      int32_t num_cols, OnlineCtcDecoderResult *result,
                      OnlineStream *s, int32_t blank_id) {
  int32_t &processed_frames = s->GetFasterDecoderProcessedFrames();
  kaldi_decoder::DecodableCtc decodable(log_probs, num_rows, num_cols,
                                        processed_frames);

  kaldi_decoder::FasterDecoder *decoder = s->GetFasterDecoder();
  if (processed_frames == 0) {
    decoder->InitDecoding();
  }

  decoder->AdvanceDecoding(&decodable);

  if (decoder->ReachedFinal()) {
    fst::VectorFst<fst::LatticeArc> fst_out;
    bool ok = decoder->GetBestPath(&fst_out);
    if (ok) {
      std::vector<int32_t> isymbols_out;
      std::vector<int32_t> osymbols_out;
      /*ok =*/fst::GetLinearSymbolSequence(fst_out, &isymbols_out,
                                           &osymbols_out, nullptr);
      // TODO(fangjun): handle ok is false
      std::vector<int> tokens;
      tokens.reserve(isymbols_out.size());

      std::vector<int32_t> timestamps;
      timestamps.reserve(isymbols_out.size());

      std::ostringstream os;
      int32_t prev_id = -1;
      int32_t &num_trailing_blanks = result->num_trailing_blanks;
      int32_t f = 0;  // frame number

      for (auto i : isymbols_out) {
        i -= 1;

        if (i == blank_id) {
          num_trailing_blanks += 1;
        } else {
          num_trailing_blanks = 0;
        }

        if (i != blank_id && i != prev_id) {
          tokens.push_back(i);
          timestamps.push_back(f);
        }
        prev_id = i;
        f += 1;
      }

      result->tokens = std::move(tokens);
      result->words = std::move(osymbols_out);
      result->timestamps = std::move(timestamps);
      // no need to set frame_offset
    }
  }

  processed_frames += num_rows;
}

void OnlineCtcFstDecoder::Decode(const float *log_probs, int32_t batch_size,
                                 int32_t num_frames, int32_t vocab_size,
                                 std::vector<OnlineCtcDecoderResult> *results,
                                 OnlineStream **ss, int32_t n) {
  if (batch_size != results->size()) {
    SHERPA_ONNX_LOGE("Size mismatch! log_probs.size(0) %d, results.size(0): %d",
                     batch_size, static_cast<int32_t>(results->size()));
    exit(-1);
  }

  if (batch_size != n) {
    SHERPA_ONNX_LOGE("Size mismatch! log_probs.size(0) %d, n: %d", batch_size,
                     n);
    exit(-1);
  }

  const float *p = log_probs;

  for (int32_t i = 0; i != batch_size; ++i) {
    DecodeOne(p + i * num_frames * vocab_size, num_frames, vocab_size,
              &(*results)[i], ss[i], blank_id_);
  }
}

}  // namespace sherpa_mnn
