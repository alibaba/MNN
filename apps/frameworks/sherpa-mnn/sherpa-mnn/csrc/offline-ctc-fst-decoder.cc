// sherpa-mnn/csrc/offline-ctc-fst-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-ctc-fst-decoder.h"

#include <string>
#include <utility>

#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/decodable-ctc.h"
#include "kaldi-decoder/csrc/eigen.h"
#include "kaldi-decoder/csrc/faster-decoder.h"
#include "sherpa-mnn/csrc/fst-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

/**
 * @param decoder
 * @param p Pointer to a 2-d array of shape (num_frames, vocab_size)
 * @param num_frames Number of rows in the 2-d array.
 * @param vocab_size Number of columns in the 2-d array.
 * @return Return the decoded result.
 */
static OfflineCtcDecoderResult DecodeOne(kaldi_decoder::FasterDecoder *decoder,
                                         const float *p, int32_t num_frames,
                                         int32_t vocab_size) {
  OfflineCtcDecoderResult r;
  kaldi_decoder::DecodableCtc decodable(p, num_frames, vocab_size);

  decoder->Decode(&decodable);

  if (!decoder->ReachedFinal()) {
    SHERPA_ONNX_LOGE("Not reached final!");
    return r;
  }

  fst::VectorFst<fst::LatticeArc> decoded;  // linear FST.
  decoder->GetBestPath(&decoded);

  if (decoded.NumStates() == 0) {
    SHERPA_ONNX_LOGE("Empty best path!");
    return r;
  }

  auto cur_state = decoded.Start();

  int32_t blank_id = 0;

  for (int32_t t = 0, prev = -1; decoded.NumArcs(cur_state) == 1; ++t) {
    fst::ArcIterator<fst::Fst<fst::LatticeArc>> iter(decoded, cur_state);
    const auto &arc = iter.Value();

    cur_state = arc.nextstate;

    if (arc.ilabel == prev) {
      continue;
    }

    // 0 is epsilon here
    if (arc.ilabel == 0 || arc.ilabel == blank_id + 1) {
      prev = arc.ilabel;
      continue;
    }

    // -1 here since the input labels are incremented during graph
    // construction
    r.tokens.push_back(arc.ilabel - 1);
    if (arc.olabel != 0) {
      r.words.push_back(arc.olabel);
    }

    r.timestamps.push_back(t);
    prev = arc.ilabel;
  }

  return r;
}

OfflineCtcFstDecoder::OfflineCtcFstDecoder(
    const OfflineCtcFstDecoderConfig &config)
    : config_(config), fst_(ReadGraph(config_.graph)) {}

std::vector<OfflineCtcDecoderResult> OfflineCtcFstDecoder::Decode(
    MNN::Express::VARP log_probs, MNN::Express::VARP log_probs_length) {
  std::vector<int> shape = log_probs->getInfo()->dim;

  assert(static_cast<int32_t>(shape.size()) == 3);
  int32_t batch_size = shape[0];
  int32_t T = shape[1];
  int32_t vocab_size = shape[2];

  std::vector<int> length_shape =
      log_probs_length->getInfo()->dim;
  assert(static_cast<int32_t>(length_shape.size()) == 1);

  assert(shape[0] == length_shape[0]);

  kaldi_decoder::FasterDecoderOptions opts;
  opts.max_active = config_.max_active;
  kaldi_decoder::FasterDecoder faster_decoder(*fst_, opts);

  const float *start = log_probs->readMap<float>();

  std::vector<OfflineCtcDecoderResult> ans;
  ans.reserve(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    const float *p = start + i * T * vocab_size;
    int32_t num_frames = log_probs_length->readMap<int>()[i];
    auto r = DecodeOne(&faster_decoder, p, num_frames, vocab_size);
    ans.push_back(std::move(r));
  }

  return ans;
}

}  // namespace sherpa_mnn
