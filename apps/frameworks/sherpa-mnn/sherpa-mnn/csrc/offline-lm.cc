// sherpa-mnn/csrc/offline-lm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-lm.h"

#include <algorithm>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mnn/csrc/offline-rnn-lm.h"

namespace sherpa_mnn {

std::unique_ptr<OfflineLM> OfflineLM::Create(const OfflineLMConfig &config) {
  return std::make_unique<OfflineRnnLM>(config);
}

template <typename Manager>
std::unique_ptr<OfflineLM> OfflineLM::Create(Manager *mgr,
                                             const OfflineLMConfig &config) {
  return std::make_unique<OfflineRnnLM>(mgr, config);
}

void OfflineLM::ComputeLMScore(float scale, int32_t context_size,
                               std::vector<Hypotheses> *hyps) {
  // compute the max token seq so that we know how much space to allocate
  int32_t max_token_seq = 0;
  int32_t num_hyps = 0;

  // we subtract context_size below since each token sequence is prepended
  // with context_size blanks
  for (const auto &h : *hyps) {
    num_hyps += h.Size();
    for (const auto &t : h) {
      max_token_seq =
          std::max<int32_t>(max_token_seq, t.second.ys.size() - context_size);
    }
  }

  MNNAllocator* allocator;
  std::array<int, 2> x_shape{num_hyps, max_token_seq};
  MNN::Express::VARP x = MNNUtilsCreateTensor<int>(allocator, x_shape.data(),
                                                   x_shape.size());

  std::array<int, 1> x_lens_shape{num_hyps};
  MNN::Express::VARP x_lens = MNNUtilsCreateTensor<int>(
      allocator, x_lens_shape.data(), x_lens_shape.size());

  int *p = x->writeMap<int>();
  std::fill(p, p + num_hyps * max_token_seq, 0);

  int *p_lens = x_lens->writeMap<int>();

  for (const auto &h : *hyps) {
    for (const auto &t : h) {
      const auto &ys = t.second.ys;
      int32_t len = ys.size() - context_size;
      std::copy(ys.begin() + context_size, ys.end(), p);
      *p_lens = len;

      p += max_token_seq;
      ++p_lens;
    }
  }
  auto negative_loglike = Rescore(std::move(x), std::move(x_lens));
  const float *p_nll = negative_loglike->readMap<float>();
  for (auto &h : *hyps) {
    for (auto &t : h) {
      // Use -scale here since we want to change negative loglike to loglike.
      t.second.lm_log_prob = -scale * (*p_nll);
      ++p_nll;
    }
  }
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineLM> OfflineLM::Create(
    AAssetManager *mgr, const OfflineLMConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflineLM> OfflineLM::Create(
    NativeResourceManager *mgr, const OfflineLMConfig &config);
#endif

}  // namespace sherpa_mnn
