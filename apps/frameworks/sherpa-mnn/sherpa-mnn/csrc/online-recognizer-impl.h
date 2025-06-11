// sherpa-mnn/csrc/online-recognizer-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/online-recognizer.h"
#include "sherpa-mnn/csrc/online-stream.h"

namespace sherpa_mnn {

class OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerImpl(const OnlineRecognizerConfig &config);

  static std::unique_ptr<OnlineRecognizerImpl> Create(
      const OnlineRecognizerConfig &config);

  template <typename Manager>
  OnlineRecognizerImpl(Manager *mgr, const OnlineRecognizerConfig &config);

  template <typename Manager>
  static std::unique_ptr<OnlineRecognizerImpl> Create(
      Manager *mgr, const OnlineRecognizerConfig &config);

  virtual ~OnlineRecognizerImpl() = default;

  virtual std::unique_ptr<OnlineStream> CreateStream() const = 0;

  virtual std::unique_ptr<OnlineStream> CreateStream(
      const std::string &hotwords) const {
    SHERPA_ONNX_LOGE("Only transducer models support contextual biasing.");
    exit(-1);
  }

  virtual bool IsReady(OnlineStream *s) const = 0;

  virtual void WarmpUpRecognizer(int32_t warmup, int32_t mbs) const {
    // ToDo extending to other  models
    SHERPA_ONNX_LOGE("Only zipformer2 model supports Warm up for now.");
    exit(-1);
  }

  virtual void DecodeStreams(OnlineStream **ss, int32_t n) const = 0;

  virtual OnlineRecognizerResult GetResult(OnlineStream *s) const = 0;

  virtual bool IsEndpoint(OnlineStream *s) const = 0;

  virtual void Reset(OnlineStream *s) const = 0;

  std::string ApplyInverseTextNormalization(std::string text) const;

 private:
  OnlineRecognizerConfig config_;
  // for inverse text normalization. Used only if
  // config.rule_fsts is not empty or
  // config.rule_fars is not empty
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> itn_list_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_IMPL_H_
