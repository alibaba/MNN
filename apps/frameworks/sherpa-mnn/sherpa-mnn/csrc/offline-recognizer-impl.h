// sherpa-mnn/csrc/offline-recognizer-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/offline-recognizer.h"
#include "sherpa-mnn/csrc/offline-stream.h"

namespace sherpa_mnn {

class OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerImpl(const OfflineRecognizerConfig &config);

  static std::unique_ptr<OfflineRecognizerImpl> Create(
      const OfflineRecognizerConfig &config);

  template <typename Manager>
  OfflineRecognizerImpl(Manager *mgr, const OfflineRecognizerConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineRecognizerImpl> Create(
      Manager *mgr, const OfflineRecognizerConfig &config);

  virtual ~OfflineRecognizerImpl() = default;

  virtual std::unique_ptr<OfflineStream> CreateStream(
      const std::string &hotwords) const {
    SHERPA_ONNX_LOGE("Only transducer models support contextual biasing.");
    exit(-1);
  }

  virtual std::unique_ptr<OfflineStream> CreateStream() const = 0;

  virtual void DecodeStreams(OfflineStream **ss, int32_t n) const = 0;

  virtual void SetConfig(const OfflineRecognizerConfig &config);

  virtual OfflineRecognizerConfig GetConfig() const = 0;

  std::string ApplyInverseTextNormalization(std::string text) const;

 private:
  OfflineRecognizerConfig config_;
  // for inverse text normalization. Used only if
  // config.rule_fsts is not empty or
  // config.rule_fars is not empty
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> itn_list_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_IMPL_H_
