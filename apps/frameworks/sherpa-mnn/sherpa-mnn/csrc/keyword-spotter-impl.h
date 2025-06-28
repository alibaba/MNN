// sherpa-mnn/csrc/keyword-spotter-impl.h
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_IMPL_H_
#define SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/keyword-spotter.h"
#include "sherpa-mnn/csrc/online-stream.h"

namespace sherpa_mnn {

class KeywordSpotterImpl {
 public:
  static std::unique_ptr<KeywordSpotterImpl> Create(
      const KeywordSpotterConfig &config);

  template <typename Manager>
  static std::unique_ptr<KeywordSpotterImpl> Create(
      Manager *mgr, const KeywordSpotterConfig &config);

  virtual ~KeywordSpotterImpl() = default;

  virtual std::unique_ptr<OnlineStream> CreateStream() const = 0;

  virtual std::unique_ptr<OnlineStream> CreateStream(
      const std::string &keywords) const = 0;

  virtual bool IsReady(OnlineStream *s) const = 0;

  virtual void Reset(OnlineStream *s) const = 0;

  virtual void DecodeStreams(OnlineStream **ss, int32_t n) const = 0;

  virtual KeywordResult GetResult(OnlineStream *s) const = 0;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_IMPL_H_
