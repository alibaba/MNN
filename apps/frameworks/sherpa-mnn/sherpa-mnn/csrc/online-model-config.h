// sherpa-mnn/csrc/online-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/online-nemo-ctc-model-config.h"
#include "sherpa-mnn/csrc/online-paraformer-model-config.h"
#include "sherpa-mnn/csrc/online-transducer-model-config.h"
#include "sherpa-mnn/csrc/online-wenet-ctc-model-config.h"
#include "sherpa-mnn/csrc/online-zipformer2-ctc-model-config.h"
#include "sherpa-mnn/csrc/provider-config.h"

namespace sherpa_mnn {

struct OnlineModelConfig {
  OnlineTransducerModelConfig transducer;
  OnlineParaformerModelConfig paraformer;
  OnlineWenetCtcModelConfig wenet_ctc;
  OnlineZipformer2CtcModelConfig zipformer2_ctc;
  OnlineNeMoCtcModelConfig nemo_ctc;
  ProviderConfig provider_config;
  std::string tokens;
  int32_t num_threads = 1;
  int32_t warm_up = 0;
  bool debug = false;

  // Valid values:
  //  - conformer, conformer transducer from icefall
  //  - lstm, lstm transducer from icefall
  //  - zipformer, zipformer transducer from icefall
  //  - zipformer2, zipformer2 transducer or CTC from icefall
  //  - wenet_ctc, wenet CTC model
  //  - nemo_ctc, NeMo CTC model
  //
  // All other values are invalid and lead to loading the model twice.
  std::string model_type;

  // Valid values:
  //  - cjkchar
  //  - bpe
  //  - cjkchar+bpe
  std::string modeling_unit = "cjkchar";
  std::string bpe_vocab;

  /// if tokens_buf is non-empty,
  /// the tokens will be loaded from the buffer instead of from the
  /// "tokens" file
  std::string tokens_buf;

  OnlineModelConfig() = default;
  OnlineModelConfig(const OnlineTransducerModelConfig &transducer,
                    const OnlineParaformerModelConfig &paraformer,
                    const OnlineWenetCtcModelConfig &wenet_ctc,
                    const OnlineZipformer2CtcModelConfig &zipformer2_ctc,
                    const OnlineNeMoCtcModelConfig &nemo_ctc,
                    const ProviderConfig &provider_config,
                    const std::string &tokens, int32_t num_threads,
                    int32_t warm_up, bool debug, const std::string &model_type,
                    const std::string &modeling_unit,
                    const std::string &bpe_vocab)
      : transducer(transducer),
        paraformer(paraformer),
        wenet_ctc(wenet_ctc),
        zipformer2_ctc(zipformer2_ctc),
        nemo_ctc(nemo_ctc),
        provider_config(provider_config),
        tokens(tokens),
        num_threads(num_threads),
        warm_up(warm_up),
        debug(debug),
        model_type(model_type),
        modeling_unit(modeling_unit),
        bpe_vocab(bpe_vocab) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_
