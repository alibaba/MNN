// sherpa-mnn/csrc/offline-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-mnn/csrc/offline-fire-red-asr-model-config.h"
#include "sherpa-mnn/csrc/offline-moonshine-model-config.h"
#include "sherpa-mnn/csrc/offline-nemo-enc-dec-ctc-model-config.h"
#include "sherpa-mnn/csrc/offline-paraformer-model-config.h"
#include "sherpa-mnn/csrc/offline-sense-voice-model-config.h"
#include "sherpa-mnn/csrc/offline-tdnn-model-config.h"
#include "sherpa-mnn/csrc/offline-transducer-model-config.h"
#include "sherpa-mnn/csrc/offline-wenet-ctc-model-config.h"
#include "sherpa-mnn/csrc/offline-whisper-model-config.h"
#include "sherpa-mnn/csrc/offline-zipformer-ctc-model-config.h"

namespace sherpa_mnn {

struct OfflineModelConfig {
  OfflineTransducerModelConfig transducer;
  OfflineParaformerModelConfig paraformer;
  OfflineNemoEncDecCtcModelConfig nemo_ctc;
  OfflineWhisperModelConfig whisper;
  OfflineFireRedAsrModelConfig fire_red_asr;
  OfflineTdnnModelConfig tdnn;
  OfflineZipformerCtcModelConfig zipformer_ctc;
  OfflineWenetCtcModelConfig wenet_ctc;
  OfflineSenseVoiceModelConfig sense_voice;
  OfflineMoonshineModelConfig moonshine;
  std::string telespeech_ctc;

  std::string tokens;
  int32_t num_threads = 2;
  bool debug = false;
  std::string provider = "cpu";

  // With the help of this field, we only need to load the model once
  // instead of twice; and therefore it reduces initialization time.
  //
  // Valid values:
  //  - transducer. The given model is from icefall
  //  - paraformer. It is a paraformer model
  //  - nemo_ctc. It is a NeMo CTC model.
  //
  // All other values are invalid and lead to loading the model twice.
  std::string model_type;

  std::string modeling_unit = "cjkchar";
  std::string bpe_vocab;

  OfflineModelConfig() = default;
  OfflineModelConfig(const OfflineTransducerModelConfig &transducer,
                     const OfflineParaformerModelConfig &paraformer,
                     const OfflineNemoEncDecCtcModelConfig &nemo_ctc,
                     const OfflineWhisperModelConfig &whisper,
                     const OfflineFireRedAsrModelConfig &fire_red_asr,
                     const OfflineTdnnModelConfig &tdnn,
                     const OfflineZipformerCtcModelConfig &zipformer_ctc,
                     const OfflineWenetCtcModelConfig &wenet_ctc,
                     const OfflineSenseVoiceModelConfig &sense_voice,
                     const OfflineMoonshineModelConfig &moonshine,
                     const std::string &telespeech_ctc,
                     const std::string &tokens, int32_t num_threads, bool debug,
                     const std::string &provider, const std::string &model_type,
                     const std::string &modeling_unit,
                     const std::string &bpe_vocab)
      : transducer(transducer),
        paraformer(paraformer),
        nemo_ctc(nemo_ctc),
        whisper(whisper),
        fire_red_asr(fire_red_asr),
        tdnn(tdnn),
        zipformer_ctc(zipformer_ctc),
        wenet_ctc(wenet_ctc),
        sense_voice(sense_voice),
        moonshine(moonshine),
        telespeech_ctc(telespeech_ctc),
        tokens(tokens),
        num_threads(num_threads),
        debug(debug),
        provider(provider),
        model_type(model_type),
        modeling_unit(modeling_unit),
        bpe_vocab(bpe_vocab) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MODEL_CONFIG_H_
