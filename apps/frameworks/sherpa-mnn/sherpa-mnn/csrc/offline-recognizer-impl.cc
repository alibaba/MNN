// sherpa-mnn/csrc/offline-recognizer-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-recognizer-impl.h"

#include <string>
#include <strstream>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/offline-recognizer-ctc-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer-fire-red-asr-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer-moonshine-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer-paraformer-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer-sense-voice-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer-transducer-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer-transducer-nemo-impl.h"
#include "sherpa-mnn/csrc/offline-recognizer-whisper-impl.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

std::unique_ptr<OfflineRecognizerImpl> OfflineRecognizerImpl::Create(
    const OfflineRecognizerConfig &config) {
  if (!config.model_config.sense_voice.model.empty()) {
    return std::make_unique<OfflineRecognizerSenseVoiceImpl>(config);
  }

  if (!config.model_config.paraformer.model.empty()) {
    return std::make_unique<OfflineRecognizerParaformerImpl>(config);
  }

  if (!config.model_config.nemo_ctc.model.empty() ||
      !config.model_config.zipformer_ctc.model.empty() ||
      !config.model_config.tdnn.model.empty() ||
      !config.model_config.wenet_ctc.model.empty()) {
    return std::make_unique<OfflineRecognizerCtcImpl>(config);
  }

  if (!config.model_config.whisper.encoder.empty()) {
    return std::make_unique<OfflineRecognizerWhisperImpl>(config);
  }

  if (!config.model_config.fire_red_asr.encoder.empty()) {
    return std::make_unique<OfflineRecognizerFireRedAsrImpl>(config);
  }

  if (!config.model_config.moonshine.preprocessor.empty()) {
    return std::make_unique<OfflineRecognizerMoonshineImpl>(config);
  }

  // TODO(fangjun): Refactor it. We only need to use model type for the
  // following models:
  //  1. transducer and nemo_transducer
  if (!config.model_config.model_type.empty()) {
    const auto &model_type = config.model_config.model_type;
    if (model_type == "transducer") {
      return std::make_unique<OfflineRecognizerTransducerImpl>(config);
    } else if (model_type == "nemo_transducer") {
      return std::make_unique<OfflineRecognizerTransducerNeMoImpl>(config);
    } else if (model_type == "paraformer") {
      return std::make_unique<OfflineRecognizerParaformerImpl>(config);
    } else if (model_type == "nemo_ctc" || model_type == "tdnn" ||
               model_type == "zipformer2_ctc" || model_type == "wenet_ctc" ||
               model_type == "telespeech_ctc") {
      return std::make_unique<OfflineRecognizerCtcImpl>(config);
    } else if (model_type == "whisper") {
      // unreachable
      return std::make_unique<OfflineRecognizerWhisperImpl>(config);
    } else if (model_type == "moonshine") {
      // unreachable
      return std::make_unique<OfflineRecognizerMoonshineImpl>(config);
    } else {
      SHERPA_ONNX_LOGE(
          "Invalid model_type: %s. Trying to load the model to get its type",
          model_type.c_str());
    }
  }

  MNNEnv env;

  std::shared_ptr<MNN::Express::Executor::RuntimeManager> sess_opts;
  
  

  std::string model_filename;
  if (!config.model_config.transducer.encoder_filename.empty()) {
    model_filename = config.model_config.transducer.encoder_filename;
  } else if (!config.model_config.paraformer.model.empty()) {
    model_filename = config.model_config.paraformer.model;
  } else if (!config.model_config.nemo_ctc.model.empty()) {
    model_filename = config.model_config.nemo_ctc.model;
  } else if (!config.model_config.telespeech_ctc.empty()) {
    model_filename = config.model_config.telespeech_ctc;
  } else if (!config.model_config.tdnn.model.empty()) {
    model_filename = config.model_config.tdnn.model;
  } else if (!config.model_config.zipformer_ctc.model.empty()) {
    model_filename = config.model_config.zipformer_ctc.model;
  } else if (!config.model_config.wenet_ctc.model.empty()) {
    model_filename = config.model_config.wenet_ctc.model;
  } else if (!config.model_config.whisper.encoder.empty()) {
    model_filename = config.model_config.whisper.encoder;
  } else {
    SHERPA_ONNX_LOGE("Please provide a model");
    exit(-1);
  }

  auto buf = ReadFile(model_filename);

  auto encoder_sess =
      std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {},  (const uint8_t*)buf.data(), buf.size(), sess_opts));

  MNNMeta meta_data = encoder_sess->getInfo()->metaData;

  MNNAllocator* allocator;  // used in the macro below

  auto model_type =
      LookupCustomModelMetaData(meta_data, "model_type", allocator);
  if (model_type.empty()) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n\n"
        "Please refer to the following URLs to add metadata"
        "\n"
        "(0) Transducer models from icefall"
        "\n    "
        "https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/"
        "pruned_transducer_stateless7/export-onnx.py#L303"
        "\n"
        "(1) Nemo CTC models\n    "
        "https://huggingface.co/csukuangfj/"
        "sherpa-mnn-nemo-ctc-en-citrinet-512/blob/main/add-model-metadata.py"
        "\n"
        "(2) Paraformer"
        "\n    "
        "https://huggingface.co/csukuangfj/"
        "paraformer-onnxruntime-python-example/blob/main/add-model-metadata.py"
        "\n    "
        "(3) Whisper"
        "\n    "
        "(4) Tdnn models of the yesno recipe from icefall"
        "\n    "
        "https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR/tdnn"
        "\n"
        "(5) Zipformer CTC models from icefall"
        "\n    "
        "https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/"
        "zipformer/export-onnx-ctc.py"
        "\n"
        "(6) CTC models from WeNet"
        "\n    "
        "https://github.com/k2-fsa/sherpa-mnn/blob/master/scripts/wenet/run.sh"
        "\n"
        "(7) CTC models from TeleSpeech"
        "\n    "
        "https://github.com/Tele-AI/TeleSpeech-ASR"
        "\n"
        "\n");
    exit(-1);
  }

  if (model_type == "conformer" || model_type == "zipformer" ||
      model_type == "zipformer2") {
    return std::make_unique<OfflineRecognizerTransducerImpl>(config);
  }

  if (model_type == "paraformer") {
    return std::make_unique<OfflineRecognizerParaformerImpl>(config);
  }

  if ((model_type == "EncDecHybridRNNTCTCBPEModel" ||
       model_type == "EncDecRNNTBPEModel") &&
      !config.model_config.transducer.decoder_filename.empty() &&
      !config.model_config.transducer.joiner_filename.empty()) {
    return std::make_unique<OfflineRecognizerTransducerNeMoImpl>(config);
  }

  if (model_type == "EncDecCTCModelBPE" || model_type == "EncDecCTCModel" ||
      model_type == "EncDecHybridRNNTCTCBPEModel" || model_type == "tdnn" ||
      model_type == "zipformer2_ctc" || model_type == "wenet_ctc" ||
      model_type == "telespeech_ctc") {
    return std::make_unique<OfflineRecognizerCtcImpl>(config);
  }

  if (strncmp(model_type.c_str(), "whisper", 7) == 0) {
    return std::make_unique<OfflineRecognizerWhisperImpl>(config);
  }

  SHERPA_ONNX_LOGE(
      "\nUnsupported model_type: %s\n"
      "We support only the following model types at present: \n"
      " - Non-streaming transducer models from icefall\n"
      " - Non-streaming Paraformer models from FunASR\n"
      " - EncDecCTCModelBPE models from NeMo\n"
      " - EncDecCTCModel models from NeMo\n"
      " - EncDecHybridRNNTCTCBPEModel models from NeMo\n"
      " - EncDecRNNTBPEModel models from NeMO"
      " - Whisper models\n"
      " - Tdnn models\n"
      " - Zipformer CTC models\n"
      " - WeNet CTC models\n"
      " - TeleSpeech CTC models\n",
      model_type.c_str());

  exit(-1);
}

template <typename Manager>
std::unique_ptr<OfflineRecognizerImpl> OfflineRecognizerImpl::Create(
    Manager *mgr, const OfflineRecognizerConfig &config) {
  if (!config.model_config.sense_voice.model.empty()) {
    return std::make_unique<OfflineRecognizerSenseVoiceImpl>(mgr, config);
  }

  if (!config.model_config.paraformer.model.empty()) {
    return std::make_unique<OfflineRecognizerParaformerImpl>(mgr, config);
  }

  if (!config.model_config.nemo_ctc.model.empty() ||
      !config.model_config.zipformer_ctc.model.empty() ||
      !config.model_config.tdnn.model.empty() ||
      !config.model_config.wenet_ctc.model.empty()) {
    return std::make_unique<OfflineRecognizerCtcImpl>(mgr, config);
  }

  if (!config.model_config.whisper.encoder.empty()) {
    return std::make_unique<OfflineRecognizerWhisperImpl>(mgr, config);
  }

  if (!config.model_config.fire_red_asr.encoder.empty()) {
    return std::make_unique<OfflineRecognizerFireRedAsrImpl>(mgr, config);
  }

  if (!config.model_config.moonshine.preprocessor.empty()) {
    return std::make_unique<OfflineRecognizerMoonshineImpl>(mgr, config);
  }

  // TODO(fangjun): Refactor it. We only need to use model type for the
  // following models:
  //  1. transducer and nemo_transducer
  if (!config.model_config.model_type.empty()) {
    const auto &model_type = config.model_config.model_type;
    if (model_type == "transducer") {
      return std::make_unique<OfflineRecognizerTransducerImpl>(mgr, config);
    } else if (model_type == "nemo_transducer") {
      return std::make_unique<OfflineRecognizerTransducerNeMoImpl>(mgr, config);
    } else if (model_type == "paraformer") {
      return std::make_unique<OfflineRecognizerParaformerImpl>(mgr, config);
    } else if (model_type == "nemo_ctc" || model_type == "tdnn" ||
               model_type == "zipformer2_ctc" || model_type == "wenet_ctc" ||
               model_type == "telespeech_ctc") {
      return std::make_unique<OfflineRecognizerCtcImpl>(mgr, config);
    } else if (model_type == "whisper") {
      return std::make_unique<OfflineRecognizerWhisperImpl>(mgr, config);
    } else if (model_type == "moonshine") {
      return std::make_unique<OfflineRecognizerMoonshineImpl>(mgr, config);
    } else {
      SHERPA_ONNX_LOGE(
          "Invalid model_type: %s. Trying to load the model to get its type",
          model_type.c_str());
    }
  }

  MNNEnv env;

  std::shared_ptr<MNN::Express::Executor::RuntimeManager> sess_opts;
  
  

  std::string model_filename;
  if (!config.model_config.transducer.encoder_filename.empty()) {
    model_filename = config.model_config.transducer.encoder_filename;
  } else if (!config.model_config.paraformer.model.empty()) {
    model_filename = config.model_config.paraformer.model;
  } else if (!config.model_config.nemo_ctc.model.empty()) {
    model_filename = config.model_config.nemo_ctc.model;
  } else if (!config.model_config.tdnn.model.empty()) {
    model_filename = config.model_config.tdnn.model;
  } else if (!config.model_config.zipformer_ctc.model.empty()) {
    model_filename = config.model_config.zipformer_ctc.model;
  } else if (!config.model_config.wenet_ctc.model.empty()) {
    model_filename = config.model_config.wenet_ctc.model;
  } else if (!config.model_config.telespeech_ctc.empty()) {
    model_filename = config.model_config.telespeech_ctc;
  } else if (!config.model_config.whisper.encoder.empty()) {
    model_filename = config.model_config.whisper.encoder;
  } else {
    SHERPA_ONNX_LOGE("Please provide a model");
    exit(-1);
  }

  auto buf = ReadFile(mgr, model_filename);

  auto encoder_sess =
      std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {},  (const uint8_t*)buf.data(), buf.size(), sess_opts));

  MNNMeta meta_data = encoder_sess->getInfo()->metaData;

  MNNAllocator* allocator;  // used in the macro below

  auto model_type =
      LookupCustomModelMetaData(meta_data, "model_type", allocator);
  if (model_type.empty()) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n\n"
        "Please refer to the following URLs to add metadata"
        "\n"
        "(0) Transducer models from icefall"
        "\n    "
        "https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/"
        "pruned_transducer_stateless7/export-onnx.py#L303"
        "\n"
        "(1) Nemo CTC models\n    "
        "https://huggingface.co/csukuangfj/"
        "sherpa-mnn-nemo-ctc-en-citrinet-512/blob/main/add-model-metadata.py"
        "\n"
        "(2) Paraformer"
        "\n    "
        "https://huggingface.co/csukuangfj/"
        "paraformer-onnxruntime-python-example/blob/main/add-model-metadata.py"
        "\n    "
        "(3) Whisper"
        "\n    "
        "(4) Tdnn models of the yesno recipe from icefall"
        "\n    "
        "https://github.com/k2-fsa/icefall/tree/master/egs/yesno/ASR/tdnn"
        "\n"
        "(5) Zipformer CTC models from icefall"
        "\n    "
        "https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/"
        "zipformer/export-onnx-ctc.py"
        "\n"
        "(6) CTC models from WeNet"
        "\n    "
        "https://github.com/k2-fsa/sherpa-mnn/blob/master/scripts/wenet/run.sh"
        "\n"
        "(7) CTC models from TeleSpeech"
        "\n    "
        "https://github.com/Tele-AI/TeleSpeech-ASR"
        "\n"
        "\n");
    exit(-1);
  }

  if (model_type == "conformer" || model_type == "zipformer" ||
      model_type == "zipformer2") {
    return std::make_unique<OfflineRecognizerTransducerImpl>(mgr, config);
  }

  if (model_type == "paraformer") {
    return std::make_unique<OfflineRecognizerParaformerImpl>(mgr, config);
  }

  if ((model_type == "EncDecHybridRNNTCTCBPEModel" ||
       model_type == "EncDecRNNTBPEModel") &&
      !config.model_config.transducer.decoder_filename.empty() &&
      !config.model_config.transducer.joiner_filename.empty()) {
    return std::make_unique<OfflineRecognizerTransducerNeMoImpl>(mgr, config);
  }

  if (model_type == "EncDecCTCModelBPE" || model_type == "EncDecCTCModel" ||
      model_type == "EncDecHybridRNNTCTCBPEModel" || model_type == "tdnn" ||
      model_type == "zipformer2_ctc" || model_type == "wenet_ctc" ||
      model_type == "telespeech_ctc") {
    return std::make_unique<OfflineRecognizerCtcImpl>(mgr, config);
  }

  if (strncmp(model_type.c_str(), "whisper", 7) == 0) {
    return std::make_unique<OfflineRecognizerWhisperImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE(
      "\nUnsupported model_type: %s\n"
      "We support only the following model types at present: \n"
      " - Non-streaming transducer models from icefall\n"
      " - Non-streaming Paraformer models from FunASR\n"
      " - EncDecCTCModelBPE models from NeMo\n"
      " - EncDecCTCModel models from NeMo\n"
      " - EncDecHybridRNNTCTCBPEModel models from NeMo\n"
      " - EncDecRNNTBPEModel models from NeMo\n"
      " - Whisper models\n"
      " - Tdnn models\n"
      " - Zipformer CTC models\n"
      " - WeNet CTC models\n"
      " - TeleSpeech CTC models\n",
      model_type.c_str());

  exit(-1);
}

OfflineRecognizerImpl::OfflineRecognizerImpl(
    const OfflineRecognizerConfig &config)
    : config_(config) {
  if (!config.rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(config.rule_fsts, ",", false, &files);
    itn_list_.reserve(files.size());
    for (const auto &f : files) {
      if (config.model_config.debug) {
        SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
      }
      itn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
    }
  }

  if (!config.rule_fars.empty()) {
    if (config.model_config.debug) {
      SHERPA_ONNX_LOGE("Loading FST archives");
    }
    std::vector<std::string> files;
    SplitStringToVector(config.rule_fars, ",", false, &files);

    itn_list_.reserve(files.size() + itn_list_.size());

    for (const auto &f : files) {
      if (config.model_config.debug) {
        SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
      }
      std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
          fst::FarReader<fst::StdArc>::Open(f));
      for (; !reader->Done(); reader->Next()) {
        std::unique_ptr<fst::StdConstFst> r(
            fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

        itn_list_.push_back(
            std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
      }
    }

    if (config.model_config.debug) {
      SHERPA_ONNX_LOGE("FST archives loaded!");
    }
  }
}

template <typename Manager>
OfflineRecognizerImpl::OfflineRecognizerImpl(
    Manager *mgr, const OfflineRecognizerConfig &config)
    : config_(config) {
  if (!config.rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(config.rule_fsts, ",", false, &files);
    itn_list_.reserve(files.size());
    for (const auto &f : files) {
      if (config.model_config.debug) {
        SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
      }
      auto buf = ReadFile(mgr, f);
      std::istrstream is(buf.data(), buf.size());
      itn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(is));
    }
  }

  if (!config.rule_fars.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(config.rule_fars, ",", false, &files);
    itn_list_.reserve(files.size() + itn_list_.size());

    for (const auto &f : files) {
      if (config.model_config.debug) {
        SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
      }

      auto buf = ReadFile(mgr, f);

      std::unique_ptr<std::istream> s(
          new std::istrstream(buf.data(), buf.size()));

      std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
          fst::FarReader<fst::StdArc>::Open(std::move(s)));

      for (; !reader->Done(); reader->Next()) {
        std::unique_ptr<fst::StdConstFst> r(
            fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

        itn_list_.push_back(
            std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
      }  // for (; !reader->Done(); reader->Next())
    }    // for (const auto &f : files)
  }      // if (!config.rule_fars.empty())
}

std::string OfflineRecognizerImpl::ApplyInverseTextNormalization(
    std::string text) const {
  text = RemoveInvalidUtf8Sequences(text);

  if (!itn_list_.empty()) {
    for (const auto &tn : itn_list_) {
      text = tn->Normalize(text);
    }
  }

  return text;
}

void OfflineRecognizerImpl::SetConfig(const OfflineRecognizerConfig &config) {
  config_ = config;
}

#if __ANDROID_API__ >= 9
template OfflineRecognizerImpl::OfflineRecognizerImpl(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);

template std::unique_ptr<OfflineRecognizerImpl> OfflineRecognizerImpl::Create(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

#if __OHOS__
template OfflineRecognizerImpl::OfflineRecognizerImpl(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
template std::unique_ptr<OfflineRecognizerImpl> OfflineRecognizerImpl::Create(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_mnn
