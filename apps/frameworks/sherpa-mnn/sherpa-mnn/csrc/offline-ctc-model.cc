// sherpa-mnn/csrc/offline-ctc-model.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-ctc-model.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/offline-nemo-enc-dec-ctc-model.h"
#include "sherpa-mnn/csrc/offline-tdnn-ctc-model.h"
#include "sherpa-mnn/csrc/offline-telespeech-ctc-model.h"
#include "sherpa-mnn/csrc/offline-wenet-ctc-model.h"
#include "sherpa-mnn/csrc/offline-zipformer-ctc-model.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace {

enum class ModelType : std::uint8_t {
  kEncDecCTCModelBPE,
  kEncDecCTCModel,
  kEncDecHybridRNNTCTCBPEModel,
  kTdnn,
  kZipformerCtc,
  kWenetCtc,
  kTeleSpeechCtc,
  kUnknown,
};

}  // namespace

namespace sherpa_mnn {

static ModelType GetModelType(char *model_data, size_t model_data_length,
                              bool debug) {
  MNNEnv env;
  std::shared_ptr<MNN::Express::Executor::RuntimeManager> sess_opts;
  
  

  auto sess = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length,
                                             sess_opts));

  MNNMeta meta_data = sess->getInfo()->metaData;
  if (debug) {
    std::ostringstream os;
    PrintModelMetadata(os, meta_data);
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
  }

  MNNAllocator* allocator;
  auto model_type =
      LookupCustomModelMetaData(meta_data, "model_type", allocator);
  if (model_type.empty()) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n"
        "If you are using models from NeMo, please refer to\n"
        "https://huggingface.co/csukuangfj/"
        "sherpa-mnn-nemo-ctc-en-citrinet-512/blob/main/add-model-metadata.py\n"
        "or "
        "https://github.com/k2-fsa/sherpa-mnn/tree/master/scripts/nemo/"
        "fast-conformer-hybrid-transducer-ctc\n"
        "If you are using models from WeNet, please refer to\n"
        "https://github.com/k2-fsa/sherpa-mnn/blob/master/scripts/wenet/"
        "run.sh\n"
        "If you are using models from TeleSpeech, please refer to\n"
        "https://github.com/k2-fsa/sherpa-mnn/blob/master/scripts/tele-speech/"
        "add-metadata.py"
        "\n"
        "for how to add metadta to model.onnx\n");
    return ModelType::kUnknown;
  }

  if (model_type == "EncDecCTCModelBPE") {
    return ModelType::kEncDecCTCModelBPE;
  } else if (model_type == "EncDecCTCModel") {
    return ModelType::kEncDecCTCModel;
  } else if (model_type == "EncDecHybridRNNTCTCBPEModel") {
    return ModelType::kEncDecHybridRNNTCTCBPEModel;
  } else if (model_type == "tdnn") {
    return ModelType::kTdnn;
  } else if (model_type == "zipformer2_ctc") {
    return ModelType::kZipformerCtc;
  } else if (model_type == "wenet_ctc") {
    return ModelType::kWenetCtc;
  } else if (model_type == "telespeech_ctc") {
    return ModelType::kTeleSpeechCtc;
  } else {
    SHERPA_ONNX_LOGE("Unsupported model_type: %s", model_type.c_str());
    return ModelType::kUnknown;
  }
}

std::unique_ptr<OfflineCtcModel> OfflineCtcModel::Create(
    const OfflineModelConfig &config) {
  // TODO(fangjun): Refactor it. We don't need to use model_type here
  ModelType model_type = ModelType::kUnknown;

  std::string filename;
  if (!config.nemo_ctc.model.empty()) {
    filename = config.nemo_ctc.model;
  } else if (!config.tdnn.model.empty()) {
    filename = config.tdnn.model;
  } else if (!config.zipformer_ctc.model.empty()) {
    filename = config.zipformer_ctc.model;
  } else if (!config.wenet_ctc.model.empty()) {
    filename = config.wenet_ctc.model;
  } else if (!config.telespeech_ctc.empty()) {
    filename = config.telespeech_ctc;
  } else {
    SHERPA_ONNX_LOGE("Please specify a CTC model");
    exit(-1);
  }

  {
    auto buffer = ReadFile(filename);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kEncDecCTCModelBPE:
    case ModelType::kEncDecCTCModel:
      return std::make_unique<OfflineNemoEncDecCtcModel>(config);
    case ModelType::kEncDecHybridRNNTCTCBPEModel:
      return std::make_unique<OfflineNemoEncDecHybridRNNTCTCBPEModel>(config);
    case ModelType::kTdnn:
      return std::make_unique<OfflineTdnnCtcModel>(config);
    case ModelType::kZipformerCtc:
      return std::make_unique<OfflineZipformerCtcModel>(config);
    case ModelType::kWenetCtc:
      return std::make_unique<OfflineWenetCtcModel>(config);
    case ModelType::kTeleSpeechCtc:
      return std::make_unique<OfflineTeleSpeechCtcModel>(config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type in offline CTC!");
      return nullptr;
  }

  return nullptr;
}

template <typename Manager>
std::unique_ptr<OfflineCtcModel> OfflineCtcModel::Create(
    Manager *mgr, const OfflineModelConfig &config) {
  // TODO(fangjun): Refactor it. We don't need to use model_type here
  ModelType model_type = ModelType::kUnknown;

  std::string filename;
  if (!config.nemo_ctc.model.empty()) {
    filename = config.nemo_ctc.model;
  } else if (!config.tdnn.model.empty()) {
    filename = config.tdnn.model;
  } else if (!config.zipformer_ctc.model.empty()) {
    filename = config.zipformer_ctc.model;
  } else if (!config.wenet_ctc.model.empty()) {
    filename = config.wenet_ctc.model;
  } else if (!config.telespeech_ctc.empty()) {
    filename = config.telespeech_ctc;
  } else {
    SHERPA_ONNX_LOGE("Please specify a CTC model");
    exit(-1);
  }

  {
    auto buffer = ReadFile(mgr, filename);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kEncDecCTCModelBPE:
    case ModelType::kEncDecCTCModel:
      return std::make_unique<OfflineNemoEncDecCtcModel>(mgr, config);
    case ModelType::kEncDecHybridRNNTCTCBPEModel:
      return std::make_unique<OfflineNemoEncDecHybridRNNTCTCBPEModel>(mgr,
                                                                      config);
    case ModelType::kTdnn:
      return std::make_unique<OfflineTdnnCtcModel>(mgr, config);
    case ModelType::kZipformerCtc:
      return std::make_unique<OfflineZipformerCtcModel>(mgr, config);
    case ModelType::kWenetCtc:
      return std::make_unique<OfflineWenetCtcModel>(mgr, config);
    case ModelType::kTeleSpeechCtc:
      return std::make_unique<OfflineTeleSpeechCtcModel>(mgr, config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type in offline CTC!");
      return nullptr;
  }

  return nullptr;
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineCtcModel> OfflineCtcModel::Create(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflineCtcModel> OfflineCtcModel::Create(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_mnn
