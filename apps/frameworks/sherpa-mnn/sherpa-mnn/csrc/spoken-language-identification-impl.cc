// sherpa-mnn/csrc/spoken-language-identification-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-mnn/csrc/spoken-language-identification-impl.h"

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/spoken-language-identification-whisper-impl.h"

namespace sherpa_mnn {

namespace {

enum class ModelType : std::uint8_t {
  kWhisper,
  kUnknown,
};

}

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
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }

  MNNAllocator* allocator;
  auto model_type =
      LookupCustomModelMetaData(meta_data, "model_type", allocator);
  if (model_type.empty()) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n"
        "Please make sure you have added metadata to the model.\n\n"
        "For instance, you can use\n"
        "https://github.com/k2-fsa/sherpa-mnn/blob/master/scripts/whisper/"
        "export-onnx.py "
        "to add metadata to models from whisper\n");
    return ModelType::kUnknown;
  }

  if (model_type.find("whisper") == 0) {
    return ModelType::kWhisper;
  } else {
    SHERPA_ONNX_LOGE("Unsupported model_type: %s", model_type.c_str());
    return ModelType::kUnknown;
  }
}

std::unique_ptr<SpokenLanguageIdentificationImpl>
SpokenLanguageIdentificationImpl::Create(
    const SpokenLanguageIdentificationConfig &config) {
  ModelType model_type = ModelType::kUnknown;
  {
    if (config.whisper.encoder.empty()) {
      SHERPA_ONNX_LOGE("Only whisper models are supported at present");
      exit(-1);
    }
    auto buffer = ReadFile(config.whisper.encoder);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kWhisper:
      return std::make_unique<SpokenLanguageIdentificationWhisperImpl>(config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE(
          "Unknown model type for spoken language identification!");
      return nullptr;
  }

  // unreachable code
  return nullptr;
}

#if __ANDROID_API__ >= 9
std::unique_ptr<SpokenLanguageIdentificationImpl>
SpokenLanguageIdentificationImpl::Create(
    AAssetManager *mgr, const SpokenLanguageIdentificationConfig &config) {
  ModelType model_type = ModelType::kUnknown;
  {
    if (config.whisper.encoder.empty()) {
      SHERPA_ONNX_LOGE("Only whisper models are supported at present");
      exit(-1);
    }
    auto buffer = ReadFile(mgr, config.whisper.encoder);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kWhisper:
      return std::make_unique<SpokenLanguageIdentificationWhisperImpl>(mgr,
                                                                       config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE(
          "Unknown model type for spoken language identification!");
      return nullptr;
  }

  // unreachable code
  return nullptr;
}
#endif

}  // namespace sherpa_mnn
