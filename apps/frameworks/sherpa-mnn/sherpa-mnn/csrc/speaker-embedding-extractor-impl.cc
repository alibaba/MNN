// sherpa-mnn/csrc/speaker-embedding-extractor-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-mnn/csrc/speaker-embedding-extractor-impl.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/speaker-embedding-extractor-general-impl.h"
#include "sherpa-mnn/csrc/speaker-embedding-extractor-nemo-impl.h"

namespace sherpa_mnn {

namespace {

enum class ModelType : std::uint8_t {
  kWeSpeaker,
  k3dSpeaker,
  kNeMo,
  kUnknown,
};

}  // namespace

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
    SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
  }

  MNNAllocator* allocator;
  auto model_type =
      LookupCustomModelMetaData(meta_data, "framework", allocator);
  if (model_type.empty()) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n"
        "Please make sure you have added metadata to the model.\n\n"
        "For instance, you can use\n"
        "https://github.com/k2-fsa/sherpa-mnn/blob/master/scripts/wespeaker/"
        "add_meta_data.py"
        "to add metadata to models from WeSpeaker\n");
    return ModelType::kUnknown;
  }

  if (model_type == "wespeaker") {
    return ModelType::kWeSpeaker;
  } else if (model_type == "3d-speaker") {
    return ModelType::k3dSpeaker;
  } else if (model_type == "nemo") {
    return ModelType::kNeMo;
  } else {
#if __OHOS__
    SHERPA_ONNX_LOGE("Unsupported model_type: %{public}s", model_type.c_str());
#else
    SHERPA_ONNX_LOGE("Unsupported model_type: %s", model_type.c_str());
#endif
    return ModelType::kUnknown;
  }
}

std::unique_ptr<SpeakerEmbeddingExtractorImpl>
SpeakerEmbeddingExtractorImpl::Create(
    const SpeakerEmbeddingExtractorConfig &config) {
  ModelType model_type = ModelType::kUnknown;

  {
    auto buffer = ReadFile(config.model);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kWeSpeaker:
      // fall through
    case ModelType::k3dSpeaker:
      return std::make_unique<SpeakerEmbeddingExtractorGeneralImpl>(config);
    case ModelType::kNeMo:
      return std::make_unique<SpeakerEmbeddingExtractorNeMoImpl>(config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type for speaker embedding extractor!");
      return nullptr;
  }

  // unreachable code
  return nullptr;
}

template <typename Manager>
std::unique_ptr<SpeakerEmbeddingExtractorImpl>
SpeakerEmbeddingExtractorImpl::Create(
    Manager *mgr, const SpeakerEmbeddingExtractorConfig &config) {
  ModelType model_type = ModelType::kUnknown;

  {
    auto buffer = ReadFile(mgr, config.model);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kWeSpeaker:
      // fall through
    case ModelType::k3dSpeaker:
      return std::make_unique<SpeakerEmbeddingExtractorGeneralImpl>(mgr,
                                                                    config);
    case ModelType::kNeMo:
      return std::make_unique<SpeakerEmbeddingExtractorNeMoImpl>(mgr, config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE(
          "Unknown model type in for speaker embedding extractor!");
      return nullptr;
  }

  // unreachable code
  return nullptr;
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<SpeakerEmbeddingExtractorImpl>
SpeakerEmbeddingExtractorImpl::Create(
    AAssetManager *mgr, const SpeakerEmbeddingExtractorConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<SpeakerEmbeddingExtractorImpl>
SpeakerEmbeddingExtractorImpl::Create(
    NativeResourceManager *mgr, const SpeakerEmbeddingExtractorConfig &config);
#endif

}  // namespace sherpa_mnn
