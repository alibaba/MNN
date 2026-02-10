// sherpa-mnn/csrc/online-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo
#include "sherpa-mnn/csrc/online-transducer-model.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/online-conformer-transducer-model.h"
#include "sherpa-mnn/csrc/online-ebranchformer-transducer-model.h"
#include "sherpa-mnn/csrc/online-lstm-transducer-model.h"
#include "sherpa-mnn/csrc/online-zipformer-transducer-model.h"
#include "sherpa-mnn/csrc/online-zipformer2-transducer-model.h"
#include "sherpa-mnn/csrc/onnx-utils.h"

namespace {

enum class ModelType : std::uint8_t {
  kConformer,
  kEbranchformer,
  kLstm,
  kZipformer,
  kZipformer2,
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
    SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
  }

  MNNAllocator* allocator;
  auto model_type =
      LookupCustomModelMetaData(meta_data, "model_type", allocator);
  if (model_type.empty()) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n"
        "Please make sure you are using the latest export-onnx.py from icefall "
        "to export your transducer models");
    return ModelType::kUnknown;
  }

  if (model_type == "conformer") {
    return ModelType::kConformer;
  } else if (model_type == "ebranchformer") {
    return ModelType::kEbranchformer;
  } else if (model_type == "lstm") {
    return ModelType::kLstm;
  } else if (model_type == "zipformer") {
    return ModelType::kZipformer;
  } else if (model_type == "zipformer2") {
    return ModelType::kZipformer2;
  } else {
    SHERPA_ONNX_LOGE("Unsupported model_type: %s", model_type.c_str());
    return ModelType::kUnknown;
  }
}

std::unique_ptr<OnlineTransducerModel> OnlineTransducerModel::Create(
    const OnlineModelConfig &config) {
  if (!config.model_type.empty()) {
    const auto &model_type = config.model_type;
    if (model_type == "conformer") {
      return std::make_unique<OnlineConformerTransducerModel>(config);
    } else if (model_type == "ebranchformer") {
      return std::make_unique<OnlineEbranchformerTransducerModel>(config);
    } else if (model_type == "lstm") {
      return std::make_unique<OnlineLstmTransducerModel>(config);
    } else if (model_type == "zipformer") {
      return std::make_unique<OnlineZipformerTransducerModel>(config);
    } else if (model_type == "zipformer2") {
      return std::make_unique<OnlineZipformer2TransducerModel>(config);
    } else {
      SHERPA_ONNX_LOGE(
          "Invalid model_type: %s. Trying to load the model to get its type",
          model_type.c_str());
    }
  }
  ModelType model_type = ModelType::kUnknown;

  {
    auto buffer = ReadFile(config.transducer.encoder);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kConformer:
      return std::make_unique<OnlineConformerTransducerModel>(config);
    case ModelType::kEbranchformer:
      return std::make_unique<OnlineEbranchformerTransducerModel>(config);
    case ModelType::kLstm:
      return std::make_unique<OnlineLstmTransducerModel>(config);
    case ModelType::kZipformer:
      return std::make_unique<OnlineZipformerTransducerModel>(config);
    case ModelType::kZipformer2:
      return std::make_unique<OnlineZipformer2TransducerModel>(config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type in online transducer!");
      return nullptr;
  }

  // unreachable code
  return nullptr;
}

MNN::Express::VARP OnlineTransducerModel::BuildDecoderInput(
    const std::vector<OnlineTransducerDecoderResult> &results) {
  int32_t batch_size = static_cast<int32_t>(results.size());
  int32_t context_size = ContextSize();
  std::array<int, 2> shape{batch_size, context_size};
  MNN::Express::VARP decoder_input = MNNUtilsCreateTensor<int>(
      Allocator(), shape.data(), shape.size());
  int *p = decoder_input->writeMap<int>();

  for (const auto &r : results) {
    const int *begin = r.tokens.data() + r.tokens.size() - context_size;
    const int *end = r.tokens.data() + r.tokens.size();
    std::copy(begin, end, p);
    p += context_size;
  }
  return decoder_input;
}

MNN::Express::VARP OnlineTransducerModel::BuildDecoderInput(
    const std::vector<Hypothesis> &hyps) {
  int32_t batch_size = static_cast<int32_t>(hyps.size());
  int32_t context_size = ContextSize();
  std::array<int, 2> shape{batch_size, context_size};
  MNN::Express::VARP decoder_input = MNNUtilsCreateTensor<int>(
      Allocator(), shape.data(), shape.size());
  int *p = decoder_input->writeMap<int>();

  for (const auto &h : hyps) {
    std::copy(h.ys.end() - context_size, h.ys.end(), p);
    p += context_size;
  }
  return decoder_input;
}

template <typename Manager>
std::unique_ptr<OnlineTransducerModel> OnlineTransducerModel::Create(
    Manager *mgr, const OnlineModelConfig &config) {
  if (!config.model_type.empty()) {
    const auto &model_type = config.model_type;
    if (model_type == "conformer") {
      return std::make_unique<OnlineConformerTransducerModel>(mgr, config);
    } else if (model_type == "ebranchformer") {
      return std::make_unique<OnlineEbranchformerTransducerModel>(mgr, config);
    } else if (model_type == "lstm") {
      return std::make_unique<OnlineLstmTransducerModel>(mgr, config);
    } else if (model_type == "zipformer") {
      return std::make_unique<OnlineZipformerTransducerModel>(mgr, config);
    } else if (model_type == "zipformer2") {
      return std::make_unique<OnlineZipformer2TransducerModel>(mgr, config);
    } else {
      SHERPA_ONNX_LOGE(
          "Invalid model_type: %s. Trying to load the model to get its type",
          model_type.c_str());
    }
  }

  auto buffer = ReadFile(mgr, config.transducer.encoder);
  auto model_type = GetModelType(buffer.data(), buffer.size(), config.debug);

  switch (model_type) {
    case ModelType::kConformer:
      return std::make_unique<OnlineConformerTransducerModel>(mgr, config);
    case ModelType::kEbranchformer:
      return std::make_unique<OnlineEbranchformerTransducerModel>(mgr, config);
    case ModelType::kLstm:
      return std::make_unique<OnlineLstmTransducerModel>(mgr, config);
    case ModelType::kZipformer:
      return std::make_unique<OnlineZipformerTransducerModel>(mgr, config);
    case ModelType::kZipformer2:
      return std::make_unique<OnlineZipformer2TransducerModel>(mgr, config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type in online transducer!");
      return nullptr;
  }

  // unreachable code
  return nullptr;
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OnlineTransducerModel> OnlineTransducerModel::Create(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OnlineTransducerModel> OnlineTransducerModel::Create(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_mnn
