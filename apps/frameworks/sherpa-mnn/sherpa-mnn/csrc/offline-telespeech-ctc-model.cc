// sherpa-mnn/csrc/offline-telespeech-ctc-model.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-telespeech-ctc-model.h"

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
#include "sherpa-mnn/csrc/session.h"
#include "sherpa-mnn/csrc/text-utils.h"
#include "sherpa-mnn/csrc/transpose.h"

namespace sherpa_mnn {

class OfflineTeleSpeechCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.telespeech_ctc);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.telespeech_ctc);
    Init(buf.data(), buf.size());
  }

  std::vector<MNN::Express::VARP> Forward(MNN::Express::VARP features,
                                  MNN::Express::VARP /*features_length*/) {
    std::vector<int> shape =
        features->getInfo()->dim;

    if (static_cast<int32_t>(shape[0]) != 1) {
      SHERPA_ONNX_LOGE("This model supports only batch size 1. Given %d",
                       static_cast<int32_t>(shape[0]));
    }

    auto out = sess_->onForward({features});

    std::vector<int> logits_shape = {1};
    MNN::Express::VARP logits_length = MNNUtilsCreateTensor<int>(
        allocator_, logits_shape.data(), logits_shape.size());

    int *dst = logits_length->writeMap<int>();
    dst[0] = out[0]->getInfo()->dim[0];

    // (T, B, C) -> (B, T, C)
    MNN::Express::VARP logits = Transpose01(allocator_, out[0]);

    std::vector<MNN::Express::VARP> ans;
    ans.reserve(2);
    ans.push_back(std::move(logits));
    ans.push_back(std::move(logits_length));

    return ans;
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }

  MNNAllocator *Allocator() { return allocator_; }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length,
                                           sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    MNNMeta meta_data = sess_->getInfo()->metaData;
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }
    auto iter = meta_data.find("vocab_size");
    if (iter != meta_data.end()){
      vocab_size_ = std::stoi(iter->second);
    }
  }

 private:
  OfflineModelConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 4;
};

OfflineTeleSpeechCtcModel::OfflineTeleSpeechCtcModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTeleSpeechCtcModel::OfflineTeleSpeechCtcModel(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTeleSpeechCtcModel::~OfflineTeleSpeechCtcModel() = default;

std::vector<MNN::Express::VARP> OfflineTeleSpeechCtcModel::Forward(
    MNN::Express::VARP features, MNN::Express::VARP features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineTeleSpeechCtcModel::VocabSize() const {
  return impl_->VocabSize();
}
int32_t OfflineTeleSpeechCtcModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

MNNAllocator *OfflineTeleSpeechCtcModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineTeleSpeechCtcModel::OfflineTeleSpeechCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineTeleSpeechCtcModel::OfflineTeleSpeechCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_mnn
