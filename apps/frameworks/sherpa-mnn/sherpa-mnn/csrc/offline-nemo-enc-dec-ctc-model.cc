// sherpa-mnn/csrc/offline-nemo-enc-dec-ctc-model.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-nemo-enc-dec-ctc-model.h"

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

class OfflineNemoEncDecCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.nemo_ctc.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.nemo_ctc.model);
    Init(buf.data(), buf.size());
  }

  std::vector<MNN::Express::VARP> Forward(MNN::Express::VARP features,
                                  MNN::Express::VARP features_length) {
    std::vector<int> shape =
        features_length->getInfo()->dim;

    MNN::Express::VARP out_features_length = MNNUtilsCreateTensor<int>(
        allocator_, shape.data(), shape.size());

    const int *src = features_length->readMap<int>();
    int *dst = out_features_length->writeMap<int>();
    for (int i = 0; i != shape[0]; ++i) {
      dst[i] = src[i] / subsampling_factor_;
    }

    // (B, T, C) -> (B, C, T)
    features = Transpose12(allocator_, features);

    std::vector<MNN::Express::VARP> inputs = {std::move(features),
                                        std::move(features_length)};
    auto out =
        sess_->onForward(inputs);

    std::vector<MNN::Express::VARP> ans;
    ans.reserve(2);
    ans.push_back(std::move(out[0]));
    ans.push_back(std::move(out_features_length));
    return ans;
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }

  MNNAllocator *Allocator() { return allocator_; }

  std::string FeatureNormalizationMethod() const { return normalize_type_; }

  bool IsGigaAM() const { return is_giga_am_; }

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

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA_STR_ALLOW_EMPTY(normalize_type_,
                                               "normalize_type");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(is_giga_am_, "is_giga_am", 0);
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
  int32_t subsampling_factor_ = 0;
  std::string normalize_type_;

  // it is 1 for models from
  // https://github.com/salute-developers/GigaAM
  int32_t is_giga_am_ = 0;
};

OfflineNemoEncDecCtcModel::OfflineNemoEncDecCtcModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineNemoEncDecCtcModel::OfflineNemoEncDecCtcModel(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineNemoEncDecCtcModel::~OfflineNemoEncDecCtcModel() = default;

std::vector<MNN::Express::VARP> OfflineNemoEncDecCtcModel::Forward(
    MNN::Express::VARP features, MNN::Express::VARP features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineNemoEncDecCtcModel::VocabSize() const {
  return impl_->VocabSize();
}
int32_t OfflineNemoEncDecCtcModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

MNNAllocator *OfflineNemoEncDecCtcModel::Allocator() const {
  return impl_->Allocator();
}

std::string OfflineNemoEncDecCtcModel::FeatureNormalizationMethod() const {
  return impl_->FeatureNormalizationMethod();
}

bool OfflineNemoEncDecCtcModel::IsGigaAM() const { return impl_->IsGigaAM(); }

#if __ANDROID_API__ >= 9
template OfflineNemoEncDecCtcModel::OfflineNemoEncDecCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineNemoEncDecCtcModel::OfflineNemoEncDecCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_mnn
