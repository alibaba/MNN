// sherpa-mnn/csrc/speaker-embedding-extractor-nemo-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/speaker-embedding-extractor-nemo-model.h"

#include <string>
#include <utility>
#include <vector>

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
#include "sherpa-mnn/csrc/speaker-embedding-extractor-nemo-model-meta-data.h"

namespace sherpa_mnn {

class SpeakerEmbeddingExtractorNeMoModel::Impl {
 public:
  explicit Impl(const SpeakerEmbeddingExtractorConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.model);
      Init(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.model);
      Init(buf.data(), buf.size());
    }
  }

  MNN::Express::VARP Compute(MNN::Express::VARP x, MNN::Express::VARP x_lens) const {
    std::vector<MNN::Express::VARP> inputs = {std::move(x), std::move(x_lens)};

    // output_names_ptr_[0] is logits
    // output_names_ptr_[1] is embeddings
    // so we use output_names_ptr_.data() + 1 here to extract only the
    // embeddings
    auto outputs = sess_->onForward(inputs);
    return std::move(outputs[0]);
  }

  MNNAllocator *Allocator() { return allocator_; }

  const SpeakerEmbeddingExtractorNeMoModelMetaData &GetMetaData() const {
    return meta_data_;
  }

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
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_data_.output_dim, "output_dim");
    SHERPA_ONNX_READ_META_DATA(meta_data_.feat_dim, "feat_dim");
    SHERPA_ONNX_READ_META_DATA(meta_data_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA(meta_data_.window_size_ms, "window_size_ms");
    SHERPA_ONNX_READ_META_DATA(meta_data_.window_stride_ms, "window_stride_ms");
    SHERPA_ONNX_READ_META_DATA_STR(meta_data_.language, "language");

    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(
        meta_data_.feature_normalize_type, "feature_normalize_type", "");

    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.window_type,
                                                "window_type", "povey");

    std::string framework;
    SHERPA_ONNX_READ_META_DATA_STR(framework, "framework");
    if (framework != "nemo") {
#if __OHOS__
      SHERPA_ONNX_LOGE("Expect a NeMo model, given: %{public}s",
                       framework.c_str());
#else
      SHERPA_ONNX_LOGE("Expect a NeMo model, given: %s", framework.c_str());
#endif
      exit(-1);
    }
  }

 private:
  SpeakerEmbeddingExtractorConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  SpeakerEmbeddingExtractorNeMoModelMetaData meta_data_;
};

SpeakerEmbeddingExtractorNeMoModel::SpeakerEmbeddingExtractorNeMoModel(
    const SpeakerEmbeddingExtractorConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
SpeakerEmbeddingExtractorNeMoModel::SpeakerEmbeddingExtractorNeMoModel(
    Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

SpeakerEmbeddingExtractorNeMoModel::~SpeakerEmbeddingExtractorNeMoModel() =
    default;

const SpeakerEmbeddingExtractorNeMoModelMetaData &
SpeakerEmbeddingExtractorNeMoModel::GetMetaData() const {
  return impl_->GetMetaData();
}

MNN::Express::VARP SpeakerEmbeddingExtractorNeMoModel::Compute(
    MNN::Express::VARP x, MNN::Express::VARP x_lens) const {
  return impl_->Compute(std::move(x), std::move(x_lens));
}

MNNAllocator *SpeakerEmbeddingExtractorNeMoModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template SpeakerEmbeddingExtractorNeMoModel::SpeakerEmbeddingExtractorNeMoModel(
    AAssetManager *mgr, const SpeakerEmbeddingExtractorConfig &config);
#endif

#if __OHOS__
template SpeakerEmbeddingExtractorNeMoModel::SpeakerEmbeddingExtractorNeMoModel(
    NativeResourceManager *mgr, const SpeakerEmbeddingExtractorConfig &config);
#endif

}  // namespace sherpa_mnn
