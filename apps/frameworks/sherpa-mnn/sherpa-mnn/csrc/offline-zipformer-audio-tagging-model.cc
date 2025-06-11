// sherpa-mnn/csrc/offline-zipformer-audio-tagging-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-zipformer-audio-tagging-model.h"

#include <string>
#include <vector>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/session.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

class OfflineZipformerAudioTaggingModel::Impl {
 public:
  explicit Impl(const AudioTaggingModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.zipformer.model);
    Init(buf.data(), buf.size());
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const AudioTaggingModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.zipformer.model);
    Init(buf.data(), buf.size());
  }
#endif

  MNN::Express::VARP Forward(MNN::Express::VARP features, MNN::Express::VARP features_length) {
    std::vector<MNN::Express::VARP> inputs = {std::move(features),
                                        std::move(features_length)};

    auto ans =
        sess_->onForward(inputs);
    return std::move(ans[0]);
  }

  int32_t NumEventClasses() const { return num_event_classes_; }

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
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }

    // get num_event_classes from the output[0].shape,
    // which is (N, num_event_classes)
    //num_event_classes_ =
    //    sess_->GetOutputTypeInfo(0)->getInfo()->dim[1];
  }

 private:
  AudioTaggingModelConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t num_event_classes_ = 0;
};

OfflineZipformerAudioTaggingModel::OfflineZipformerAudioTaggingModel(
    const AudioTaggingModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineZipformerAudioTaggingModel::OfflineZipformerAudioTaggingModel(
    AAssetManager *mgr, const AudioTaggingModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineZipformerAudioTaggingModel::~OfflineZipformerAudioTaggingModel() =
    default;

MNN::Express::VARP OfflineZipformerAudioTaggingModel::Forward(
    MNN::Express::VARP features, MNN::Express::VARP features_length) const {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineZipformerAudioTaggingModel::NumEventClasses() const {
  return impl_->NumEventClasses();
}

MNNAllocator *OfflineZipformerAudioTaggingModel::Allocator() const {
  return impl_->Allocator();
}

}  // namespace sherpa_mnn
