// sherpa-mnn/csrc/offline-speaker-segmentation-pyannote-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-speaker-segmentation-pyannote-model.h"

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
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/session.h"

namespace sherpa_mnn {

class OfflineSpeakerSegmentationPyannoteModel::Impl {
 public:
  explicit Impl(const OfflineSpeakerSegmentationModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.pyannote.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineSpeakerSegmentationModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.pyannote.model);
    Init(buf.data(), buf.size());
  }

  const OfflineSpeakerSegmentationPyannoteModelMetaData &GetModelMetaData()
      const {
    return meta_data_;
  }

  MNN::Express::VARP Forward(MNN::Express::VARP x) {
    auto out = sess_->onForward({x});

    return std::move(out[0]);
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
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_data_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA(meta_data_.window_size, "window_size");

    meta_data_.window_shift =
        static_cast<int32_t>(0.1 * meta_data_.window_size);

    SHERPA_ONNX_READ_META_DATA(meta_data_.receptive_field_size,
                               "receptive_field_size");
    SHERPA_ONNX_READ_META_DATA(meta_data_.receptive_field_shift,
                               "receptive_field_shift");
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_speakers, "num_speakers");
    SHERPA_ONNX_READ_META_DATA(meta_data_.powerset_max_classes,
                               "powerset_max_classes");
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_classes, "num_classes");
  }

 private:
  OfflineSpeakerSegmentationModelConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  OfflineSpeakerSegmentationPyannoteModelMetaData meta_data_;
};

OfflineSpeakerSegmentationPyannoteModel::
    OfflineSpeakerSegmentationPyannoteModel(
        const OfflineSpeakerSegmentationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSpeakerSegmentationPyannoteModel::
    OfflineSpeakerSegmentationPyannoteModel(
        Manager *mgr, const OfflineSpeakerSegmentationModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineSpeakerSegmentationPyannoteModel::
    ~OfflineSpeakerSegmentationPyannoteModel() = default;

const OfflineSpeakerSegmentationPyannoteModelMetaData &
OfflineSpeakerSegmentationPyannoteModel::GetModelMetaData() const {
  return impl_->GetModelMetaData();
}

MNN::Express::VARP OfflineSpeakerSegmentationPyannoteModel::Forward(
    MNN::Express::VARP x) const {
  return impl_->Forward(std::move(x));
}

#if __ANDROID_API__ >= 9
template OfflineSpeakerSegmentationPyannoteModel::
    OfflineSpeakerSegmentationPyannoteModel(
        AAssetManager *mgr,
        const OfflineSpeakerSegmentationModelConfig &config);
#endif

#if __OHOS__
template OfflineSpeakerSegmentationPyannoteModel::
    OfflineSpeakerSegmentationPyannoteModel(
        NativeResourceManager *mgr,
        const OfflineSpeakerSegmentationModelConfig &config);
#endif

}  // namespace sherpa_mnn
