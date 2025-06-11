// sherpa-mnn/csrc/offline-sense-voice-model.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-sense-voice-model.h"

#include <algorithm>
#include <string>
#include <utility>

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

namespace sherpa_mnn {

class OfflineSenseVoiceModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.sense_voice.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.sense_voice.model);
    Init(buf.data(), buf.size());
  }

  MNN::Express::VARP Forward(MNN::Express::VARP features, MNN::Express::VARP features_length,
                     MNN::Express::VARP language, MNN::Express::VARP text_norm) {
    std::vector<MNN::Express::VARP> inputs = {
        std::move(features),
        std::move(features_length),
        std::move(language),
        std::move(text_norm),
    };

    auto ans =
        sess_->onForward(inputs);
    return std::move(ans[0]);
  }

  const OfflineSenseVoiceModelMetaData& metaData() const {
    return meta_data_;
  }

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

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_data_.vocab_size, "vocab_size");
    SHERPA_ONNX_READ_META_DATA(meta_data_.window_size, "lfr_window_size");
    SHERPA_ONNX_READ_META_DATA(meta_data_.window_shift, "lfr_window_shift");
    SHERPA_ONNX_READ_META_DATA(meta_data_.normalize_samples,
                               "normalize_samples");

    SHERPA_ONNX_READ_META_DATA(meta_data_.with_itn_id, "with_itn");

    SHERPA_ONNX_READ_META_DATA(meta_data_.without_itn_id, "without_itn");

    int32_t lang_auto = 0;
    int32_t lang_zh = 0;
    int32_t lang_en = 0;
    int32_t lang_ja = 0;
    int32_t lang_ko = 0;
    int32_t lang_yue = 0;

    SHERPA_ONNX_READ_META_DATA(lang_auto, "lang_auto");
    SHERPA_ONNX_READ_META_DATA(lang_zh, "lang_zh");
    SHERPA_ONNX_READ_META_DATA(lang_en, "lang_en");
    SHERPA_ONNX_READ_META_DATA(lang_ja, "lang_ja");
    SHERPA_ONNX_READ_META_DATA(lang_ko, "lang_ko");
    SHERPA_ONNX_READ_META_DATA(lang_yue, "lang_yue");

    meta_data_.lang2id = {
        {"auto", lang_auto}, {"zh", lang_zh}, {"en", lang_en},
        {"ja", lang_ja},     {"ko", lang_ko}, {"yue", lang_yue},
    };

    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.neg_mean, "neg_mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.inv_stddev, "inv_stddev");
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

  OfflineSenseVoiceModelMetaData meta_data_;
};

OfflineSenseVoiceModel::OfflineSenseVoiceModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSenseVoiceModel::OfflineSenseVoiceModel(Manager *mgr,
                                               const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineSenseVoiceModel::~OfflineSenseVoiceModel() = default;

MNN::Express::VARP OfflineSenseVoiceModel::Forward(MNN::Express::VARP features,
                                           MNN::Express::VARP features_length,
                                           MNN::Express::VARP language,
                                           MNN::Express::VARP text_norm) const {
  return impl_->Forward(std::move(features), std::move(features_length),
                        std::move(language), std::move(text_norm));
}

const OfflineSenseVoiceModelMetaData &OfflineSenseVoiceModel::metaData()
    const {
  return impl_->metaData();
}

MNNAllocator *OfflineSenseVoiceModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineSenseVoiceModel::OfflineSenseVoiceModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineSenseVoiceModel::OfflineSenseVoiceModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_mnn
