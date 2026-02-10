// sherpa-mnn/csrc/offline-tts-matcha-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-tts-matcha-model.h"

#include <algorithm>
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

namespace sherpa_mnn {

class OfflineTtsMatchaModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto model_path = config.matcha.acoustic_model.c_str();
    Init(model_path);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto model_path = config.matcha.acoustic_model.c_str();
    Init(model_path);
  }

  const OfflineTtsMatchaModelMetaData &GetMetaData() const {
    return meta_data_;
  }

  MNN::Express::VARP Run(MNN::Express::VARP x, int sid, float speed) {
    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::vector<int> x_shape = x->getInfo()->dim;
    if (x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Support only batch_size == 1. Given: %d",
                       static_cast<int32_t>(x_shape[0]));
      exit(-1);
    }

    int len = x_shape[1];
    int len_shape = 1;

    MNN::Express::VARP x_length =
        MNNUtilsCreateTensor(memory_info, &len, 1, &len_shape, 1);

    int scale_shape = 1;
    float noise_scale = config_.matcha.noise_scale;
    float length_scale = config_.matcha.length_scale;

    if (speed != 1 && speed > 0) {
      length_scale = 1. / speed;
    }

    MNN::Express::VARP noise_scale_tensor =
        MNNUtilsCreateTensor(memory_info, &noise_scale, 1, &scale_shape, 1);

    MNN::Express::VARP length_scale_tensor = MNNUtilsCreateTensor(
        memory_info, &length_scale, 1, &scale_shape, 1);

    MNN::Express::VARP sid_tensor =
        MNNUtilsCreateTensor(memory_info, &sid, 1, &scale_shape, 1);

    std::array<float, 2> scales = {noise_scale, length_scale};
    int scales_shape = 2;

    MNN::Express::VARP scales_tensor = MNNUtilsCreateTensor(
        memory_info, scales.data(), scales.size(), &scales_shape, 1);

    std::vector<MNN::Express::VARP> inputs;
    inputs.reserve(5);
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(x_length));
    if (input_names_[2] == "scales") {
      // for models from
      // https://github.com/shivammehta25/Matcha-TTS
      inputs.push_back(std::move(scales_tensor));
    } else {
      // for models from icefall
      inputs.push_back(std::move(noise_scale_tensor));
      inputs.push_back(std::move(length_scale_tensor));
    }

    if (input_names_.size() == 5 && input_names_.back() == "sid") {
      // for models from icefall
      inputs.push_back(std::move(sid_tensor));

      // Note that we have not supported multi-speaker tts models from
      // https://github.com/shivammehta25/Matcha-TTS
    }

    auto out =
        sess_->onForward(inputs);

    return std::move(out[0]);
  }

 private:
  void Init(const char *model_path) {
    sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, model_path,
                                           sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    MNNMeta meta_data = sess_->getInfo()->metaData;
    if (config_.debug) {
      std::ostringstream os;
      os << "---matcha model---\n";
      PrintModelMetadata(os, meta_data);

      os << "----------input names----------\n";
      int32_t i = 0;
      for (const auto &s : input_names_) {
        os << i << " " << s << "\n";
        ++i;
      }
      os << "----------output names----------\n";
      i = 0;
      for (const auto &s : output_names_) {
        os << i << " " << s << "\n";
        ++i;
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_data_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.version, "version", 1);
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_speakers, "n_speakers");
    SHERPA_ONNX_READ_META_DATA(meta_data_.jieba, "jieba");
    SHERPA_ONNX_READ_META_DATA(meta_data_.has_espeak, "has_espeak");
    SHERPA_ONNX_READ_META_DATA(meta_data_.use_eos_bos, "use_eos_bos");
    SHERPA_ONNX_READ_META_DATA(meta_data_.pad_id, "pad_id");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.voice, "voice",
                                                "en-us");
  }

 private:
  OfflineTtsModelConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  OfflineTtsMatchaModelMetaData meta_data_;
};

OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsMatchaModel::~OfflineTtsMatchaModel() = default;

const OfflineTtsMatchaModelMetaData &OfflineTtsMatchaModel::GetMetaData()
    const {
  return impl_->GetMetaData();
}

MNN::Express::VARP OfflineTtsMatchaModel::Run(MNN::Express::VARP x, int sid /*= 0*/,
                                      float speed /*= 1.0*/) const {
  return impl_->Run(std::move(x), sid, speed);
}

#if __ANDROID_API__ >= 9
template OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_mnn
