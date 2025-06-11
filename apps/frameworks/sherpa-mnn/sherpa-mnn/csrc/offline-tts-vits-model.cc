// sherpa-mnn/csrc/offline-tts-vits-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-tts-vits-model.h"

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

class OfflineTtsVitsModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config.vits.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.vits.model);
    Init(buf.data(), buf.size());
  }

  MNN::Express::VARP Run(MNN::Express::VARP x, int sid, float speed) {
    if (meta_data_.is_piper || meta_data_.is_coqui) {
      return RunVitsPiperOrCoqui(std::move(x), sid, speed);
    }

    return RunVits(std::move(x), sid, speed);
  }

  MNN::Express::VARP Run(MNN::Express::VARP x, MNN::Express::VARP tones, int sid, float speed) {
    if (meta_data_.num_speakers == 1) {
      // For MeloTTS, we hardcode sid to the one contained in the meta data
      sid = meta_data_.speaker_id;
    }

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
    float noise_scale = config_.vits.noise_scale;
    float length_scale = config_.vits.length_scale;
    float noise_scale_w = config_.vits.noise_scale_w;

    if (speed != 1 && speed > 0) {
      length_scale = 1. / speed;
    }

    MNN::Express::VARP noise_scale_tensor =
        MNNUtilsCreateTensor(memory_info, &noise_scale, 1, &scale_shape, 1);

    MNN::Express::VARP length_scale_tensor = MNNUtilsCreateTensor(
        memory_info, &length_scale, 1, &scale_shape, 1);

    MNN::Express::VARP noise_scale_w_tensor = MNNUtilsCreateTensor(
        memory_info, &noise_scale_w, 1, &scale_shape, 1);

    MNN::Express::VARP sid_tensor =
        MNNUtilsCreateTensor(memory_info, &sid, 1, &scale_shape, 1);

    std::vector<MNN::Express::VARP> inputs;
    inputs.reserve(7);
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(x_length));
    inputs.push_back(std::move(tones));
    inputs.push_back(std::move(sid_tensor));
    inputs.push_back(std::move(noise_scale_tensor));
    inputs.push_back(std::move(length_scale_tensor));
    inputs.push_back(std::move(noise_scale_w_tensor));

    auto out =
        sess_->onForward(inputs);

    return std::move(out[0]);
  }

  const OfflineTtsVitsModelMetaData &GetMetaData() const { return meta_data_; }

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
      os << "---vits model---\n";
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
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.add_blank, "add_blank",
                                            0);

    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.speaker_id, "speaker_id",
                                            0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.version, "version", 0);
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_speakers, "n_speakers");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.punctuations,
                                                "punctuation", "");
    SHERPA_ONNX_READ_META_DATA_STR(meta_data_.language, "language");

    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.voice, "voice", "");

    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.frontend, "frontend",
                                                "");

    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.jieba, "jieba", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.blank_id, "blank_id", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.bos_id, "bos_id", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.eos_id, "eos_id", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.use_eos_bos,
                                            "use_eos_bos", 1);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.pad_id, "pad_id", 0);

    std::string comment;
    SHERPA_ONNX_READ_META_DATA_STR(comment, "comment");

    if (comment.find("piper") != std::string::npos) {
      meta_data_.is_piper = true;
    }

    if (comment.find("coqui") != std::string::npos) {
      meta_data_.is_coqui = true;
    }

    if (comment.find("icefall") != std::string::npos) {
      meta_data_.is_icefall = true;
    }

    if (comment.find("melo") != std::string::npos) {
      meta_data_.is_melo_tts = true;
      int32_t expected_version = 2;
      if (meta_data_.version < expected_version) {
        SHERPA_ONNX_LOGE(
            "Please download the latest MeloTTS model and retry. Current "
            "version: %d. Expected version: %d",
            meta_data_.version, expected_version);
        exit(-1);
      }

      // NOTE(fangjun):
      // version 0 is the first version
      // version 2: add jieba=1 to the metadata
    }
  }

  MNN::Express::VARP RunVitsPiperOrCoqui(MNN::Express::VARP x, int sid, float speed) {
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

    float noise_scale = config_.vits.noise_scale;
    float length_scale = config_.vits.length_scale;
    float noise_scale_w = config_.vits.noise_scale_w;

    if (speed != 1 && speed > 0) {
      length_scale = 1. / speed;
    }
    std::array<float, 3> scales = {noise_scale, length_scale, noise_scale_w};

    int scale_shape = 3;

    MNN::Express::VARP scales_tensor = MNNUtilsCreateTensor(
        memory_info, scales.data(), scales.size(), &scale_shape, 1);

    int sid_shape = 1;
    MNN::Express::VARP sid_tensor =
        MNNUtilsCreateTensor(memory_info, &sid, 1, &sid_shape, 1);

    int lang_id_shape = 1;
    int lang_id = 0;
    MNN::Express::VARP lang_id_tensor =
        MNNUtilsCreateTensor(memory_info, &lang_id, 1, &lang_id_shape, 1);

    std::vector<MNN::Express::VARP> inputs;
    inputs.reserve(5);
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(x_length));
    inputs.push_back(std::move(scales_tensor));

    if (input_names_.size() >= 4 && input_names_[3] == "sid") {
      inputs.push_back(std::move(sid_tensor));
    }

    if (input_names_.size() >= 5 && input_names_[4] == "langid") {
      inputs.push_back(std::move(lang_id_tensor));
    }

    auto out =
        sess_->onForward(inputs);

    return std::move(out[0]);
  }

  MNN::Express::VARP RunVits(MNN::Express::VARP x, int sid, float speed) {
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
    float noise_scale = config_.vits.noise_scale;
    float length_scale = config_.vits.length_scale;
    float noise_scale_w = config_.vits.noise_scale_w;

    if (speed != 1 && speed > 0) {
      length_scale = 1. / speed;
    }

    MNN::Express::VARP noise_scale_tensor =
        MNNUtilsCreateTensor(memory_info, &noise_scale, 1, &scale_shape, 1);

    MNN::Express::VARP length_scale_tensor = MNNUtilsCreateTensor(
        memory_info, &length_scale, 1, &scale_shape, 1);

    MNN::Express::VARP noise_scale_w_tensor = MNNUtilsCreateTensor(
        memory_info, &noise_scale_w, 1, &scale_shape, 1);

    MNN::Express::VARP sid_tensor =
        MNNUtilsCreateTensor(memory_info, &sid, 1, &scale_shape, 1);

    std::vector<MNN::Express::VARP> inputs;
    inputs.reserve(6);
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(x_length));
    inputs.push_back(std::move(noise_scale_tensor));
    inputs.push_back(std::move(length_scale_tensor));
    inputs.push_back(std::move(noise_scale_w_tensor));

    if (input_names_.size() == 6 &&
        (input_names_.back() == "sid" || input_names_.back() == "speaker")) {
      inputs.push_back(std::move(sid_tensor));
    }

    auto out =
        sess_->onForward(inputs);

    return std::move(out[0]);
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

  OfflineTtsVitsModelMetaData meta_data_;
};

OfflineTtsVitsModel::OfflineTtsVitsModel(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsVitsModel::OfflineTtsVitsModel(Manager *mgr,
                                         const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsVitsModel::~OfflineTtsVitsModel() = default;

MNN::Express::VARP OfflineTtsVitsModel::Run(MNN::Express::VARP x, int sid /*=0*/,
                                    float speed /*= 1.0*/) {
  return impl_->Run(std::move(x), sid, speed);
}

MNN::Express::VARP OfflineTtsVitsModel::Run(MNN::Express::VARP x, MNN::Express::VARP tones,
                                    int sid /*= 0*/,
                                    float speed /*= 1.0*/) const {
  return impl_->Run(std::move(x), std::move(tones), sid, speed);
}

const OfflineTtsVitsModelMetaData &OfflineTtsVitsModel::GetMetaData() const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineTtsVitsModel::OfflineTtsVitsModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsVitsModel::OfflineTtsVitsModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_mnn
