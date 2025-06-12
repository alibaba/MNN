// sherpa-mnn/csrc/offline-tts-kokoro-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-tts-kokoro-model.h"

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
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

class OfflineTtsKokoroModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto model_buf = ReadFile(config.kokoro.model);
    auto voices_buf = ReadFile(config.kokoro.voices);
    Init(model_buf.data(), model_buf.size(), voices_buf.data(),
         voices_buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto model_buf = ReadFile(mgr, config.kokoro.model);
    auto voices_buf = ReadFile(mgr, config.kokoro.voices);
    Init(model_buf.data(), model_buf.size(), voices_buf.data(),
         voices_buf.size());
  }

  const OfflineTtsKokoroModelMetaData &GetMetaData() const {
    return meta_data_;
  }

  MNN::Express::VARP Run(MNN::Express::VARP x, int32_t sid, float speed) {
    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::vector<int> x_shape = x->getInfo()->dim;
    if (x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Support only batch_size == 1. Given: %d",
                       static_cast<int32_t>(x_shape[0]));
      exit(-1);
    }

    // there is a 0 at the front and end of x
    int32_t len = static_cast<int32_t>(x_shape[1]) - 2;
    int32_t num_speakers = meta_data_.num_speakers;
    int32_t dim0 = style_dim_[0];
    int32_t dim1 = style_dim_[2];
    if (len >= dim0) {
      SHERPA_ONNX_LOGE("Bad things happened! %d vs %d", len, dim0);
      SHERPA_ONNX_EXIT(-1);
    }

    /*const*/ float *p = styles_.data() + sid * dim0 * dim1 + len * dim1;

    std::array<int, 2> style_embedding_shape = {1, dim1};
    MNN::Express::VARP style_embedding = MNNUtilsCreateTensor(
        memory_info, p, dim1, style_embedding_shape.data(),
        style_embedding_shape.size());

    int speed_shape = 1;

    MNN::Express::VARP speed_tensor =
        MNNUtilsCreateTensor(memory_info, &speed, 1, &speed_shape, 1);

    std::vector<MNN::Express::VARP> inputs = {
        std::move(x), std::move(style_embedding), std::move(speed_tensor)};

    auto out =
        sess_->onForward(inputs);

    return std::move(out[0]);
  }

 private:
  void Init(void *model_data, size_t model_data_length, const char *voices_data,
            size_t voices_data_length) {
    sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length,
                                           sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
    // get meta data
    MNNMeta meta_data = sess_->getInfo()->metaData;
    if (config_.debug) {
      std::ostringstream os;
      os << "---kokoro model---\n";
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
    SHERPA_ONNX_READ_META_DATA(meta_data_.has_espeak, "has_espeak");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.voice, "voice",
                                                "en-us");

    if (config_.debug) {
      std::vector<std::string> speaker_names;
      SHERPA_ONNX_READ_META_DATA_VEC_STRING(speaker_names, "speaker_names");
      std::ostringstream os;
      os << "\n";
      for (int32_t i = 0; i != speaker_names.size(); ++i) {
        os << i << "->" << speaker_names[i] << ", ";
      }
      os << "\n";

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    SHERPA_ONNX_READ_META_DATA_VEC(style_dim_, "style_dim");
    if (style_dim_.size() != 3) {
      SHERPA_ONNX_LOGE("style_dim should be 3-d, given: %d",
                       static_cast<int32_t>(style_dim_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (style_dim_[1] != 1) {
      SHERPA_ONNX_LOGE("style_dim[0] should be 1, given: %d", style_dim_[1]);
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t actual_num_floats = voices_data_length / sizeof(float);
    int32_t expected_num_floats =
        style_dim_[0] * style_dim_[2] * meta_data_.num_speakers;

    if (actual_num_floats != expected_num_floats) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Corrupted --kokoro-voices '%{public}s'. Expected #floats: "
          "%{public}d, actual: %{public}d",
          config_.kokoro.voices.c_str(), expected_num_floats,
          actual_num_floats);
#else
      SHERPA_ONNX_LOGE(
          "Corrupted --kokoro-voices '%s'. Expected #floats: %d, actual: %d",
          config_.kokoro.voices.c_str(), expected_num_floats,
          actual_num_floats);
#endif

      SHERPA_ONNX_EXIT(-1);
    }

    styles_ = std::vector<float>(
        reinterpret_cast<const float *>(voices_data),
        reinterpret_cast<const float *>(voices_data) + expected_num_floats);

    meta_data_.max_token_len = style_dim_[0];
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

  OfflineTtsKokoroModelMetaData meta_data_;
  std::vector<int32_t> style_dim_;

  // (num_speakers, style_dim_[0], style_dim_[2])
  std::vector<float> styles_;
};

OfflineTtsKokoroModel::OfflineTtsKokoroModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsKokoroModel::OfflineTtsKokoroModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsKokoroModel::~OfflineTtsKokoroModel() = default;

const OfflineTtsKokoroModelMetaData &OfflineTtsKokoroModel::GetMetaData()
    const {
  return impl_->GetMetaData();
}

MNN::Express::VARP OfflineTtsKokoroModel::Run(MNN::Express::VARP x, int sid /*= 0*/,
                                      float speed /*= 1.0*/) const {
  return impl_->Run(std::move(x), sid, speed);
}

#if __ANDROID_API__ >= 9
template OfflineTtsKokoroModel::OfflineTtsKokoroModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsKokoroModel::OfflineTtsKokoroModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_mnn
