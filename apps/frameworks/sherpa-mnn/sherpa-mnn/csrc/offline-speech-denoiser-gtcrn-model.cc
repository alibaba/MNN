// sherpa-mnn/csrc/offline-speech-denoiser-gtcrn-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-speech-denoiser-gtcrn-model.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/session.h"
#include "sherpa-mnn/csrc/text-utils.h"

namespace sherpa_mnn {

class OfflineSpeechDenoiserGtcrnModel::Impl {
 public:
  explicit Impl(const OfflineSpeechDenoiserModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.gtcrn.model);
      Init(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineSpeechDenoiserModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.gtcrn.model);
      Init(buf.data(), buf.size());
    }
  }

  const OfflineSpeechDenoiserGtcrnModelMetaData &GetMetaData() const {
    return meta_;
  }

  States GetInitStates() const {
    MNN::Express::VARP conv_cache = MNNUtilsCreateTensor<float>(
        allocator_, meta_.conv_cache_shape.data(),
        meta_.conv_cache_shape.size());

    MNN::Express::VARP tra_cache = MNNUtilsCreateTensor<float>(
        allocator_, meta_.tra_cache_shape.data(), meta_.tra_cache_shape.size());

    MNN::Express::VARP inter_cache = MNNUtilsCreateTensor<float>(
        allocator_, meta_.inter_cache_shape.data(),
        meta_.inter_cache_shape.size());

    Fill<float>(conv_cache, 0);
    Fill<float>(tra_cache, 0);
    Fill<float>(inter_cache, 0);

    std::vector<MNN::Express::VARP> states;

    states.reserve(3);
    states.push_back(std::move(conv_cache));
    states.push_back(std::move(tra_cache));
    states.push_back(std::move(inter_cache));

    return states;
  }

  std::pair<MNN::Express::VARP, States> Run(MNN::Express::VARP x, States states) const {
    std::vector<MNN::Express::VARP> inputs;
    inputs.reserve(1 + states.size());
    inputs.push_back(std::move(x));
    for (auto &s : states) {
      inputs.push_back(std::move(s));
    }

    auto out =
        sess_->onForward(inputs);

    std::vector<MNN::Express::VARP> next_states;
    next_states.reserve(out.size() - 1);
    for (int32_t k = 1; k < out.size(); ++k) {
      next_states.push_back(std::move(out[k]));
    }

    return {std::move(out[0]), std::move(next_states)};
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length,
                                           sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    MNNMeta meta_data = sess_->getInfo()->metaData;
    if (config_.debug) {
      std::ostringstream os;
      os << "---gtcrn model---\n";
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

    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR(model_type, "model_type");
    if (model_type != "gtcrn") {
      SHERPA_ONNX_LOGE("Expect model type 'gtcrn'. Given: '%s'",
                       model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(meta_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA(meta_.n_fft, "n_fft");
    SHERPA_ONNX_READ_META_DATA(meta_.hop_length, "hop_length");
    SHERPA_ONNX_READ_META_DATA(meta_.window_length, "window_length");
    SHERPA_ONNX_READ_META_DATA_STR(meta_.window_type, "window_type");
    SHERPA_ONNX_READ_META_DATA(meta_.version, "version");

    SHERPA_ONNX_READ_META_DATA_VEC(meta_.conv_cache_shape, "conv_cache_shape");
    SHERPA_ONNX_READ_META_DATA_VEC(meta_.tra_cache_shape, "tra_cache_shape");
    SHERPA_ONNX_READ_META_DATA_VEC(meta_.inter_cache_shape,
                                   "inter_cache_shape");
  }

 private:
  OfflineSpeechDenoiserModelConfig config_;
  OfflineSpeechDenoiserGtcrnModelMetaData meta_;

  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineSpeechDenoiserGtcrnModel::~OfflineSpeechDenoiserGtcrnModel() = default;

OfflineSpeechDenoiserGtcrnModel::OfflineSpeechDenoiserGtcrnModel(
    const OfflineSpeechDenoiserModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSpeechDenoiserGtcrnModel::OfflineSpeechDenoiserGtcrnModel(
    Manager *mgr, const OfflineSpeechDenoiserModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineSpeechDenoiserGtcrnModel::States
OfflineSpeechDenoiserGtcrnModel::GetInitStates() const {
  return impl_->GetInitStates();
}

std::pair<MNN::Express::VARP, OfflineSpeechDenoiserGtcrnModel::States>
OfflineSpeechDenoiserGtcrnModel::Run(MNN::Express::VARP x, States states) const {
  return impl_->Run(std::move(x), std::move(states));
}

const OfflineSpeechDenoiserGtcrnModelMetaData &
OfflineSpeechDenoiserGtcrnModel::GetMetaData() const {
  return impl_->GetMetaData();
}

}  // namespace sherpa_mnn
