// sherpa-mnn/csrc/offline-transducer-nemo-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-transducer-nemo-model.h"

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
#include "sherpa-mnn/csrc/offline-transducer-decoder.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/session.h"
#include "sherpa-mnn/csrc/transpose.h"

namespace sherpa_mnn {

class OfflineTransducerNeMoModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.transducer.encoder_filename);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.decoder_filename);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.joiner_filename);
      InitJoiner(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.transducer.encoder_filename);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.decoder_filename);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.joiner_filename);
      InitJoiner(buf.data(), buf.size());
    }
  }

  std::vector<MNN::Express::VARP> RunEncoder(MNN::Express::VARP features,
                                     MNN::Express::VARP features_length) {
    // (B, T, C) -> (B, C, T)
    features = Transpose12(allocator_, features);

    std::vector<MNN::Express::VARP> encoder_inputs = {std::move(features),
                                                std::move(features_length)};

    auto encoder_out = encoder_sess_->onForward(encoder_inputs);

    return encoder_out;
  }

  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> RunDecoder(
      MNN::Express::VARP targets, MNN::Express::VARP targets_length,
      std::vector<MNN::Express::VARP> states) {
    std::vector<MNN::Express::VARP> decoder_inputs;
    decoder_inputs.reserve(2 + states.size());

    decoder_inputs.push_back(std::move(targets));
    decoder_inputs.push_back(std::move(targets_length));

    for (auto &s : states) {
      decoder_inputs.push_back(std::move(s));
    }

    auto decoder_out = decoder_sess_->onForward(decoder_inputs);

    std::vector<MNN::Express::VARP> states_next;
    states_next.reserve(states.size());

    // decoder_out[0]: decoder_output
    // decoder_out[1]: decoder_output_length
    // decoder_out[2:] states_next

    for (int32_t i = 0; i != states.size(); ++i) {
      states_next.push_back(std::move(decoder_out[i + 2]));
    }

    // we discard decoder_out[1]
    return {std::move(decoder_out[0]), std::move(states_next)};
  }

  MNN::Express::VARP RunJoiner(MNN::Express::VARP encoder_out, MNN::Express::VARP decoder_out) {
    std::vector<MNN::Express::VARP> joiner_input = {std::move(encoder_out),
                                              std::move(decoder_out)};
    auto logit = joiner_sess_->onForward(
                                   joiner_input);

    return std::move(logit[0]);
  }

  std::vector<MNN::Express::VARP> GetDecoderInitStates(int32_t batch_size) {
    std::array<int, 3> s0_shape{pred_rnn_layers_, batch_size, pred_hidden_};
    MNN::Express::VARP s0 = MNNUtilsCreateTensor<float>(allocator_, s0_shape.data(),
                                                    s0_shape.size());

    Fill<float>(s0, 0);

    std::array<int, 3> s1_shape{pred_rnn_layers_, batch_size, pred_hidden_};

    MNN::Express::VARP s1 = MNNUtilsCreateTensor<float>(allocator_, s1_shape.data(),
                                                    s1_shape.size());

    Fill<float>(s1, 0);

    std::vector<MNN::Express::VARP> states;

    states.reserve(2);
    states.push_back(std::move(s0));
    states.push_back(std::move(s1));

    return states;
  }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }
  int32_t VocabSize() const { return vocab_size_; }

  MNNAllocator *Allocator() { return allocator_; }

  std::string FeatureNormalizationMethod() const { return normalize_type_; }

  bool IsGigaAM() const { return is_giga_am_; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    encoder_sess_ = std::unique_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    // get meta data
    MNNMeta meta_data = encoder_sess_->getInfo()->metaData;
    if (config_.debug) {
      std::ostringstream os;
      os << "---encoder---\n";
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");

    // need to increase by 1 since the blank token is not included in computing
    // vocab_size in NeMo.
    vocab_size_ += 1;

    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA_STR_ALLOW_EMPTY(normalize_type_,
                                               "normalize_type");
    SHERPA_ONNX_READ_META_DATA(pred_rnn_layers_, "pred_rnn_layers");
    SHERPA_ONNX_READ_META_DATA(pred_hidden_, "pred_hidden");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(is_giga_am_, "is_giga_am", 0);

    if (normalize_type_ == "NA") {
      normalize_type_ = "";
    }
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    decoder_sess_ = std::unique_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);
  }

  void InitJoiner(void *model_data, size_t model_data_length) {
    joiner_sess_ = std::unique_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(joiner_sess_.get(), &joiner_input_names_,
                  &joiner_input_names_ptr_);

    GetOutputNames(joiner_sess_.get(), &joiner_output_names_,
                   &joiner_output_names_ptr_);
  }

 private:
  OfflineModelConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> encoder_sess_;
  std::unique_ptr<MNN::Express::Module> decoder_sess_;
  std::unique_ptr<MNN::Express::Module> joiner_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<std::string> joiner_input_names_;
  std::vector<const char *> joiner_input_names_ptr_;

  std::vector<std::string> joiner_output_names_;
  std::vector<const char *> joiner_output_names_ptr_;

  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 8;
  std::string normalize_type_;
  int32_t pred_rnn_layers_ = -1;
  int32_t pred_hidden_ = -1;
  int32_t is_giga_am_ = 0;
};

OfflineTransducerNeMoModel::OfflineTransducerNeMoModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTransducerNeMoModel::OfflineTransducerNeMoModel(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTransducerNeMoModel::~OfflineTransducerNeMoModel() = default;

std::vector<MNN::Express::VARP> OfflineTransducerNeMoModel::RunEncoder(
    MNN::Express::VARP features, MNN::Express::VARP features_length) const {
  return impl_->RunEncoder(std::move(features), std::move(features_length));
}

std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>>
OfflineTransducerNeMoModel::RunDecoder(MNN::Express::VARP targets,
                                       MNN::Express::VARP targets_length,
                                       std::vector<MNN::Express::VARP> states) const {
  return impl_->RunDecoder(std::move(targets), std::move(targets_length),
                           std::move(states));
}

std::vector<MNN::Express::VARP> OfflineTransducerNeMoModel::GetDecoderInitStates(
    int32_t batch_size) const {
  return impl_->GetDecoderInitStates(batch_size);
}

MNN::Express::VARP OfflineTransducerNeMoModel::RunJoiner(MNN::Express::VARP encoder_out,
                                                 MNN::Express::VARP decoder_out) const {
  return impl_->RunJoiner(std::move(encoder_out), std::move(decoder_out));
}

int32_t OfflineTransducerNeMoModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

int32_t OfflineTransducerNeMoModel::VocabSize() const {
  return impl_->VocabSize();
}

MNNAllocator *OfflineTransducerNeMoModel::Allocator() const {
  return impl_->Allocator();
}

std::string OfflineTransducerNeMoModel::FeatureNormalizationMethod() const {
  return impl_->FeatureNormalizationMethod();
}

bool OfflineTransducerNeMoModel::IsGigaAM() const { return impl_->IsGigaAM(); }

#if __ANDROID_API__ >= 9
template OfflineTransducerNeMoModel::OfflineTransducerNeMoModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineTransducerNeMoModel::OfflineTransducerNeMoModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_mnn
