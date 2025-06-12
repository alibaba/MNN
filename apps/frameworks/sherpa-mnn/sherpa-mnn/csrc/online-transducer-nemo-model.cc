// sherpa-mnn/csrc/online-transducer-nemo-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#include "sherpa-mnn/csrc/online-transducer-nemo-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <numeric>
#include <sstream>
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

#include "sherpa-mnn/csrc/cat.h"
#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/online-transducer-decoder.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/session.h"
#include "sherpa-mnn/csrc/text-utils.h"
#include "sherpa-mnn/csrc/transpose.h"
#include "sherpa-mnn/csrc/unbind.h"

namespace sherpa_mnn {

class OnlineTransducerNeMoModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.joiner);
      InitJoiner(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.joiner);
      InitJoiner(buf.data(), buf.size());
    }
  }

  std::vector<MNN::Express::VARP> RunEncoder(MNN::Express::VARP features,
                                     std::vector<MNN::Express::VARP> states) {
    MNN::Express::VARP &cache_last_channel = states[0];
    MNN::Express::VARP &cache_last_time = states[1];
    MNN::Express::VARP &cache_last_channel_len = states[2];

    int32_t batch_size = features->getInfo()->dim[0];

    std::array<int, 1> length_shape{batch_size};

    MNN::Express::VARP length = MNNUtilsCreateTensor<int>(
        allocator_, length_shape.data(), length_shape.size());

    int *p_length = length->writeMap<int>();

    std::fill(p_length, p_length + batch_size, ChunkSize());

    // (B, T, C) -> (B, C, T)
    features = Transpose12(allocator_, features);

    std::vector<MNN::Express::VARP> inputs = {
        std::move(features), View(length), std::move(cache_last_channel),
        std::move(cache_last_time), std::move(cache_last_channel_len)};

    auto out = encoder_sess_->onForward(inputs);
    // out[0]: logit
    // out[1] logit_length
    // out[2:] states_next
    //
    // we need to remove out[1]

    std::vector<MNN::Express::VARP> ans;
    ans.reserve(out.size() - 1);

    for (int32_t i = 0; i != out.size(); ++i) {
      if (i == 1) {
        continue;
      }

      ans.push_back(std::move(out[i]));
    }

    return ans;
  }

  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> RunDecoder(
      MNN::Express::VARP targets, std::vector<MNN::Express::VARP> states) {
    MNNAllocator* memory_info = nullptr;

    auto shape = targets->getInfo()->dim;
    int32_t batch_size = static_cast<int32_t>(shape[0]);

    std::vector<int> length_shape = {batch_size};
    std::vector<int> length_value(batch_size, 1);

    MNN::Express::VARP targets_length = MNNUtilsCreateTensor<int>(
        memory_info, length_value.data(), batch_size, length_shape.data(),
        length_shape.size());

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
    // decoder_out[1]: decoder_output_length (discarded)
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

  std::vector<MNN::Express::VARP> GetDecoderInitStates() {
    std::vector<MNN::Express::VARP> ans;
    ans.reserve(2);
    ans.push_back(View(lstm0_));
    ans.push_back(View(lstm1_));

    return ans;
  }

  int32_t ChunkSize() const { return window_size_; }

  int32_t ChunkShift() const { return chunk_shift_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }

  int32_t VocabSize() const { return vocab_size_; }

  MNNAllocator *Allocator() { return allocator_; }

  std::string FeatureNormalizationMethod() const { return normalize_type_; }

  // Return a vector containing 3 tensors
  // - cache_last_channel
  // - cache_last_time_
  // - cache_last_channel_len
  std::vector<MNN::Express::VARP> GetEncoderInitStates() {
    std::vector<MNN::Express::VARP> ans;
    ans.reserve(3);
    ans.push_back(View(cache_last_channel_));
    ans.push_back(View(cache_last_time_));
    ans.push_back(View(cache_last_channel_len_));

    return ans;
  }

  std::vector<MNN::Express::VARP> StackStates(
      std::vector<std::vector<MNN::Express::VARP>> states) const {
    int32_t batch_size = static_cast<int32_t>(states.size());
    if (batch_size == 1) {
      return std::move(states[0]);
    }

    std::vector<MNN::Express::VARP> ans;

    auto allocator = const_cast<Impl *>(this)->allocator_;

    // stack cache_last_channel
    std::vector<MNN::Express::VARP > buf(batch_size);

    // there are 3 states to be stacked
    for (int32_t i = 0; i != 3; ++i) {
      buf.clear();
      buf.reserve(batch_size);

      for (int32_t b = 0; b != batch_size; ++b) {
        assert(states[b].size() == 3);
        buf.push_back(states[b][i]);
      }

      MNN::Express::VARP c{nullptr};
      if (i == 2) {
        c = Cat<int>(allocator, buf, 0);
      } else {
        c = Cat(allocator, buf, 0);
      }

      ans.push_back(std::move(c));
    }

    return ans;
  }

  std::vector<std::vector<MNN::Express::VARP>> UnStackStates(
      std::vector<MNN::Express::VARP> states) {
    assert(states.size() == 3);

    std::vector<std::vector<MNN::Express::VARP>> ans;

    auto shape = states[0]->getInfo()->dim;
    int32_t batch_size = shape[0];
    ans.resize(batch_size);

    if (batch_size == 1) {
      ans[0] = std::move(states);
      return ans;
    }

    for (int32_t i = 0; i != 3; ++i) {
      std::vector<MNN::Express::VARP> v;
      if (i == 2) {
        v = Unbind<int>(allocator_, states[i], 0);
      } else {
        v = Unbind(allocator_, states[i], 0);
      }

      assert(v.size() == batch_size);

      for (int32_t b = 0; b != batch_size; ++b) {
        ans[b].push_back(std::move(v[b]));
      }
    }

    return ans;
  }

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
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");

    // need to increase by 1 since the blank token is not included in computing
    // vocab_size in NeMo.
    vocab_size_ += 1;

    SHERPA_ONNX_READ_META_DATA(window_size_, "window_size");
    SHERPA_ONNX_READ_META_DATA(chunk_shift_, "chunk_shift");
    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA_STR(normalize_type_, "normalize_type");
    SHERPA_ONNX_READ_META_DATA(pred_rnn_layers_, "pred_rnn_layers");
    SHERPA_ONNX_READ_META_DATA(pred_hidden_, "pred_hidden");

    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim1_,
                               "cache_last_channel_dim1");
    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim2_,
                               "cache_last_channel_dim2");
    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim3_,
                               "cache_last_channel_dim3");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim1_, "cache_last_time_dim1");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim2_, "cache_last_time_dim2");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim3_, "cache_last_time_dim3");

    if (normalize_type_ == "NA") {
      normalize_type_ = "";
    }

    InitEncoderStates();
  }

  void InitEncoderStates() {
    std::array<int, 4> cache_last_channel_shape{1, cache_last_channel_dim1_,
                                                    cache_last_channel_dim2_,
                                                    cache_last_channel_dim3_};

    cache_last_channel_ = MNNUtilsCreateTensor<float>(
        allocator_, cache_last_channel_shape.data(),
        cache_last_channel_shape.size());

    Fill<float>(cache_last_channel_, 0);

    std::array<int, 4> cache_last_time_shape{
        1, cache_last_time_dim1_, cache_last_time_dim2_, cache_last_time_dim3_};

    cache_last_time_ = MNNUtilsCreateTensor<float>(
        allocator_, cache_last_time_shape.data(), cache_last_time_shape.size());

    Fill<float>(cache_last_time_, 0);

    int shape = 1;
    cache_last_channel_len_ =
        MNNUtilsCreateTensor<int>(allocator_, &shape, 1);

    cache_last_channel_len_->writeMap<int>()[0] = 0;
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    decoder_sess_ = std::unique_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);

    InitDecoderStates();
  }

  void InitDecoderStates() {
    int32_t batch_size = 1;
    std::array<int, 3> s0_shape{pred_rnn_layers_, batch_size, pred_hidden_};
    lstm0_ = MNNUtilsCreateTensor<float>(allocator_, s0_shape.data(),
                                             s0_shape.size());

    Fill<float>(lstm0_, 0);

    std::array<int, 3> s1_shape{pred_rnn_layers_, batch_size, pred_hidden_};

    lstm1_ = MNNUtilsCreateTensor<float>(allocator_, s1_shape.data(),
                                             s1_shape.size());

    Fill<float>(lstm1_, 0);
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
  OnlineModelConfig config_;
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

  int32_t window_size_ = 0;
  int32_t chunk_shift_ = 0;
  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 8;
  std::string normalize_type_;
  int32_t pred_rnn_layers_ = -1;
  int32_t pred_hidden_ = -1;

  // encoder states
  int32_t cache_last_channel_dim1_ = 0;
  int32_t cache_last_channel_dim2_ = 0;
  int32_t cache_last_channel_dim3_ = 0;
  int32_t cache_last_time_dim1_ = 0;
  int32_t cache_last_time_dim2_ = 0;
  int32_t cache_last_time_dim3_ = 0;

  // init encoder states
  MNN::Express::VARP cache_last_channel_{nullptr};
  MNN::Express::VARP cache_last_time_{nullptr};
  MNN::Express::VARP cache_last_channel_len_{nullptr};

  // init decoder states
  MNN::Express::VARP lstm0_{nullptr};
  MNN::Express::VARP lstm1_{nullptr};
};

OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OnlineTransducerNeMoModel::~OnlineTransducerNeMoModel() = default;

std::vector<MNN::Express::VARP> OnlineTransducerNeMoModel::RunEncoder(
    MNN::Express::VARP features, std::vector<MNN::Express::VARP> states) const {
  return impl_->RunEncoder(std::move(features), std::move(states));
}

std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>>
OnlineTransducerNeMoModel::RunDecoder(MNN::Express::VARP targets,
                                      std::vector<MNN::Express::VARP> states) const {
  return impl_->RunDecoder(std::move(targets), std::move(states));
}

std::vector<MNN::Express::VARP> OnlineTransducerNeMoModel::GetDecoderInitStates()
    const {
  return impl_->GetDecoderInitStates();
}

MNN::Express::VARP OnlineTransducerNeMoModel::RunJoiner(MNN::Express::VARP encoder_out,
                                                MNN::Express::VARP decoder_out) const {
  return impl_->RunJoiner(std::move(encoder_out), std::move(decoder_out));
}

int32_t OnlineTransducerNeMoModel::ChunkSize() const {
  return impl_->ChunkSize();
}

int32_t OnlineTransducerNeMoModel::ChunkShift() const {
  return impl_->ChunkShift();
}

int32_t OnlineTransducerNeMoModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

int32_t OnlineTransducerNeMoModel::VocabSize() const {
  return impl_->VocabSize();
}

MNNAllocator *OnlineTransducerNeMoModel::Allocator() const {
  return impl_->Allocator();
}

std::string OnlineTransducerNeMoModel::FeatureNormalizationMethod() const {
  return impl_->FeatureNormalizationMethod();
}

std::vector<MNN::Express::VARP> OnlineTransducerNeMoModel::GetEncoderInitStates()
    const {
  return impl_->GetEncoderInitStates();
}

std::vector<MNN::Express::VARP> OnlineTransducerNeMoModel::StackStates(
    std::vector<std::vector<MNN::Express::VARP>> states) const {
  return impl_->StackStates(std::move(states));
}

std::vector<std::vector<MNN::Express::VARP>> OnlineTransducerNeMoModel::UnStackStates(
    std::vector<MNN::Express::VARP> states) const {
  return impl_->UnStackStates(std::move(states));
}

#if __ANDROID_API__ >= 9
template OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_mnn
