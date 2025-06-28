// sherpa-mnn/csrc/online-zipformer-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-zipformer-transducer-model.h"

#include <algorithm>
#include <cassert>
#include <memory>
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
#include "sherpa-mnn/csrc/unbind.h"

namespace sherpa_mnn {

OnlineZipformerTransducerModel::OnlineZipformerTransducerModel(
    const OnlineModelConfig &config)
    : 
      config_(config),
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
OnlineZipformerTransducerModel::OnlineZipformerTransducerModel(
    Manager *mgr, const OnlineModelConfig &config)
    : 
      config_(config),
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

void OnlineZipformerTransducerModel::InitEncoder(void *model_data,
                                                 size_t model_data_length) {
  encoder_sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data,
                                                 model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

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
  SHERPA_ONNX_READ_META_DATA_VEC(encoder_dims_, "encoder_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(attention_dims_, "attention_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(num_encoder_layers_, "num_encoder_layers");
  SHERPA_ONNX_READ_META_DATA_VEC(cnn_module_kernels_, "cnn_module_kernels");
  SHERPA_ONNX_READ_META_DATA_VEC(left_context_len_, "left_context_len");

  SHERPA_ONNX_READ_META_DATA(T_, "T");
  SHERPA_ONNX_READ_META_DATA(decode_chunk_len_, "decode_chunk_len");

  if (config_.debug) {
    auto print = [](const std::vector<int32_t> &v, const char *name) {
      std::ostringstream os;
      os << name << ": ";
      for (auto i : v) {
        os << i << " ";
      }
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    };
    print(encoder_dims_, "encoder_dims");
    print(attention_dims_, "attention_dims");
    print(num_encoder_layers_, "num_encoder_layers");
    print(cnn_module_kernels_, "cnn_module_kernels");
    print(left_context_len_, "left_context_len");
#if __OHOS__
    SHERPA_ONNX_LOGE("T: %{public}d", T_);
    SHERPA_ONNX_LOGE("decode_chunk_len_: %{public}d", decode_chunk_len_);
#else
    SHERPA_ONNX_LOGE("T: %d", T_);
    SHERPA_ONNX_LOGE("decode_chunk_len_: %d", decode_chunk_len_);
#endif
  }
}

void OnlineZipformerTransducerModel::InitDecoder(void *model_data,
                                                 size_t model_data_length) {
  decoder_sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data,
                                                 model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

  GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                &decoder_input_names_ptr_);

  GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                 &decoder_output_names_ptr_);

  // get meta data
  MNNMeta meta_data = decoder_sess_->getInfo()->metaData;
  if (config_.debug) {
    std::ostringstream os;
    os << "---decoder---\n";
    PrintModelMetadata(os, meta_data);
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
  }

  MNNAllocator* allocator;  // used in the macro below
  SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
  SHERPA_ONNX_READ_META_DATA(context_size_, "context_size");
}

void OnlineZipformerTransducerModel::InitJoiner(void *model_data,
                                                size_t model_data_length) {
  joiner_sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data,
                                                model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

  GetInputNames(joiner_sess_.get(), &joiner_input_names_,
                &joiner_input_names_ptr_);

  GetOutputNames(joiner_sess_.get(), &joiner_output_names_,
                 &joiner_output_names_ptr_);

  // get meta data
  MNNMeta meta_data = joiner_sess_->getInfo()->metaData;
  if (config_.debug) {
    std::ostringstream os;
    os << "---joiner---\n";
    PrintModelMetadata(os, meta_data);
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
  }
}

std::vector<MNN::Express::VARP> OnlineZipformerTransducerModel::StackStates(
    const std::vector<std::vector<MNN::Express::VARP>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());
  int32_t num_encoders = static_cast<int32_t>(num_encoder_layers_.size());

  std::vector<MNN::Express::VARP > buf(batch_size);

  std::vector<MNN::Express::VARP> ans;
  ans.reserve(states[0].size());

  auto allocator =
      const_cast<OnlineZipformerTransducerModel *>(this)->allocator_;

  // cached_len
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = states[n][i];
    }
    auto v = Cat<int>(allocator, buf, 1);  // (num_layers, 1)
    ans.push_back(std::move(v));
  }

  // cached_avg
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = states[n][num_encoders + i];
    }
    auto v = Cat(allocator, buf, 1);  // (num_layers, 1, encoder_dims)
    ans.push_back(std::move(v));
  }

  // cached_key
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = states[n][num_encoders * 2 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims)
    auto v = Cat(allocator, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_val
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = states[n][num_encoders * 3 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims/2)
    auto v = Cat(allocator, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_val2
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = states[n][num_encoders * 4 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims/2)
    auto v = Cat(allocator, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_conv1
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = states[n][num_encoders * 5 + i];
    }
    // (num_layers, 1, encoder_dims, cnn_module_kernels-1)
    auto v = Cat(allocator, buf, 1);
    ans.push_back(std::move(v));
  }

  // cached_conv2
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = states[n][num_encoders * 6 + i];
    }
    // (num_layers, 1, encoder_dims, cnn_module_kernels-1)
    auto v = Cat(allocator, buf, 1);
    ans.push_back(std::move(v));
  }

  return ans;
}

std::vector<std::vector<MNN::Express::VARP>>
OnlineZipformerTransducerModel::UnStackStates(
    const std::vector<MNN::Express::VARP> &states) const {
  assert(states.size() == num_encoder_layers_.size() * 7);

  int32_t batch_size = states[0]->getInfo()->dim[1];
  int32_t num_encoders = num_encoder_layers_.size();

  auto allocator =
      const_cast<OnlineZipformerTransducerModel *>(this)->allocator_;

  std::vector<std::vector<MNN::Express::VARP>> ans;
  ans.resize(batch_size);

  // cached_len
  for (int32_t i = 0; i != num_encoders; ++i) {
    auto v = Unbind<int>(allocator, states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_avg
  for (int32_t i = num_encoders; i != 2 * num_encoders; ++i) {
    auto v = Unbind(allocator, states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_key
  for (int32_t i = 2 * num_encoders; i != 3 * num_encoders; ++i) {
    auto v = Unbind(allocator, states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_val
  for (int32_t i = 3 * num_encoders; i != 4 * num_encoders; ++i) {
    auto v = Unbind(allocator, states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_val2
  for (int32_t i = 4 * num_encoders; i != 5 * num_encoders; ++i) {
    auto v = Unbind(allocator, states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_conv1
  for (int32_t i = 5 * num_encoders; i != 6 * num_encoders; ++i) {
    auto v = Unbind(allocator, states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_conv2
  for (int32_t i = 6 * num_encoders; i != 7 * num_encoders; ++i) {
    auto v = Unbind(allocator, states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  return ans;
}

std::vector<MNN::Express::VARP> OnlineZipformerTransducerModel::GetEncoderInitStates() {
  // Please see
  // https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py#L673
  // for details

  int32_t n = static_cast<int32_t>(encoder_dims_.size());
  std::vector<MNN::Express::VARP> cached_len_vec;
  std::vector<MNN::Express::VARP> cached_avg_vec;
  std::vector<MNN::Express::VARP> cached_key_vec;
  std::vector<MNN::Express::VARP> cached_val_vec;
  std::vector<MNN::Express::VARP> cached_val2_vec;
  std::vector<MNN::Express::VARP> cached_conv1_vec;
  std::vector<MNN::Express::VARP> cached_conv2_vec;

  cached_len_vec.reserve(n);
  cached_avg_vec.reserve(n);
  cached_key_vec.reserve(n);
  cached_val_vec.reserve(n);
  cached_val2_vec.reserve(n);
  cached_conv1_vec.reserve(n);
  cached_conv2_vec.reserve(n);

  for (int32_t i = 0; i != n; ++i) {
    {
      std::array<int, 2> s{num_encoder_layers_[i], 1};
      auto v =
          MNNUtilsCreateTensor<int>(allocator_, s.data(), s.size());
      Fill<int>(v, 0);
      cached_len_vec.push_back(std::move(v));
    }

    {
      std::array<int, 3> s{num_encoder_layers_[i], 1, encoder_dims_[i]};
      auto v = MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
      Fill(v, 0);
      cached_avg_vec.push_back(std::move(v));
    }

    {
      std::array<int, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i]};
      auto v = MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
      Fill(v, 0);
      cached_key_vec.push_back(std::move(v));
    }

    {
      std::array<int, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i] / 2};
      auto v = MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
      Fill(v, 0);
      cached_val_vec.push_back(std::move(v));
    }

    {
      std::array<int, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i] / 2};
      auto v = MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
      Fill(v, 0);
      cached_val2_vec.push_back(std::move(v));
    }

    {
      std::array<int, 4> s{num_encoder_layers_[i], 1, encoder_dims_[i],
                               cnn_module_kernels_[i] - 1};
      auto v = MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
      Fill(v, 0);
      cached_conv1_vec.push_back(std::move(v));
    }

    {
      std::array<int, 4> s{num_encoder_layers_[i], 1, encoder_dims_[i],
                               cnn_module_kernels_[i] - 1};
      auto v = MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
      Fill(v, 0);
      cached_conv2_vec.push_back(std::move(v));
    }
  }

  std::vector<MNN::Express::VARP> ans;
  ans.reserve(n * 7);

  for (auto &v : cached_len_vec) ans.push_back(std::move(v));
  for (auto &v : cached_avg_vec) ans.push_back(std::move(v));
  for (auto &v : cached_key_vec) ans.push_back(std::move(v));
  for (auto &v : cached_val_vec) ans.push_back(std::move(v));
  for (auto &v : cached_val2_vec) ans.push_back(std::move(v));
  for (auto &v : cached_conv1_vec) ans.push_back(std::move(v));
  for (auto &v : cached_conv2_vec) ans.push_back(std::move(v));

  return ans;
}

std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>>
OnlineZipformerTransducerModel::RunEncoder(MNN::Express::VARP features,
                                           std::vector<MNN::Express::VARP> states,
                                           MNN::Express::VARP /* processed_frames */) {
  std::vector<MNN::Express::VARP> encoder_inputs;
  encoder_inputs.reserve(1 + states.size());

  encoder_inputs.push_back(std::move(features));
  for (auto &v : states) {
    encoder_inputs.push_back(std::move(v));
  }

  auto encoder_out = encoder_sess_->onForward(encoder_inputs);

  std::vector<MNN::Express::VARP> next_states;
  next_states.reserve(states.size());

  for (int32_t i = 1; i != static_cast<int32_t>(encoder_out.size()); ++i) {
    next_states.push_back(std::move(encoder_out[i]));
  }

  return {std::move(encoder_out[0]), std::move(next_states)};
}

MNN::Express::VARP OnlineZipformerTransducerModel::RunDecoder(
    MNN::Express::VARP decoder_input) {
  auto decoder_out = decoder_sess_->onForward({decoder_input});
  return std::move(decoder_out[0]);
}

MNN::Express::VARP OnlineZipformerTransducerModel::RunJoiner(MNN::Express::VARP encoder_out,
                                                     MNN::Express::VARP decoder_out) {
  std::vector<MNN::Express::VARP> joiner_input = {std::move(encoder_out),
                                            std::move(decoder_out)};
  auto logit =
      joiner_sess_->onForward(joiner_input);

  return std::move(logit[0]);
}

#if __ANDROID_API__ >= 9
template OnlineZipformerTransducerModel::OnlineZipformerTransducerModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineZipformerTransducerModel::OnlineZipformerTransducerModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_mnn
