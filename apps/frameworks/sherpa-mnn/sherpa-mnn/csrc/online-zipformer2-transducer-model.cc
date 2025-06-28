// sherpa-mnn/csrc/online-zipformer2-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-zipformer2-transducer-model.h"

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
#include "sherpa-mnn/csrc/unbind.h"

namespace sherpa_mnn {

OnlineZipformer2TransducerModel::OnlineZipformer2TransducerModel(
    const OnlineModelConfig &config)
    : 
      sess_opts_(GetSessionOptions(config)),
      config_(config),
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
OnlineZipformer2TransducerModel::OnlineZipformer2TransducerModel(
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

void OnlineZipformer2TransducerModel::InitEncoder(void *model_data,
                                                  size_t model_data_length) {
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
  SHERPA_ONNX_READ_META_DATA_VEC(encoder_dims_, "encoder_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(query_head_dims_, "query_head_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(value_head_dims_, "value_head_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(num_heads_, "num_heads");
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
    print(query_head_dims_, "query_head_dims");
    print(value_head_dims_, "value_head_dims");
    print(num_heads_, "num_heads");
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

void OnlineZipformer2TransducerModel::InitDecoder(void *model_data,
                                                  size_t model_data_length) {
  decoder_sess_ = std::unique_ptr<MNN::Express::Module>(
    MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

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
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }

  MNNAllocator* allocator;  // used in the macro below
  SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
  SHERPA_ONNX_READ_META_DATA(context_size_, "context_size");
}

void OnlineZipformer2TransducerModel::InitJoiner(void *model_data,
                                                 size_t model_data_length) {
  joiner_sess_ = std::unique_ptr<MNN::Express::Module>(
    MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

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
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }
}

std::vector<MNN::Express::VARP> OnlineZipformer2TransducerModel::StackStates(
    const std::vector<std::vector<MNN::Express::VARP>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());

  std::vector<MNN::Express::VARP > buf(batch_size);

  auto allocator =
      const_cast<OnlineZipformer2TransducerModel *>(this)->allocator_;

  std::vector<MNN::Express::VARP> ans;
  int32_t num_states = static_cast<int32_t>(states[0].size());
  ans.reserve(num_states);

  for (int32_t i = 0; i != (num_states - 2) / 6; ++i) {
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = states[n][6 * i];
      }
      auto v = Cat(allocator, buf, 1);
      ans.push_back(std::move(v));
    }
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = states[n][6 * i + 1];
      }
      auto v = Cat(allocator, buf, 1);
      ans.push_back(std::move(v));
    }
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = states[n][6 * i + 2];
      }
      auto v = Cat(allocator, buf, 1);
      ans.push_back(std::move(v));
    }
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = states[n][6 * i + 3];
      }
      auto v = Cat(allocator, buf, 1);
      ans.push_back(std::move(v));
    }
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = states[n][6 * i + 4];
      }
      auto v = Cat(allocator, buf, 0);
      ans.push_back(std::move(v));
    }
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = states[n][6 * i + 5];
      }
      auto v = Cat(allocator, buf, 0);
      ans.push_back(std::move(v));
    }
  }

  {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = states[n][num_states - 2];
    }
    auto v = Cat(allocator, buf, 0);
    ans.push_back(std::move(v));
  }

  {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = states[n][num_states - 1];
    }
    auto v = Cat<int>(allocator, buf, 0);
    ans.push_back(std::move(v));
  }
  return ans;
}

std::vector<std::vector<MNN::Express::VARP>>
OnlineZipformer2TransducerModel::UnStackStates(
    const std::vector<MNN::Express::VARP> &states) const {
  int32_t m = std::accumulate(num_encoder_layers_.begin(),
                              num_encoder_layers_.end(), 0);
  assert(static_cast<int32_t>(states.size()) == m * 6 + 2);

  int32_t batch_size = states[0]->getInfo()->dim[1];

  auto allocator =
      const_cast<OnlineZipformer2TransducerModel *>(this)->allocator_;

  std::vector<std::vector<MNN::Express::VARP>> ans;
  ans.resize(batch_size);

  for (int32_t i = 0; i != m; ++i) {
    {
      auto v = Unbind(allocator, states[i * 6], 1);
      assert(static_cast<int32_t>(v.size()) == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind(allocator, states[i * 6 + 1], 1);
      assert(static_cast<int32_t>(v.size()) == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind(allocator, states[i * 6 + 2], 1);
      assert(static_cast<int32_t>(v.size()) == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind(allocator, states[i * 6 + 3], 1);
      assert(static_cast<int32_t>(v.size()) == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind(allocator, states[i * 6 + 4], 0);
      assert(static_cast<int32_t>(v.size()) == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind(allocator, states[i * 6 + 5], 0);
      assert(static_cast<int32_t>(v.size()) == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
  }

  {
    auto v = Unbind(allocator, states[m * 6], 0);
    assert(static_cast<int32_t>(v.size()) == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }
  {
    auto v = Unbind<int>(allocator, states[m * 6 + 1], 0);
    assert(static_cast<int32_t>(v.size()) == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  return ans;
}

std::vector<MNN::Express::VARP>
OnlineZipformer2TransducerModel::GetEncoderInitStates() {
  std::vector<MNN::Express::VARP> ans;
  int32_t n = static_cast<int32_t>(encoder_dims_.size());
  int32_t m = std::accumulate(num_encoder_layers_.begin(),
                              num_encoder_layers_.end(), 0);
  ans.reserve(m * 6 + 2);

  for (int32_t i = 0; i != n; ++i) {
    int32_t num_layers = num_encoder_layers_[i];
    int32_t key_dim = query_head_dims_[i] * num_heads_[i];
    int32_t value_dim = value_head_dims_[i] * num_heads_[i];
    int32_t nonlin_attn_head_dim = 3 * encoder_dims_[i] / 4;

    for (int32_t j = 0; j != num_layers; ++j) {
      {
        std::array<int, 3> s{left_context_len_[i], 1, key_dim};
        auto v =
            MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
        Fill(v, 0);
        ans.push_back(std::move(v));
      }

      {
        std::array<int, 4> s{1, 1, left_context_len_[i],
                                 nonlin_attn_head_dim};
        auto v =
            MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
        Fill(v, 0);
        ans.push_back(std::move(v));
      }

      {
        std::array<int, 3> s{left_context_len_[i], 1, value_dim};
        auto v =
            MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
        Fill(v, 0);
        ans.push_back(std::move(v));
      }

      {
        std::array<int, 3> s{left_context_len_[i], 1, value_dim};
        auto v =
            MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
        Fill(v, 0);
        ans.push_back(std::move(v));
      }

      {
        std::array<int, 3> s{1, encoder_dims_[i],
                                 cnn_module_kernels_[i] / 2};
        auto v =
            MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
        Fill(v, 0);
        ans.push_back(std::move(v));
      }

      {
        std::array<int, 3> s{1, encoder_dims_[i],
                                 cnn_module_kernels_[i] / 2};
        auto v =
            MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
        Fill(v, 0);
        ans.push_back(std::move(v));
      }
    }
  }

  {
    SHERPA_ONNX_CHECK_NE(feature_dim_, 0);
    int32_t embed_dim = (((feature_dim_ - 1) / 2) - 1) / 2;
    std::array<int, 4> s{1, 128, 3, embed_dim};

    auto v = MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
    Fill(v, 0);
    ans.push_back(std::move(v));
  }

  {
    std::array<int, 1> s{1};
    auto v = MNNUtilsCreateTensor<int>(allocator_, s.data(), s.size());
    Fill<int>(v, 0);
    ans.push_back(std::move(v));
  }
  return ans;
}

std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>>
OnlineZipformer2TransducerModel::RunEncoder(MNN::Express::VARP features,
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

MNN::Express::VARP OnlineZipformer2TransducerModel::RunDecoder(
    MNN::Express::VARP decoder_input) {
  auto decoder_out = decoder_sess_->onForward({decoder_input});
  return std::move(decoder_out[0]);
}

MNN::Express::VARP OnlineZipformer2TransducerModel::RunJoiner(MNN::Express::VARP encoder_out,
                                                      MNN::Express::VARP decoder_out) {
  std::vector<MNN::Express::VARP> joiner_input = {std::move(encoder_out),
                                            std::move(decoder_out)};
  auto logit =
      joiner_sess_->onForward(joiner_input);

  return std::move(logit[0]);
}

#if __ANDROID_API__ >= 9
template OnlineZipformer2TransducerModel::OnlineZipformer2TransducerModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineZipformer2TransducerModel::OnlineZipformer2TransducerModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_mnn
