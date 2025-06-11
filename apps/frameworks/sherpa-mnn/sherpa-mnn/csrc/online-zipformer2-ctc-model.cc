// sherpa-mnn/csrc/online-zipformer2-ctc-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-zipformer2-ctc-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <string>

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
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/session.h"
#include "sherpa-mnn/csrc/text-utils.h"
#include "sherpa-mnn/csrc/unbind.h"

namespace sherpa_mnn {

class OnlineZipformer2CtcModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.zipformer2_ctc.model);
      Init(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.zipformer2_ctc.model);
      Init(buf.data(), buf.size());
    }
  }

  std::vector<MNN::Express::VARP> Forward(MNN::Express::VARP features,
                                  std::vector<MNN::Express::VARP> states) {
    std::vector<MNN::Express::VARP> inputs;
    inputs.reserve(1 + states.size());

    inputs.push_back(std::move(features));
    for (auto &v : states) {
      inputs.push_back(std::move(v));
    }

    return sess_->onForward(inputs);
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t ChunkLength() const { return T_; }

  int32_t ChunkShift() const { return decode_chunk_len_; }

  MNNAllocator *Allocator() { return allocator_; }

  // Return a vector containing 3 tensors
  // - attn_cache
  // - conv_cache
  // - offset
  std::vector<MNN::Express::VARP> GetInitStates() {
    std::vector<MNN::Express::VARP> ans;
    ans.reserve(initial_states_.size());
    for (auto &s : initial_states_) {
      ans.push_back(View(s));
    }
    return ans;
  }

  std::vector<MNN::Express::VARP> StackStates(
      std::vector<std::vector<MNN::Express::VARP>> states) {
    int32_t batch_size = static_cast<int32_t>(states.size());

    std::vector<MNN::Express::VARP > buf(batch_size);

    std::vector<MNN::Express::VARP> ans;
    int32_t num_states = static_cast<int32_t>(states[0].size());
    ans.reserve(num_states);

    for (int32_t i = 0; i != (num_states - 2) / 6; ++i) {
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = states[n][6 * i];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = states[n][6 * i + 1];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = states[n][6 * i + 2];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = states[n][6 * i + 3];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = states[n][6 * i + 4];
        }
        auto v = Cat(allocator_, buf, 0);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = states[n][6 * i + 5];
        }
        auto v = Cat(allocator_, buf, 0);
        ans.push_back(std::move(v));
      }
    }

    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = states[n][num_states - 2];
      }
      auto v = Cat(allocator_, buf, 0);
      ans.push_back(std::move(v));
    }

    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = states[n][num_states - 1];
      }
      auto v = Cat<int>(allocator_, buf, 0);
      ans.push_back(std::move(v));
    }
    return ans;
  }

  std::vector<std::vector<MNN::Express::VARP>> UnStackStates(
      std::vector<MNN::Express::VARP> states) {
    int32_t m = std::accumulate(num_encoder_layers_.begin(),
                                num_encoder_layers_.end(), 0);
    assert(states.size() == m * 6 + 2);

    int32_t batch_size = states[0]->getInfo()->dim[1];

    std::vector<std::vector<MNN::Express::VARP>> ans;
    ans.resize(batch_size);

    for (int32_t i = 0; i != m; ++i) {
      {
        auto v = Unbind(allocator_, states[i * 6], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, states[i * 6 + 1], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, states[i * 6 + 2], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, states[i * 6 + 3], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, states[i * 6 + 4], 0);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, states[i * 6 + 5], 0);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
    }

    {
      auto v = Unbind(allocator_, states[m * 6], 0);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind<int>(allocator_, states[m * 6 + 1], 0);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }

    return ans;
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
      os << "---zipformer2_ctc---\n";
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

    if (meta_data.find("vocab_size") != meta_data.end()) {
      vocab_size_ = std::stoi(meta_data["vocab_size"]);
    }

    if (config_.debug) {
      auto print = [](const std::vector<int32_t> &v, const char *name) {
        std::ostringstream os;
        os << name << ": ";
        for (auto i : v) {
          os << i << " ";
        }
        SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
      };
      print(encoder_dims_, "encoder_dims");
      print(query_head_dims_, "query_head_dims");
      print(value_head_dims_, "value_head_dims");
      print(num_heads_, "num_heads");
      print(num_encoder_layers_, "num_encoder_layers");
      print(cnn_module_kernels_, "cnn_module_kernels");
      print(left_context_len_, "left_context_len");
      SHERPA_ONNX_LOGE("T: %d", T_);
      SHERPA_ONNX_LOGE("decode_chunk_len_: %d", decode_chunk_len_);
      SHERPA_ONNX_LOGE("vocab_size_: %d", vocab_size_);
    }

    InitStates();
  }

  void InitStates() {
    int32_t n = static_cast<int32_t>(encoder_dims_.size());
    int32_t m = std::accumulate(num_encoder_layers_.begin(),
                                num_encoder_layers_.end(), 0);
    initial_states_.reserve(m * 6 + 2);

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
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int, 4> s{1, 1, left_context_len_[i],
                                   nonlin_attn_head_dim};
          auto v =
              MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
          Fill(v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int, 3> s{left_context_len_[i], 1, value_dim};
          auto v =
              MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
          Fill(v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int, 3> s{left_context_len_[i], 1, value_dim};
          auto v =
              MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
          Fill(v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int, 3> s{1, encoder_dims_[i],
                                   cnn_module_kernels_[i] / 2};
          auto v =
              MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
          Fill(v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int, 3> s{1, encoder_dims_[i],
                                   cnn_module_kernels_[i] / 2};
          auto v =
              MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
          Fill(v, 0);
          initial_states_.push_back(std::move(v));
        }
      }
    }

    {
      std::array<int, 4> s{1, 128, 3, 19};
      auto v = MNNUtilsCreateTensor<float>(allocator_, s.data(), s.size());
      Fill(v, 0);
      initial_states_.push_back(std::move(v));
    }

    {
      std::array<int, 1> s{1};
      auto v =
          MNNUtilsCreateTensor<int>(allocator_, s.data(), s.size());
      Fill<int>(v, 0);
      initial_states_.push_back(std::move(v));
    }
  }

 private:
  OnlineModelConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  std::vector<MNN::Express::VARP> initial_states_;

  std::vector<int32_t> encoder_dims_;
  std::vector<int32_t> query_head_dims_;
  std::vector<int32_t> value_head_dims_;
  std::vector<int32_t> num_heads_;
  std::vector<int32_t> num_encoder_layers_;
  std::vector<int32_t> cnn_module_kernels_;
  std::vector<int32_t> left_context_len_;

  int32_t T_ = 0;
  int32_t decode_chunk_len_ = 0;
  int32_t vocab_size_ = 0;
};

OnlineZipformer2CtcModel::OnlineZipformer2CtcModel(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineZipformer2CtcModel::OnlineZipformer2CtcModel(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OnlineZipformer2CtcModel::~OnlineZipformer2CtcModel() = default;

std::vector<MNN::Express::VARP> OnlineZipformer2CtcModel::Forward(
    MNN::Express::VARP x, std::vector<MNN::Express::VARP> states) const {
  return impl_->Forward(std::move(x), std::move(states));
}

int32_t OnlineZipformer2CtcModel::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OnlineZipformer2CtcModel::ChunkLength() const {
  return impl_->ChunkLength();
}

int32_t OnlineZipformer2CtcModel::ChunkShift() const {
  return impl_->ChunkShift();
}

MNNAllocator *OnlineZipformer2CtcModel::Allocator() const {
  return impl_->Allocator();
}

std::vector<MNN::Express::VARP> OnlineZipformer2CtcModel::GetInitStates() const {
  return impl_->GetInitStates();
}

std::vector<MNN::Express::VARP> OnlineZipformer2CtcModel::StackStates(
    std::vector<std::vector<MNN::Express::VARP>> states) const {
  return impl_->StackStates(std::move(states));
}

std::vector<std::vector<MNN::Express::VARP>> OnlineZipformer2CtcModel::UnStackStates(
    std::vector<MNN::Express::VARP> states) const {
  return impl_->UnStackStates(std::move(states));
}

#if __ANDROID_API__ >= 9
template OnlineZipformer2CtcModel::OnlineZipformer2CtcModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineZipformer2CtcModel::OnlineZipformer2CtcModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_mnn
