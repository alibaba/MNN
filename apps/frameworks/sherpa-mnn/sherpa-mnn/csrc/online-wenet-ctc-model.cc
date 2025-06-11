// sherpa-mnn/csrc/online-wenet-ctc-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-wenet-ctc-model.h"

#include <algorithm>
#include <cmath>
#include <string>

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

class OnlineWenetCtcModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.wenet_ctc.model);
      Init(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.wenet_ctc.model);
      Init(buf.data(), buf.size());
    }
  }

  std::vector<MNN::Express::VARP> Forward(MNN::Express::VARP x,
                                  std::vector<MNN::Express::VARP> states) {
    MNN::Express::VARP &attn_cache = states[0];
    MNN::Express::VARP &conv_cache = states[1];
    MNN::Express::VARP &offset = states[2];

    int32_t chunk_size = config_.wenet_ctc.chunk_size;
    int32_t left_chunks = config_.wenet_ctc.num_left_chunks;
    // build attn_mask
    std::array<int, 3> attn_mask_shape{1, 1,
                                           required_cache_size_ + chunk_size};
    MNN::Express::VARP attn_mask = MNNUtilsCreateTensor<bool>(
        allocator_, attn_mask_shape.data(), attn_mask_shape.size());
    bool *p = attn_mask->writeMap<bool>();
    int32_t chunk_idx =
        offset->readMap<int>()[0] / chunk_size - left_chunks;
    if (chunk_idx < left_chunks) {
      std::fill(p, p + required_cache_size_ - chunk_idx * chunk_size, 0);
      std::fill(p + required_cache_size_ - chunk_idx * chunk_size,
                p + attn_mask_shape[2], 1);
    } else {
      std::fill(p, p + attn_mask_shape[2], 1);
    }

    std::vector<MNN::Express::VARP> inputs = {std::move(x),
                                        View(offset),
                                        View(required_cache_size_tensor_),
                                        std::move(attn_cache),
                                        std::move(conv_cache),
                                        std::move(attn_mask)};

    auto out =
        sess_->onForward(inputs);

    offset->writeMap<int>()[0] +=
        out[0]->getInfo()->dim[1];
    out.push_back(std::move(offset));

    return out;
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t ChunkLength() const {
    // When chunk_size is 16, subsampling_factor_ is 4, right_context_ is 6,
    // the returned value is (16 - 1)*4 + 6 + 1 = 67
    return (config_.wenet_ctc.chunk_size - 1) * subsampling_factor_ +
           right_context_ + 1;
  }

  int32_t ChunkShift() const {
    return config_.wenet_ctc.chunk_size * subsampling_factor_;
  }

  MNNAllocator *Allocator() { return allocator_; }

  // Return a vector containing 3 tensors
  // - attn_cache
  // - conv_cache
  // - offset
  std::vector<MNN::Express::VARP> GetInitStates() {
    std::vector<MNN::Express::VARP> ans;
    ans.reserve(3);
    ans.push_back(View(attn_cache_));
    ans.push_back(View(conv_cache_));

    int offset_shape = 1;

    MNN::Express::VARP offset =
        MNNUtilsCreateTensor<int>(allocator_, &offset_shape, 1);

    offset->writeMap<int>()[0] = required_cache_size_;

    ans.push_back(std::move(offset));

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
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(head_, "head");
    SHERPA_ONNX_READ_META_DATA(num_blocks_, "num_blocks");
    SHERPA_ONNX_READ_META_DATA(output_size_, "output_size");
    SHERPA_ONNX_READ_META_DATA(cnn_module_kernel_, "cnn_module_kernel");
    SHERPA_ONNX_READ_META_DATA(right_context_, "right_context");
    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");

    required_cache_size_ =
        config_.wenet_ctc.chunk_size * config_.wenet_ctc.num_left_chunks;

    InitStates();
  }

  void InitStates() {
    std::array<int, 4> attn_cache_shape{
        num_blocks_, head_, required_cache_size_, output_size_ / head_ * 2};
    attn_cache_ = MNNUtilsCreateTensor<float>(
        allocator_, attn_cache_shape.data(), attn_cache_shape.size());

    Fill<float>(attn_cache_, 0);

    std::array<int, 4> conv_cache_shape{num_blocks_, 1, output_size_,
                                            cnn_module_kernel_ - 1};
    conv_cache_ = MNNUtilsCreateTensor<float>(
        allocator_, conv_cache_shape.data(), conv_cache_shape.size());

    Fill<float>(conv_cache_, 0);

    int shape = 1;
    required_cache_size_tensor_ =
        MNNUtilsCreateTensor<int>(allocator_, &shape, 1);

    required_cache_size_tensor_->writeMap<int>()[0] =
        required_cache_size_;
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

  int32_t head_ = 0;
  int32_t num_blocks_ = 0;
  int32_t output_size_ = 0;
  int32_t cnn_module_kernel_ = 0;
  int32_t right_context_ = 0;
  int32_t subsampling_factor_ = 0;
  int32_t vocab_size_ = 0;

  int32_t required_cache_size_ = 0;

  MNN::Express::VARP attn_cache_{nullptr};
  MNN::Express::VARP conv_cache_{nullptr};
  MNN::Express::VARP required_cache_size_tensor_{nullptr};
};

OnlineWenetCtcModel::OnlineWenetCtcModel(const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineWenetCtcModel::OnlineWenetCtcModel(Manager *mgr,
                                         const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OnlineWenetCtcModel::~OnlineWenetCtcModel() = default;

std::vector<MNN::Express::VARP> OnlineWenetCtcModel::Forward(
    MNN::Express::VARP x, std::vector<MNN::Express::VARP> states) const {
  return impl_->Forward(std::move(x), std::move(states));
}

int32_t OnlineWenetCtcModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OnlineWenetCtcModel::ChunkLength() const {
  return impl_->ChunkLength();
}

int32_t OnlineWenetCtcModel::ChunkShift() const { return impl_->ChunkShift(); }

MNNAllocator *OnlineWenetCtcModel::Allocator() const {
  return impl_->Allocator();
}

std::vector<MNN::Express::VARP> OnlineWenetCtcModel::GetInitStates() const {
  return impl_->GetInitStates();
}

std::vector<MNN::Express::VARP> OnlineWenetCtcModel::StackStates(
    std::vector<std::vector<MNN::Express::VARP>> states) const {
  if (states.size() != 1) {
    SHERPA_ONNX_LOGE("wenet CTC model supports only batch_size==1. Given: %d",
                     static_cast<int32_t>(states.size()));
  }

  return std::move(states[0]);
}

std::vector<std::vector<MNN::Express::VARP>> OnlineWenetCtcModel::UnStackStates(
    std::vector<MNN::Express::VARP> states) const {
  std::vector<std::vector<MNN::Express::VARP>> ans(1);
  ans[0] = std::move(states);
  return ans;
}

#if __ANDROID_API__ >= 9
template OnlineWenetCtcModel::OnlineWenetCtcModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineWenetCtcModel::OnlineWenetCtcModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_mnn
