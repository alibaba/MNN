// sherpa-mnn/csrc/offline-fire-red-asr-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-fire-red-asr-model.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

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

class OfflineFireRedAsrModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.fire_red_asr.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.fire_red_asr.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.fire_red_asr.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.fire_red_asr.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  std::pair<MNN::Express::VARP, MNN::Express::VARP> ForwardEncoder(MNN::Express::VARP features,
                                                   MNN::Express::VARP features_length) {
    std::vector<MNN::Express::VARP> inputs{std::move(features),
                                     std::move(features_length)};

    auto encoder_out = encoder_sess_->onForward(inputs);

    return {std::move(encoder_out[0]), std::move(encoder_out[1])};
  }

  std::tuple<MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP,
             MNN::Express::VARP>
  ForwardDecoder(MNN::Express::VARP tokens, MNN::Express::VARP n_layer_self_k_cache,
                 MNN::Express::VARP n_layer_self_v_cache, MNN::Express::VARP n_layer_cross_k,
                 MNN::Express::VARP n_layer_cross_v, MNN::Express::VARP offset) {
    std::vector<MNN::Express::VARP> decoder_input = {std::move(tokens),
                                               std::move(n_layer_self_k_cache),
                                               std::move(n_layer_self_v_cache),
                                               std::move(n_layer_cross_k),
                                               std::move(n_layer_cross_v),
                                               std::move(offset)};

    auto decoder_out = decoder_sess_->onForward(decoder_input);

    return std::tuple<MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP,
                      MNN::Express::VARP, MNN::Express::VARP>{
        std::move(decoder_out[0]),   std::move(decoder_out[1]),
        std::move(decoder_out[2]),   std::move(decoder_input[3]),
        std::move(decoder_input[4]), std::move(decoder_input[5])};
  }

  std::pair<MNN::Express::VARP, MNN::Express::VARP> GetInitialSelfKVCache() {
    int32_t batch_size = 1;
    std::array<int, 5> shape{meta_data_.num_decoder_layers, batch_size,
                                 meta_data_.max_len, meta_data_.num_head,
                                 meta_data_.head_dim};

    MNN::Express::VARP n_layer_self_k_cache = MNNUtilsCreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    MNN::Express::VARP n_layer_self_v_cache = MNNUtilsCreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    auto n = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];

    float *p_k = n_layer_self_k_cache->writeMap<float>();
    float *p_v = n_layer_self_v_cache->writeMap<float>();

    memset(p_k, 0, sizeof(float) * n);
    memset(p_v, 0, sizeof(float) * n);

    return {std::move(n_layer_self_k_cache), std::move(n_layer_self_v_cache)};
  }

  MNNAllocator *Allocator() { return allocator_; }

  const OfflineFireRedAsrModelMetaData& metaData() const {
    return meta_data_;
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
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_decoder_layers,
                               "num_decoder_layers");
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_head, "num_head");
    SHERPA_ONNX_READ_META_DATA(meta_data_.head_dim, "head_dim");
    SHERPA_ONNX_READ_META_DATA(meta_data_.sos_id, "sos");
    SHERPA_ONNX_READ_META_DATA(meta_data_.eos_id, "eos");
    SHERPA_ONNX_READ_META_DATA(meta_data_.max_len, "max_len");

    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.mean, "cmvn_mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.inv_stddev,
                                         "cmvn_inv_stddev");
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    decoder_sess_ = std::unique_ptr<MNN::Express::Module>(
        MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);
  }

 private:
  OfflineModelConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> encoder_sess_;
  std::unique_ptr<MNN::Express::Module> decoder_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  OfflineFireRedAsrModelMetaData meta_data_;
};

OfflineFireRedAsrModel::OfflineFireRedAsrModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineFireRedAsrModel::OfflineFireRedAsrModel(Manager *mgr,
                                               const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineFireRedAsrModel::~OfflineFireRedAsrModel() = default;

std::pair<MNN::Express::VARP, MNN::Express::VARP> OfflineFireRedAsrModel::ForwardEncoder(
    MNN::Express::VARP features, MNN::Express::VARP features_length) const {
  return impl_->ForwardEncoder(std::move(features), std::move(features_length));
}

std::tuple<MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP, MNN::Express::VARP,
           MNN::Express::VARP>
OfflineFireRedAsrModel::ForwardDecoder(MNN::Express::VARP tokens,
                                       MNN::Express::VARP n_layer_self_k_cache,
                                       MNN::Express::VARP n_layer_self_v_cache,
                                       MNN::Express::VARP n_layer_cross_k,
                                       MNN::Express::VARP n_layer_cross_v,
                                       MNN::Express::VARP offset) const {
  return impl_->ForwardDecoder(
      std::move(tokens), std::move(n_layer_self_k_cache),
      std::move(n_layer_self_v_cache), std::move(n_layer_cross_k),
      std::move(n_layer_cross_v), std::move(offset));
}

std::pair<MNN::Express::VARP, MNN::Express::VARP>
OfflineFireRedAsrModel::GetInitialSelfKVCache() const {
  return impl_->GetInitialSelfKVCache();
}

MNNAllocator *OfflineFireRedAsrModel::Allocator() const {
  return impl_->Allocator();
}

const OfflineFireRedAsrModelMetaData& OfflineFireRedAsrModel::metaData()
    const {
  return impl_->metaData();
}

#if __ANDROID_API__ >= 9
template OfflineFireRedAsrModel::OfflineFireRedAsrModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineFireRedAsrModel::OfflineFireRedAsrModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_mnn
