// sherpa-mnn/csrc/offline-moonshine-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-moonshine-model.h"

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

class OfflineMoonshineModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.moonshine.preprocessor);
      InitPreprocessor(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.moonshine.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.moonshine.uncached_decoder);
      InitUnCachedDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.moonshine.cached_decoder);
      InitCachedDecoder(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.moonshine.preprocessor);
      InitPreprocessor(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.moonshine.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.moonshine.uncached_decoder);
      InitUnCachedDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.moonshine.cached_decoder);
      InitCachedDecoder(buf.data(), buf.size());
    }
  }

  MNN::Express::VARP ForwardPreprocessor(MNN::Express::VARP audio) {
    auto features = preprocessor_sess_->onForward({audio});

    return std::move(features[0]);
  }

  MNN::Express::VARP ForwardEncoder(MNN::Express::VARP features, MNN::Express::VARP features_len) {
    std::vector<MNN::Express::VARP> encoder_inputs{std::move(features),
                                             std::move(features_len)};
    auto encoder_out = encoder_sess_->onForward(encoder_inputs);

    return std::move(encoder_out[0]);
  }

  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> ForwardUnCachedDecoder(
      MNN::Express::VARP tokens, MNN::Express::VARP seq_len, MNN::Express::VARP encoder_out) {
    std::vector<MNN::Express::VARP> uncached_decoder_input = {
        std::move(tokens),
        std::move(encoder_out),
        std::move(seq_len),
    };

    auto uncached_decoder_out = uncached_decoder_sess_->onForward(
        uncached_decoder_input);

    std::vector<MNN::Express::VARP> states;
    states.reserve(uncached_decoder_out.size() - 1);

    int32_t i = -1;
    for (auto &s : uncached_decoder_out) {
      ++i;
      if (i == 0) {
        continue;
      }

      states.push_back(std::move(s));
    }

    return {std::move(uncached_decoder_out[0]), std::move(states)};
  }

  std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>> ForwardCachedDecoder(
      MNN::Express::VARP tokens, MNN::Express::VARP seq_len, MNN::Express::VARP encoder_out,
      std::vector<MNN::Express::VARP> states) {
    std::vector<MNN::Express::VARP> cached_decoder_input;
    cached_decoder_input.reserve(3 + states.size());
    cached_decoder_input.push_back(std::move(tokens));
    cached_decoder_input.push_back(std::move(encoder_out));
    cached_decoder_input.push_back(std::move(seq_len));

    for (auto &s : states) {
      cached_decoder_input.push_back(std::move(s));
    }

    auto cached_decoder_out = cached_decoder_sess_->onForward(cached_decoder_input);

    std::vector<MNN::Express::VARP> next_states;
    next_states.reserve(cached_decoder_out.size() - 1);

    int32_t i = -1;
    for (auto &s : cached_decoder_out) {
      ++i;
      if (i == 0) {
        continue;
      }

      next_states.push_back(std::move(s));
    }

    return {std::move(cached_decoder_out[0]), std::move(next_states)};
  }

  MNNAllocator *Allocator() { return allocator_; }

 private:
  void InitPreprocessor(void *model_data, size_t model_data_length) {
    preprocessor_sess_ = std::unique_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(preprocessor_sess_.get(), &preprocessor_input_names_,
                  &preprocessor_input_names_ptr_);

    GetOutputNames(preprocessor_sess_.get(), &preprocessor_output_names_,
                   &preprocessor_output_names_ptr_);
  }

  void InitEncoder(void *model_data, size_t model_data_length) {
    encoder_sess_ = std::unique_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);
  }

  void InitUnCachedDecoder(void *model_data, size_t model_data_length) {
    uncached_decoder_sess_ = std::unique_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(uncached_decoder_sess_.get(), &uncached_decoder_input_names_,
                  &uncached_decoder_input_names_ptr_);

    GetOutputNames(uncached_decoder_sess_.get(),
                   &uncached_decoder_output_names_,
                   &uncached_decoder_output_names_ptr_);
  }

  void InitCachedDecoder(void *model_data, size_t model_data_length) {
    cached_decoder_sess_ = std::unique_ptr<MNN::Express::Module>(
      MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(cached_decoder_sess_.get(), &cached_decoder_input_names_,
                  &cached_decoder_input_names_ptr_);

    GetOutputNames(cached_decoder_sess_.get(), &cached_decoder_output_names_,
                   &cached_decoder_output_names_ptr_);
  }

 private:
  OfflineModelConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> preprocessor_sess_;
  std::unique_ptr<MNN::Express::Module> encoder_sess_;
  std::unique_ptr<MNN::Express::Module> uncached_decoder_sess_;
  std::unique_ptr<MNN::Express::Module> cached_decoder_sess_;

  std::vector<std::string> preprocessor_input_names_;
  std::vector<const char *> preprocessor_input_names_ptr_;

  std::vector<std::string> preprocessor_output_names_;
  std::vector<const char *> preprocessor_output_names_ptr_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> uncached_decoder_input_names_;
  std::vector<const char *> uncached_decoder_input_names_ptr_;

  std::vector<std::string> uncached_decoder_output_names_;
  std::vector<const char *> uncached_decoder_output_names_ptr_;

  std::vector<std::string> cached_decoder_input_names_;
  std::vector<const char *> cached_decoder_input_names_ptr_;

  std::vector<std::string> cached_decoder_output_names_;
  std::vector<const char *> cached_decoder_output_names_ptr_;
};

OfflineMoonshineModel::OfflineMoonshineModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineMoonshineModel::OfflineMoonshineModel(Manager *mgr,
                                             const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineMoonshineModel::~OfflineMoonshineModel() = default;

MNN::Express::VARP OfflineMoonshineModel::ForwardPreprocessor(MNN::Express::VARP audio) const {
  return impl_->ForwardPreprocessor(std::move(audio));
}

MNN::Express::VARP OfflineMoonshineModel::ForwardEncoder(
    MNN::Express::VARP features, MNN::Express::VARP features_len) const {
  return impl_->ForwardEncoder(std::move(features), std::move(features_len));
}

std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>>
OfflineMoonshineModel::ForwardUnCachedDecoder(MNN::Express::VARP token,
                                              MNN::Express::VARP seq_len,
                                              MNN::Express::VARP encoder_out) const {
  return impl_->ForwardUnCachedDecoder(std::move(token), std::move(seq_len),
                                       std::move(encoder_out));
}

std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>>
OfflineMoonshineModel::ForwardCachedDecoder(
    MNN::Express::VARP token, MNN::Express::VARP seq_len, MNN::Express::VARP encoder_out,
    std::vector<MNN::Express::VARP> states) const {
  return impl_->ForwardCachedDecoder(std::move(token), std::move(seq_len),
                                     std::move(encoder_out), std::move(states));
}

MNNAllocator *OfflineMoonshineModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineMoonshineModel::OfflineMoonshineModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineMoonshineModel::OfflineMoonshineModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_mnn
