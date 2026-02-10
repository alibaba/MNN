// sherpa-mnn/csrc/offline-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-transducer-model.h"

#include <algorithm>
#include <string>
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

namespace sherpa_mnn {

class OfflineTransducerModel::Impl {
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

  std::pair<MNN::Express::VARP, MNN::Express::VARP> RunEncoder(MNN::Express::VARP features,
                                               MNN::Express::VARP features_length) {
    std::vector<MNN::Express::VARP> encoder_inputs = {std::move(features),
                                                std::move(features_length)};

    auto encoder_out = encoder_sess_->onForward(encoder_inputs);

    return {std::move(encoder_out[0]), std::move(encoder_out[1])};
  }

  MNN::Express::VARP RunDecoder(MNN::Express::VARP decoder_input) {
    auto decoder_out = decoder_sess_->onForward({decoder_input});
    return std::move(decoder_out[0]);
  }

  MNN::Express::VARP RunJoiner(MNN::Express::VARP encoder_out, MNN::Express::VARP decoder_out) {
    std::vector<MNN::Express::VARP> joiner_input = {std::move(encoder_out),
                                              std::move(decoder_out)};
    auto logit = joiner_sess_->onForward(
                                   joiner_input);

    return std::move(logit[0]);
  }

  int32_t VocabSize() const { return vocab_size_; }
  int32_t ContextSize() const { return context_size_; }
  int32_t SubsamplingFactor() const { return 4; }
  MNNAllocator *Allocator() { return allocator_; }

  MNN::Express::VARP BuildDecoderInput(
      const std::vector<OfflineTransducerDecoderResult> &results,
      int32_t end_index) {
    assert(end_index <= results.size());

    int32_t batch_size = end_index;
    int32_t context_size = ContextSize();
    std::array<int, 2> shape{batch_size, context_size};

    MNN::Express::VARP decoder_input = MNNUtilsCreateTensor<int>(
        Allocator(), shape.data(), shape.size());
    int *p = decoder_input->writeMap<int>();

    for (int32_t i = 0; i != batch_size; ++i) {
      const auto &r = results[i];
      const int *begin = r.tokens.data() + r.tokens.size() - context_size;
      const int *end = r.tokens.data() + r.tokens.size();
      std::copy(begin, end, p);
      p += context_size;
    }

    return decoder_input;
  }

  MNN::Express::VARP BuildDecoderInput(const std::vector<Hypothesis> &results,
                               int32_t end_index) {
    assert(end_index <= results.size());

    int32_t batch_size = end_index;
    int32_t context_size = ContextSize();
    std::array<int, 2> shape{batch_size, context_size};

    MNN::Express::VARP decoder_input = MNNUtilsCreateTensor<int>(
        Allocator(), shape.data(), shape.size());
    int *p = decoder_input->writeMap<int>();

    for (int32_t i = 0; i != batch_size; ++i) {
      const auto &r = results[i];
      const int *begin = r.ys.data() + r.ys.size() - context_size;
      const int *end = r.ys.data() + r.ys.size();
      std::copy(begin, end, p);
      p += context_size;
    }

    return decoder_input;
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
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
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
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    SHERPA_ONNX_READ_META_DATA(context_size_, "context_size");
  }

  void InitJoiner(void *model_data, size_t model_data_length) {
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
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }
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

  int32_t vocab_size_ = 0;    // initialized in InitDecoder
  int32_t context_size_ = 0;  // initialized in InitDecoder
};

OfflineTransducerModel::OfflineTransducerModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTransducerModel::OfflineTransducerModel(Manager *mgr,
                                               const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTransducerModel::~OfflineTransducerModel() = default;

std::pair<MNN::Express::VARP, MNN::Express::VARP> OfflineTransducerModel::RunEncoder(
    MNN::Express::VARP features, MNN::Express::VARP features_length) {
  return impl_->RunEncoder(std::move(features), std::move(features_length));
}

MNN::Express::VARP OfflineTransducerModel::RunDecoder(MNN::Express::VARP decoder_input) {
  return impl_->RunDecoder(std::move(decoder_input));
}

MNN::Express::VARP OfflineTransducerModel::RunJoiner(MNN::Express::VARP encoder_out,
                                             MNN::Express::VARP decoder_out) {
  return impl_->RunJoiner(std::move(encoder_out), std::move(decoder_out));
}

int32_t OfflineTransducerModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OfflineTransducerModel::ContextSize() const {
  return impl_->ContextSize();
}

int32_t OfflineTransducerModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

MNNAllocator *OfflineTransducerModel::Allocator() const {
  return impl_->Allocator();
}

MNN::Express::VARP OfflineTransducerModel::BuildDecoderInput(
    const std::vector<OfflineTransducerDecoderResult> &results,
    int32_t end_index) const {
  return impl_->BuildDecoderInput(results, end_index);
}

MNN::Express::VARP OfflineTransducerModel::BuildDecoderInput(
    const std::vector<Hypothesis> &results, int32_t end_index) const {
  return impl_->BuildDecoderInput(results, end_index);
}

#if __ANDROID_API__ >= 9
template OfflineTransducerModel::OfflineTransducerModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineTransducerModel::OfflineTransducerModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_mnn
