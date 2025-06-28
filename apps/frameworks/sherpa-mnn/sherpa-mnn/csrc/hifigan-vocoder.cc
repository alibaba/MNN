// sherpa-mnn/csrc/hifigan-vocoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/hifigan-vocoder.h"

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

namespace sherpa_mnn {

class HifiganVocoder::Impl {
 public:
  explicit Impl(int32_t num_threads, const std::string &provider,
                const std::string &model)
      :
        sess_opts_(GetSessionOptions(num_threads, provider)),
        allocator_{} {
    auto buf = ReadFile(model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  explicit Impl(Manager *mgr, int32_t num_threads, const std::string &provider,
                const std::string &model)
      :
        sess_opts_(GetSessionOptions(num_threads, provider)),
        allocator_{} {
    auto buf = ReadFile(mgr, model);
    Init(buf.data(), buf.size());
  }

  MNN::Express::VARP Run(MNN::Express::VARP mel) const {
    auto out = sess_->onForward({mel});
    return std::move(out[0]);
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length,
                                           sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
  }

 private:
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

HifiganVocoder::HifiganVocoder(int32_t num_threads, const std::string &provider,
                               const std::string &model)
    : impl_(std::make_unique<Impl>(num_threads, provider, model)) {}

template <typename Manager>
HifiganVocoder::HifiganVocoder(Manager *mgr, int32_t num_threads,
                               const std::string &provider,
                               const std::string &model)
    : impl_(std::make_unique<Impl>(mgr, num_threads, provider, model)) {}

HifiganVocoder::~HifiganVocoder() = default;

MNN::Express::VARP HifiganVocoder::Run(MNN::Express::VARP mel) const {
  return impl_->Run(std::move(mel));
}

#if __ANDROID_API__ >= 9
template HifiganVocoder::HifiganVocoder(AAssetManager *mgr, int32_t num_threads,
                                        const std::string &provider,
                                        const std::string &model);
#endif

#if __OHOS__
template HifiganVocoder::HifiganVocoder(NativeResourceManager *mgr,
                                        int32_t num_threads,
                                        const std::string &provider,
                                        const std::string &model);
#endif

}  // namespace sherpa_mnn
