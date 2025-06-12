// sherpa-mnn/csrc/offline-rnn-lm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-rnn-lm.h"

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

class OfflineRnnLM::Impl {
 public:
  explicit Impl(const OfflineLMConfig &config)
      : config_(config),
        sess_opts_{GetSessionOptions(config)},
        allocator_{} {
    auto buf = ReadFile(config_.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineLMConfig &config)
      : config_(config),
        sess_opts_{GetSessionOptions(config)},
        allocator_{} {
    auto buf = ReadFile(mgr, config_.model);
    Init(buf.data(), buf.size());
  }

  MNN::Express::VARP Rescore(MNN::Express::VARP x, MNN::Express::VARP x_lens) {
    std::vector<MNN::Express::VARP> inputs = {std::move(x), std::move(x_lens)};

    auto out =
        sess_->onForward(inputs);

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
  OfflineLMConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineRnnLM::OfflineRnnLM(const OfflineLMConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineRnnLM::OfflineRnnLM(Manager *mgr, const OfflineLMConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineRnnLM::~OfflineRnnLM() = default;

MNN::Express::VARP OfflineRnnLM::Rescore(MNN::Express::VARP x, MNN::Express::VARP x_lens) {
  return impl_->Rescore(std::move(x), std::move(x_lens));
}

#if __ANDROID_API__ >= 9
template OfflineRnnLM::OfflineRnnLM(AAssetManager *mgr,
                                    const OfflineLMConfig &config);
#endif

#if __OHOS__
template OfflineRnnLM::OfflineRnnLM(NativeResourceManager *mgr,
                                    const OfflineLMConfig &config);
#endif

}  // namespace sherpa_mnn
