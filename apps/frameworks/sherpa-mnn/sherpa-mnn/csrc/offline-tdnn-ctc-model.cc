// sherpa-mnn/csrc/offline-tdnn-ctc-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-tdnn-ctc-model.h"

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
#include "sherpa-mnn/csrc/transpose.h"

namespace sherpa_mnn {

class OfflineTdnnCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.tdnn.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.tdnn.model);
    Init(buf.data(), buf.size());
  }

  std::vector<MNN::Express::VARP> Forward(MNN::Express::VARP features) {
    auto nnet_out =
        sess_->onForward({features});

    std::vector<int> nnet_out_shape =
        nnet_out[0]->getInfo()->dim;

    std::vector<int> out_length_vec(nnet_out_shape[0], nnet_out_shape[1]);
    std::vector<int> out_length_shape(1, nnet_out_shape[0]);

    auto memory_info =
        (MNNAllocator*)(nullptr);

    MNN::Express::VARP nnet_out_length = MNNUtilsCreateTensor(
        memory_info, out_length_vec.data(), out_length_vec.size(),
        out_length_shape.data(), out_length_shape.size());

    std::vector<MNN::Express::VARP> ans;
    ans.reserve(2);
    ans.push_back(std::move(nnet_out[0]));
    ans.push_back(Clone(nullptr, nnet_out_length));
    return ans;
  }

  int32_t VocabSize() const { return vocab_size_; }

  MNNAllocator *Allocator() { return allocator_; }

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
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    MNNAllocator* allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
  }

 private:
  OfflineModelConfig config_;
  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t vocab_size_ = 0;
};

OfflineTdnnCtcModel::OfflineTdnnCtcModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTdnnCtcModel::OfflineTdnnCtcModel(Manager *mgr,
                                         const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTdnnCtcModel::~OfflineTdnnCtcModel() = default;

std::vector<MNN::Express::VARP> OfflineTdnnCtcModel::Forward(
    MNN::Express::VARP features, MNN::Express::VARP /*features_length*/) {
  return impl_->Forward(std::move(features));
}

int32_t OfflineTdnnCtcModel::VocabSize() const { return impl_->VocabSize(); }

MNNAllocator *OfflineTdnnCtcModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineTdnnCtcModel::OfflineTdnnCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineTdnnCtcModel::OfflineTdnnCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_mnn
