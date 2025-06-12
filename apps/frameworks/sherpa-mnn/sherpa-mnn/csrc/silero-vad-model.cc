// sherpa-mnn/csrc/silero-vad-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/csrc/silero-vad-model.h"

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

class SileroVadModel::Impl {
 public:
  explicit Impl(const VadModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(config.silero_vad.model);
    Init(buf.data(), buf.size());

    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config.sample_rate);
      exit(-1);
    }

    min_silence_samples_ =
        sample_rate_ * config_.silero_vad.min_silence_duration;

    min_speech_samples_ = sample_rate_ * config_.silero_vad.min_speech_duration;
  }

  template <typename Manager>
  Impl(Manager *mgr, const VadModelConfig &config)
      : config_(config),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(mgr, config.silero_vad.model);
    Init(buf.data(), buf.size());

    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config.sample_rate);
      exit(-1);
    }

    min_silence_samples_ =
        sample_rate_ * config_.silero_vad.min_silence_duration;

    min_speech_samples_ = sample_rate_ * config_.silero_vad.min_speech_duration;
  }

  void Reset() {
    if (is_v5_) {
      ResetV5();
    } else {
      ResetV4();
    }

    triggered_ = false;
    current_sample_ = 0;
    temp_start_ = 0;
    temp_end_ = 0;
  }

  bool IsSpeech(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_ONNX_LOGE("n: %d != window_size: %d", n, WindowSize());
      exit(-1);
    }

    float prob = Run(samples, n);

    float threshold = config_.silero_vad.threshold;

    current_sample_ += config_.silero_vad.window_size;

    if (prob > threshold && temp_end_ != 0) {
      temp_end_ = 0;
    }

    if (prob > threshold && temp_start_ == 0) {
      // start speaking, but we require that it must satisfy
      // min_speech_duration
      temp_start_ = current_sample_;
      return false;
    }

    if (prob > threshold && temp_start_ != 0 && !triggered_) {
      if (current_sample_ - temp_start_ < min_speech_samples_) {
        return false;
      }

      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && !triggered_) {
      // silence
      temp_start_ = 0;
      temp_end_ = 0;
      return false;
    }

    if ((prob > threshold - 0.15) && triggered_) {
      // speaking
      return true;
    }

    if ((prob > threshold) && !triggered_) {
      // start speaking
      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && triggered_) {
      // stop to speak
      if (temp_end_ == 0) {
        temp_end_ = current_sample_;
      }

      if (current_sample_ - temp_end_ < min_silence_samples_) {
        // continue speaking
        return true;
      }
      // stopped speaking
      temp_start_ = 0;
      temp_end_ = 0;
      triggered_ = false;
      return false;
    }

    return false;
  }

  int32_t WindowShift() const { return config_.silero_vad.window_size; }

  int32_t WindowSize() const {
    return config_.silero_vad.window_size + window_overlap_;
  }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }

  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

  void SetMinSilenceDuration(float s) {
    min_silence_samples_ = sample_rate_ * s;
  }

  void SetThreshold(float threshold) {
    config_.silero_vad.threshold = threshold;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data, model_data_length,
                                           sess_opts_.pManager, &sess_opts_.pConfig));

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    if (input_names_.size() == 4 && output_names_.size() == 3) {
      is_v5_ = false;
    } else if (input_names_.size() == 3 && output_names_.size() == 2) {
      is_v5_ = true;

      // 64 for 16kHz
      // 32 for 8kHz
      window_overlap_ = 64;

      if (config_.silero_vad.window_size != 512) {
        SHERPA_ONNX_LOGE(
            "For silero_vad  v5, we require window_size to be 512 for 16kHz");
        exit(-1);
      }
    } else {
      SHERPA_ONNX_LOGE("Unsupported silero vad model");
      exit(-1);
    }

    Check();

    Reset();
  }

  void ResetV5() {
    // 2 - number of LSTM layer
    // 1 - batch size
    // 128 - hidden dim
    std::array<int, 3> shape{2, 1, 128};

    MNN::Express::VARP s =
        MNNUtilsCreateTensor<float>(allocator_, shape.data(), shape.size());

    Fill<float>(s, 0);
    states_.clear();
    states_.push_back(std::move(s));
  }

  void ResetV4() {
    // 2 - number of LSTM layer
    // 1 - batch size
    // 64 - hidden dim
    std::array<int, 3> shape{2, 1, 64};

    MNN::Express::VARP h =
        MNNUtilsCreateTensor<float>(allocator_, shape.data(), shape.size());

    MNN::Express::VARP c =
        MNNUtilsCreateTensor<float>(allocator_, shape.data(), shape.size());

    Fill<float>(h, 0);
    Fill<float>(c, 0);

    states_.clear();

    states_.reserve(2);
    states_.push_back(std::move(h));
    states_.push_back(std::move(c));
  }

  void Check() const {
    if (is_v5_) {
      CheckV5();
    } else {
      CheckV4();
    }
  }

  void CheckV4() const {
    if (input_names_.size() != 4) {
      SHERPA_ONNX_LOGE("Expect 4 inputs. Given: %d",
                       static_cast<int32_t>(input_names_.size()));
      exit(-1);
    }

    if (input_names_[0] != "input") {
      SHERPA_ONNX_LOGE("Input[0]: %s. Expected: input",
                       input_names_[0].c_str());
      exit(-1);
    }

    if (input_names_[1] != "sr") {
      SHERPA_ONNX_LOGE("Input[1]: %s. Expected: sr", input_names_[1].c_str());
      exit(-1);
    }

    if (input_names_[2] != "h") {
      SHERPA_ONNX_LOGE("Input[2]: %s. Expected: h", input_names_[2].c_str());
      exit(-1);
    }

    if (input_names_[3] != "c") {
      SHERPA_ONNX_LOGE("Input[3]: %s. Expected: c", input_names_[3].c_str());
      exit(-1);
    }

    // Now for outputs
    if (output_names_.size() != 3) {
      SHERPA_ONNX_LOGE("Expect 3 outputs. Given: %d",
                       static_cast<int32_t>(output_names_.size()));
      exit(-1);
    }

    if (output_names_[0] != "output") {
      SHERPA_ONNX_LOGE("Output[0]: %s. Expected: output",
                       output_names_[0].c_str());
      exit(-1);
    }

    if (output_names_[1] != "hn") {
      SHERPA_ONNX_LOGE("Output[1]: %s. Expected: sr", output_names_[1].c_str());
      exit(-1);
    }

    if (output_names_[2] != "cn") {
      SHERPA_ONNX_LOGE("Output[2]: %s. Expected: sr", output_names_[2].c_str());
      exit(-1);
    }
  }

  void CheckV5() const {
    if (input_names_.size() != 3) {
      SHERPA_ONNX_LOGE("Expect 3 inputs. Given: %d",
                       static_cast<int32_t>(input_names_.size()));
      exit(-1);
    }

    if (input_names_[0] != "input") {
      SHERPA_ONNX_LOGE("Input[0]: %s. Expected: input",
                       input_names_[0].c_str());
      exit(-1);
    }

    if (input_names_[1] != "state") {
      SHERPA_ONNX_LOGE("Input[1]: %s. Expected: state",
                       input_names_[1].c_str());
      exit(-1);
    }

    if (input_names_[2] != "sr") {
      SHERPA_ONNX_LOGE("Input[2]: %s. Expected: sr", input_names_[2].c_str());
      exit(-1);
    }

    // Now for outputs
    if (output_names_.size() != 2) {
      SHERPA_ONNX_LOGE("Expect 2 outputs. Given: %d",
                       static_cast<int32_t>(output_names_.size()));
      exit(-1);
    }

    if (output_names_[0] != "output") {
      SHERPA_ONNX_LOGE("Output[0]: %s. Expected: output",
                       output_names_[0].c_str());
      exit(-1);
    }

    if (output_names_[1] != "stateN") {
      SHERPA_ONNX_LOGE("Output[1]: %s. Expected: stateN",
                       output_names_[1].c_str());
      exit(-1);
    }
  }

  float Run(const float *samples, int32_t n) {
    if (is_v5_) {
      return RunV5(samples, n);
    } else {
      return RunV4(samples, n);
    }
  }

  float RunV5(const float *samples, int32_t n) {
    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 2> x_shape = {1, n};

    MNN::Express::VARP x =
        MNNUtilsCreateTensor(memory_info, const_cast<float *>(samples), n,
                                 x_shape.data(), x_shape.size());

    int sr_shape = 1;
    MNN::Express::VARP sr =
        MNNUtilsCreateTensor(memory_info, &sample_rate_, 1, &sr_shape, 1);

    std::vector<MNN::Express::VARP> inputs = {std::move(x), std::move(states_[0]),
                                        std::move(sr)};

    auto out =
        sess_->onForward(inputs);

    states_[0] = std::move(out[1]);

    float prob = out[0]->readMap<float>()[0];
    return prob;
  }

  float RunV4(const float *samples, int32_t n) {
    auto memory_info =
        (MNNAllocator*)(nullptr);

    std::array<int, 2> x_shape = {1, n};

    MNN::Express::VARP x =
        MNNUtilsCreateTensor(memory_info, const_cast<float *>(samples), n,
                                 x_shape.data(), x_shape.size());

    int sr_shape = 1;
    MNN::Express::VARP sr =
        MNNUtilsCreateTensor(memory_info, &sample_rate_, 1, &sr_shape, 1);

    std::vector<MNN::Express::VARP> inputs = {std::move(x), std::move(sr),
                                        std::move(states_[0]),
                                        std::move(states_[1])};

    auto out =
        sess_->onForward(inputs);

    states_[0] = std::move(out[1]);
    states_[1] = std::move(out[2]);

    float prob = out[0]->readMap<float>()[0];
    return prob;
  }

 private:
  VadModelConfig config_;

  MNNEnv env_;
  MNNConfig sess_opts_;
  MNNAllocator* allocator_;

  std::unique_ptr<MNN::Express::Module> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  std::vector<MNN::Express::VARP> states_;
  int sample_rate_;
  int32_t min_silence_samples_;
  int32_t min_speech_samples_;

  bool triggered_ = false;
  int32_t current_sample_ = 0;
  int32_t temp_start_ = 0;
  int32_t temp_end_ = 0;

  int32_t window_overlap_ = 0;

  bool is_v5_ = false;
};

SileroVadModel::SileroVadModel(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
SileroVadModel::SileroVadModel(Manager *mgr, const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

SileroVadModel::~SileroVadModel() = default;

void SileroVadModel::Reset() { return impl_->Reset(); }

bool SileroVadModel::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}

int32_t SileroVadModel::WindowSize() const { return impl_->WindowSize(); }

int32_t SileroVadModel::WindowShift() const { return impl_->WindowShift(); }

int32_t SileroVadModel::MinSilenceDurationSamples() const {
  return impl_->MinSilenceDurationSamples();
}

int32_t SileroVadModel::MinSpeechDurationSamples() const {
  return impl_->MinSpeechDurationSamples();
}

void SileroVadModel::SetMinSilenceDuration(float s) {
  impl_->SetMinSilenceDuration(s);
}

void SileroVadModel::SetThreshold(float threshold) {
  impl_->SetThreshold(threshold);
}

#if __ANDROID_API__ >= 9
template SileroVadModel::SileroVadModel(AAssetManager *mgr,
                                        const VadModelConfig &config);
#endif

#if __OHOS__
template SileroVadModel::SileroVadModel(NativeResourceManager *mgr,
                                        const VadModelConfig &config);
#endif

}  // namespace sherpa_mnn
