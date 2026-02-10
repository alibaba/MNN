// sherpa-mnn/csrc/online-lstm-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-mnn/csrc/online-lstm-transducer-model.h"

#include <algorithm>
#include <cassert>
#include <memory>
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
#include "sherpa-mnn/csrc/unbind.h"

namespace sherpa_mnn {

OnlineLstmTransducerModel::OnlineLstmTransducerModel(
    const OnlineModelConfig &config)
    : 
      config_(config),
      sess_opts_(GetSessionOptions(config)),
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
OnlineLstmTransducerModel::OnlineLstmTransducerModel(
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

void OnlineLstmTransducerModel::InitEncoder(void *model_data,
                                            size_t model_data_length) {
  encoder_sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data,
                                                 model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

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
  SHERPA_ONNX_READ_META_DATA(num_encoder_layers_, "num_encoder_layers");
  SHERPA_ONNX_READ_META_DATA(T_, "T");
  SHERPA_ONNX_READ_META_DATA(decode_chunk_len_, "decode_chunk_len");
  SHERPA_ONNX_READ_META_DATA(rnn_hidden_size_, "rnn_hidden_size");
  SHERPA_ONNX_READ_META_DATA(d_model_, "d_model");
}

void OnlineLstmTransducerModel::InitDecoder(void *model_data,
                                            size_t model_data_length) {
  decoder_sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data,
                                                 model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

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

void OnlineLstmTransducerModel::InitJoiner(void *model_data,
                                           size_t model_data_length) {
  joiner_sess_ = std::unique_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, (const uint8_t*)model_data,
                                                model_data_length, sess_opts_.pManager, &sess_opts_.pConfig));

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

std::vector<MNN::Express::VARP> OnlineLstmTransducerModel::StackStates(
    const std::vector<std::vector<MNN::Express::VARP>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());

  std::vector<MNN::Express::VARP > h_buf(batch_size);
  std::vector<MNN::Express::VARP > c_buf(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    assert(states[i].size() == 2);
    h_buf[i] = states[i][0];
    c_buf[i] = states[i][1];
  }
  auto allocator = const_cast<OnlineLstmTransducerModel *>(this)->allocator_;

  MNN::Express::VARP h = Cat(allocator, h_buf, 1);
  MNN::Express::VARP c = Cat(allocator, c_buf, 1);

  std::vector<MNN::Express::VARP> ans;
  ans.reserve(2);
  ans.push_back(std::move(h));
  ans.push_back(std::move(c));

  return ans;
}

std::vector<std::vector<MNN::Express::VARP>> OnlineLstmTransducerModel::UnStackStates(
    const std::vector<MNN::Express::VARP> &states) const {
  int32_t batch_size = states[0]->getInfo()->dim[1];
  assert(states.size() == 2);

  std::vector<std::vector<MNN::Express::VARP>> ans(batch_size);

  auto allocator = const_cast<OnlineLstmTransducerModel *>(this)->allocator_;

  std::vector<MNN::Express::VARP> h_vec = Unbind(allocator, states[0], 1);
  std::vector<MNN::Express::VARP> c_vec = Unbind(allocator, states[1], 1);

  assert(h_vec.size() == batch_size);
  assert(c_vec.size() == batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    ans[i].push_back(std::move(h_vec[i]));
    ans[i].push_back(std::move(c_vec[i]));
  }

  return ans;
}

std::vector<MNN::Express::VARP> OnlineLstmTransducerModel::GetEncoderInitStates() {
  // Please see
  // https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/lstm_transducer_stateless2/export-onnx.py#L185
  // for details
  constexpr int32_t kBatchSize = 1;
  std::array<int, 3> h_shape{num_encoder_layers_, kBatchSize, d_model_};
  MNN::Express::VARP h = MNNUtilsCreateTensor<float>(allocator_, h_shape.data(),
                                                 h_shape.size());

  Fill<float>(h, 0);

  std::array<int, 3> c_shape{num_encoder_layers_, kBatchSize,
                                 rnn_hidden_size_};

  MNN::Express::VARP c = MNNUtilsCreateTensor<float>(allocator_, c_shape.data(),
                                                 c_shape.size());

  Fill<float>(c, 0);

  std::vector<MNN::Express::VARP> states;

  states.reserve(2);
  states.push_back(std::move(h));
  states.push_back(std::move(c));

  return states;
}

std::pair<MNN::Express::VARP, std::vector<MNN::Express::VARP>>
OnlineLstmTransducerModel::RunEncoder(MNN::Express::VARP features,
                                      std::vector<MNN::Express::VARP> states,
                                      MNN::Express::VARP /* processed_frames */) {
  std::vector<MNN::Express::VARP> encoder_inputs = {
      std::move(features), std::move(states[0]), std::move(states[1])};

  auto encoder_out = encoder_sess_->onForward(encoder_inputs);

  std::vector<MNN::Express::VARP> next_states;
  next_states.reserve(2);
  next_states.push_back(std::move(encoder_out[1]));
  next_states.push_back(std::move(encoder_out[2]));

  return {std::move(encoder_out[0]), std::move(next_states)};
}

MNN::Express::VARP OnlineLstmTransducerModel::RunDecoder(MNN::Express::VARP decoder_input) {
  auto decoder_out = decoder_sess_->onForward({decoder_input});
  return std::move(decoder_out[0]);
}

MNN::Express::VARP OnlineLstmTransducerModel::RunJoiner(MNN::Express::VARP encoder_out,
                                                MNN::Express::VARP decoder_out) {
  std::vector<MNN::Express::VARP> joiner_input = {std::move(encoder_out),
                                            std::move(decoder_out)};
  auto logit =
      joiner_sess_->onForward(joiner_input);

  return std::move(logit[0]);
}

#if __ANDROID_API__ >= 9
template OnlineLstmTransducerModel::OnlineLstmTransducerModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineLstmTransducerModel::OnlineLstmTransducerModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_mnn
