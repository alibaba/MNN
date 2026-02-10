// sherpa-mnn/csrc/online-punctuation-model-config.cc
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#include "sherpa-mnn/csrc/online-punctuation-model-config.h"

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"

namespace sherpa_mnn {

void OnlinePunctuationModelConfig::Register(ParseOptions *po) {
  po->Register("cnn-bilstm", &cnn_bilstm,
               "Path to the light-weight CNN-BiLSTM model");

  po->Register("bpe-vocab", &bpe_vocab, "Path to the bpe vocab file");

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool OnlinePunctuationModelConfig::Validate() const {
  if (cnn_bilstm.empty()) {
    SHERPA_ONNX_LOGE("Please provide --cnn-bilstm");
    return false;
  }

  if (!FileExists(cnn_bilstm)) {
    SHERPA_ONNX_LOGE("--cnn-bilstm '%s' does not exist", cnn_bilstm.c_str());
    return false;
  }

  if (bpe_vocab.empty()) {
    SHERPA_ONNX_LOGE("Please provide --bpe-vocab");
    return false;
  }

  if (!FileExists(bpe_vocab)) {
    SHERPA_ONNX_LOGE("--bpe-vocab '%s' does not exist", bpe_vocab.c_str());
    return false;
  }

  return true;
}

std::string OnlinePunctuationModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlinePunctuationModelConfig(";
  os << "cnn_bilstm=\"" << cnn_bilstm << "\", ";
  os << "bpe_vocab=\"" << bpe_vocab << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_mnn
