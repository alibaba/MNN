// sherpa-mnn/csrc/spoken-language-identification.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/spoken-language-identification.h"

#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/spoken-language-identification-impl.h"

namespace sherpa_mnn {

void SpokenLanguageIdentificationWhisperConfig::Register(ParseOptions *po) {
  po->Register(
      "whisper-encoder", &encoder,
      "Path to then encoder of a whisper multilingual model. Support only "
      "tiny, base, small, medium, large.");

  po->Register(
      "whisper-decoder", &decoder,
      "Path to the decoder of a whisper multilingual model. Support only "
      "tiny, base, small, medium, large.");

  po->Register(
      "whisper-tail-paddings", &tail_paddings,
      "Suggested value: 300 for multilingual models. "
      "Since we have removed the 30-second constraint, we need to add some "
      "tail padding frames "
      "so that whisper can detect the eot token. Leave it to -1 to use 1000");
}

bool SpokenLanguageIdentificationWhisperConfig::Validate() const {
  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --whisper-encoder");
    return false;
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("whisper encoder file '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --whisper-decoder");
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("whisper decoder file '%s' does not exist",
                     decoder.c_str());
    return false;
  }

  return true;
}

std::string SpokenLanguageIdentificationWhisperConfig::ToString() const {
  std::ostringstream os;

  os << "SpokenLanguageIdentificationWhisperConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "tail_paddings=" << tail_paddings << ")";

  return os.str();
}

void SpokenLanguageIdentificationConfig::Register(ParseOptions *po) {
  whisper.Register(po);

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool SpokenLanguageIdentificationConfig::Validate() const {
  if (!whisper.Validate()) {
    return false;
  }

  return true;
}

std::string SpokenLanguageIdentificationConfig::ToString() const {
  std::ostringstream os;

  os << "SpokenLanguageIdentificationConfig(";
  os << "whisper=" << whisper.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

SpokenLanguageIdentification::SpokenLanguageIdentification(
    const SpokenLanguageIdentificationConfig &config)
    : impl_(SpokenLanguageIdentificationImpl::Create(config)) {}

#if __ANDROID_API__ >= 9
SpokenLanguageIdentification::SpokenLanguageIdentification(
    AAssetManager *mgr, const SpokenLanguageIdentificationConfig &config)
    : impl_(SpokenLanguageIdentificationImpl::Create(mgr, config)) {}
#endif

SpokenLanguageIdentification::~SpokenLanguageIdentification() = default;

std::unique_ptr<OfflineStream> SpokenLanguageIdentification::CreateStream()
    const {
  return impl_->CreateStream();
}

std::string SpokenLanguageIdentification::Compute(OfflineStream *s) const {
  return impl_->Compute(s);
}

}  // namespace sherpa_mnn
