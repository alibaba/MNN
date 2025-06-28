// sherpa-mnn/csrc/sherpa-mnn-offline-language-identification.cc
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#include <stdio.h>

#include <chrono>  // NOLINT
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/spoken-language-identification.h"
#include "sherpa-mnn/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Spoken language identification with sherpa-mnn.

Usage:

(1) Use a whisper multilingual model

wget https://github.com/k2-fsa/sherpa-mnn/releases/download/asr-models/sherpa-mnn-whisper-tiny.tar.bz2
tar xvf sherpa-mnn-whisper-tiny.tar.bz2
rm sherpa-mnn-whisper-tiny.tar.bz2

We only use the int8.onnx models below.

./bin/sherpa-mnn-offline-spoken-language-identification \
  --whisper-encoder=sherpa-mnn-whisper-tiny/tiny-encoder.int8.onnx \
  --whisper-decoder=sherpa-mnn-whisper-tiny/tiny-decoder.int8.onnx \
  --num-threads=1 \
  /path/to/foo.wav

foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.
You can find test waves for different languages at
https://hf-mirror.com/spaces/k2-fsa/spoken-language-identification/tree/main/test_wavs

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/index.html
Note that only whisper multilingual models are supported. For instance,
"tiny" is supported but "tiny.en" is not.
for a list of pre-trained models to download.
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::SpokenLanguageIdentificationConfig config;
  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 1) {
    fprintf(stderr, "Error: Please provide 1 wave file.\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  fprintf(stderr, "Creating spoken language identifier ...\n");
  sherpa_mnn::SpokenLanguageIdentification slid(config);

  fprintf(stderr, "Started\n");
  const std::string wav_filename = po.GetArg(1);

  int32_t sampling_rate = -1;
  bool is_ok = false;
  const std::vector<float> samples =
      sherpa_mnn::ReadWave(wav_filename, &sampling_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "Failed to read '%s'\n", wav_filename.c_str());
    return -1;
  }
  float duration = samples.size() / static_cast<float>(sampling_rate);

  const auto begin = std::chrono::steady_clock::now();

  auto s = slid.CreateStream();
  s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

  auto language = slid.Compute(s.get());

  const auto end = std::chrono::steady_clock::now();

  fprintf(stderr, "Done!\n\n");
  fprintf(stderr, "%s\nDetected language: %s\n", wav_filename.c_str(),
          language.c_str());

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "num threads: %d\n", config.num_threads);

  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
