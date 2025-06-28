// sherpa-mnn/csrc/sherpa-mnn-offline-audio-tagging.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <stdio.h>

#include "sherpa-mnn/csrc/audio-tagging.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/wave-reader.h"

int32_t main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Audio tagging from a file.

Usage:

wget https://github.com/k2-fsa/sherpa-mnn/releases/download/audio-tagging-models/sherpa-mnn-zipformer-audio-tagging-2024-04-09.tar.bz2
tar xvf sherpa-mnn-zipformer-audio-tagging-2024-04-09.tar.bz2
rm sherpa-mnn-zipformer-audio-tagging-2024-04-09.tar.bz2

./bin/sherpa-mnn-offline-audio-tagging \
  --zipformer-model=./sherpa-mnn-zipformer-audio-tagging-2024-04-09/model.onnx \
  --labels=./sherpa-mnn-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv \
  sherpa-mnn-zipformer-audio-tagging-2024-04-09/test_wavs/0.wav

Input wave files should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please see
https://github.com/k2-fsa/sherpa-mnn/releases/tag/audio-tagging-models
for more models.
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::AudioTaggingConfig config;
  config.Register(&po);
  po.Read(argc, argv);

  if (po.NumArgs() != 1) {
    fprintf(stderr, "\nError: Please provide 1 wave file\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  sherpa_mnn::AudioTagging tagger(config);
  std::string wav_filename = po.GetArg(1);

  int32_t sampling_rate = -1;

  bool is_ok = false;
  const std::vector<float> samples =
      sherpa_mnn::ReadWave(wav_filename, &sampling_rate, &is_ok);

  if (!is_ok) {
    fprintf(stderr, "Failed to read '%s'\n", wav_filename.c_str());
    return -1;
  }

  const float duration = samples.size() / static_cast<float>(sampling_rate);

  fprintf(stderr, "Start to compute\n");
  const auto begin = std::chrono::steady_clock::now();

  auto stream = tagger.CreateStream();

  stream->AcceptWaveform(sampling_rate, samples.data(), samples.size());

  auto results = tagger.Compute(stream.get());
  const auto end = std::chrono::steady_clock::now();
  fprintf(stderr, "Done\n");

  int32_t i = 0;

  for (const auto &event : results) {
    fprintf(stderr, "%d: %s\n", i, event.ToString().c_str());
    i += 1;
  }

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Num threads: %d\n", config.model.num_threads);
  fprintf(stderr, "Wave duration: %.3f\n", duration);
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
