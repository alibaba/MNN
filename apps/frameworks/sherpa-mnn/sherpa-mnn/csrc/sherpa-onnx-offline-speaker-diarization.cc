// sherpa-mnn/csrc/sherpa-mnn-offline-speaker-diarization.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-speaker-diarization.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/wave-reader.h"

static int32_t ProgressCallback(int32_t processed_chunks, int32_t num_chunks,
                                void *) {
  float progress = 100.0 * processed_chunks / num_chunks;
  fprintf(stderr, "progress %.2f%%\n", progress);

  // the return value is currently ignored
  return 0;
}

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Offline/Non-streaming speaker diarization with sherpa-mnn
Usage example:

Step 1: Download a speaker segmentation model

Please visit https://github.com/k2-fsa/sherpa-mnn/releases/tag/speaker-segmentation-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-mnn/releases/download/speaker-segmentation-models/sherpa-mnn-pyannote-segmentation-3-0.tar.bz2
  tar xvf sherpa-mnn-pyannote-segmentation-3-0.tar.bz2
  rm sherpa-mnn-pyannote-segmentation-3-0.tar.bz2

Step 2: Download a speaker embedding extractor model

Please visit https://github.com/k2-fsa/sherpa-mnn/releases/tag/speaker-recongition-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-mnn/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

Step 3. Download test wave files

Please visit https://github.com/k2-fsa/sherpa-mnn/releases/tag/speaker-segmentation-models
for a list of available test wave files. The following is an example

  wget https://github.com/k2-fsa/sherpa-mnn/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Step 4. Build sherpa-mnn

Step 5. Run it

  ./bin/sherpa-mnn-offline-speaker-diarization \
    --clustering.num-clusters=4 \
    --segmentation.pyannote-model=./sherpa-mnn-pyannote-segmentation-3-0/model.onnx \
    --embedding.model=./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
    ./0-four-speakers-zh.wav

Since we know that there are four speakers in the test wave file, we use
--clustering.num-clusters=4 in the above example.

If we don't know number of speakers in the given wave file, we can use
the argument --clustering.cluster-threshold. The following is an example:

  ./bin/sherpa-mnn-offline-speaker-diarization \
    --clustering.cluster-threshold=0.90 \
    --segmentation.pyannote-model=./sherpa-mnn-pyannote-segmentation-3-0/model.onnx \
    --embedding.model=./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
    ./0-four-speakers-zh.wav

A larger threshold leads to few clusters, i.e., few speakers;
a smaller threshold leads to more clusters, i.e., more speakers
  )usage";
  sherpa_mnn::OfflineSpeakerDiarizationConfig config;
  sherpa_mnn::ParseOptions po(kUsageMessage);
  config.Register(&po);
  po.Read(argc, argv);

  std::cout << config.ToString() << "\n";

  if (!config.Validate()) {
    po.PrintUsage();
    std::cerr << "Errors in config!\n";
    return -1;
  }

  if (po.NumArgs() != 1) {
    std::cerr << "Error: Please provide exactly 1 wave file.\n\n";
    po.PrintUsage();
    return -1;
  }

  sherpa_mnn::OfflineSpeakerDiarization sd(config);

  std::cout << "Started\n";
  const auto begin = std::chrono::steady_clock::now();
  const std::string wav_filename = po.GetArg(1);
  int32_t sample_rate = -1;
  bool is_ok = false;
  const std::vector<float> samples =
      sherpa_mnn::ReadWave(wav_filename, &sample_rate, &is_ok);
  if (!is_ok) {
    std::cerr << "Failed to read " << wav_filename.c_str() << "\n";
    return -1;
  }

  if (sample_rate != sd.SampleRate()) {
    std::cerr << "Expect sample rate " << sd.SampleRate()
              << ". Given: " << sample_rate << "\n";
    return -1;
  }

  float duration = samples.size() / static_cast<float>(sample_rate);

  auto result =
      sd.Process(samples.data(), samples.size(), ProgressCallback, nullptr)
          .SortByStartTime();

  for (const auto &r : result) {
    std::cout << r.ToString() << "\n";
  }

  const auto end = std::chrono::steady_clock::now();
  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "Duration : %.3f s\n", duration);
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
