// sherpa-mnn/csrc/sherpa-mnn-offline.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>

#include <chrono>  // NOLINT
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/offline-recognizer.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Speech recognition using non-streaming models with sherpa-mnn.

Usage:

(1) Transducer from icefall

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html

  ./bin/sherpa-mnn-offline \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    /path/to/foo.wav [bar.wav foobar.wav ...]


(2) Paraformer from FunASR

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html

  ./bin/sherpa-mnn-offline \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(3) Moonshine models

See https://k2-fsa.github.io/sherpa/onnx/moonshine/index.html

  ./bin/sherpa-mnn-offline \
    --moonshine-preprocessor=/Users/fangjun/open-source/sherpa-mnn/scripts/moonshine/preprocess.onnx \
    --moonshine-encoder=/Users/fangjun/open-source/sherpa-mnn/scripts/moonshine/encode.int8.onnx \
    --moonshine-uncached-decoder=/Users/fangjun/open-source/sherpa-mnn/scripts/moonshine/uncached_decode.int8.onnx \
    --moonshine-cached-decoder=/Users/fangjun/open-source/sherpa-mnn/scripts/moonshine/cached_decode.int8.onnx \
    --tokens=/Users/fangjun/open-source/sherpa-mnn/scripts/moonshine/tokens.txt \
    --num-threads=1 \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(4) Whisper models

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html

  ./bin/sherpa-mnn-offline \
    --whisper-encoder=./sherpa-mnn-whisper-base.en/base.en-encoder.int8.onnx \
    --whisper-decoder=./sherpa-mnn-whisper-base.en/base.en-decoder.int8.onnx \
    --tokens=./sherpa-mnn-whisper-base.en/base.en-tokens.txt \
    --num-threads=1 \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(5) NeMo CTC models

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html

  ./bin/sherpa-mnn-offline \
    --tokens=./sherpa-mnn-nemo-ctc-en-conformer-medium/tokens.txt \
    --nemo-ctc-model=./sherpa-mnn-nemo-ctc-en-conformer-medium/model.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    --debug=false \
    ./sherpa-mnn-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
    ./sherpa-mnn-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
    ./sherpa-mnn-nemo-ctc-en-conformer-medium/test_wavs/8k.wav

(6) TDNN CTC model for the yesno recipe from icefall

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/yesno/index.html
      //
  ./build/bin/sherpa-mnn-offline \
    --sample-rate=8000 \
    --feat-dim=23 \
    --tokens=./sherpa-mnn-tdnn-yesno/tokens.txt \
    --tdnn-model=./sherpa-mnn-tdnn-yesno/model-epoch-14-avg-2.onnx \
    ./sherpa-mnn-tdnn-yesno/test_wavs/0_0_0_1_0_0_0_1.wav \
    ./sherpa-mnn-tdnn-yesno/test_wavs/0_0_1_0_0_0_1_0.wav

Note: It supports decoding multiple files in batches

foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::OfflineRecognizerConfig config;
  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() < 1) {
    fprintf(stderr, "Error: Please provide at least 1 wave file.\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  fprintf(stderr, "Creating recognizer ...\n");
  sherpa_mnn::OfflineRecognizer recognizer(config);

  fprintf(stderr, "Started\n");
  const auto begin = std::chrono::steady_clock::now();

  std::vector<std::unique_ptr<sherpa_mnn::OfflineStream>> ss;
  std::vector<sherpa_mnn::OfflineStream *> ss_pointers;
  float duration = 0;
  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    std::string wav_filename = po.GetArg(i);
    int32_t sampling_rate = -1;
    bool is_ok = false;
    std::vector<float> samples =
        sherpa_mnn::ReadWave(wav_filename, &sampling_rate, &is_ok);
    if (!is_ok) {
      fprintf(stderr, "Failed to read '%s'\n", wav_filename.c_str());
      return -1;
    }
    duration += samples.size() / static_cast<float>(sampling_rate);

    auto s = recognizer.CreateStream();
    s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

    ss.push_back(std::move(s));
    ss_pointers.push_back(ss.back().get());
  }

  recognizer.DecodeStreams(ss_pointers.data(), ss_pointers.size());

  const auto end = std::chrono::steady_clock::now();

  fprintf(stderr, "Done!\n\n");
  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    fprintf(stderr, "%s\n%s\n----\n", po.GetArg(i).c_str(),
            ss[i - 1]->GetResult().AsJsonString().c_str());
  }

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "num threads: %d\n", config.model_config.num_threads);
  fprintf(stderr, "decoding method: %s\n", config.decoding_method.c_str());
  if (config.decoding_method == "modified_beam_search") {
    fprintf(stderr, "max active paths: %d\n", config.max_active_paths);
  }

  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
