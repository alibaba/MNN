// sherpa-mnn/csrc/sherpa-mnn-vad-with-offline-asr.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <stdio.h>

#include <chrono>  // NOLINT
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/offline-recognizer.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/resample.h"
#include "sherpa-mnn/csrc/voice-activity-detector.h"
#include "sherpa-mnn/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Speech recognition using VAD + non-streaming models with sherpa-mnn.

Usage:

Note you can download silero_vad.onnx using

wget https://github.com/k2-fsa/sherpa-mnn/releases/download/asr-models/silero_vad.onnx

(0) FireRedAsr

See https://k2-fsa.github.io/sherpa/onnx/FireRedAsr/pretrained.html

  ./bin/sherpa-mnn-vad-with-offline-asr \
    --tokens=./sherpa-mnn-fire-red-asr-large-zh_en-2025-02-16/tokens.txt \
    --fire-red-asr-encoder=./sherpa-mnn-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx \
    --fire-red-asr-decoder=./sherpa-mnn-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx \
    --num-threads=1 \
    --silero-vad-model=/path/to/silero_vad.onnx \
    /path/to/foo.wav

(1) Transducer from icefall

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html

  ./bin/sherpa-mnn-vad-with-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    /path/to/foo.wav


(2) Paraformer from FunASR

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html

  ./bin/sherpa-mnn-vad-with-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    /path/to/foo.wav

(3) Moonshine models

See https://k2-fsa.github.io/sherpa/onnx/moonshine/index.html

  ./bin/sherpa-mnn-vad-with-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --moonshine-preprocessor=/Users/fangjun/open-source/sherpa-mnn/scripts/moonshine/preprocess.onnx \
    --moonshine-encoder=/Users/fangjun/open-source/sherpa-mnn/scripts/moonshine/encode.int8.onnx \
    --moonshine-uncached-decoder=/Users/fangjun/open-source/sherpa-mnn/scripts/moonshine/uncached_decode.int8.onnx \
    --moonshine-cached-decoder=/Users/fangjun/open-source/sherpa-mnn/scripts/moonshine/cached_decode.int8.onnx \
    --tokens=/Users/fangjun/open-source/sherpa-mnn/scripts/moonshine/tokens.txt \
    --num-threads=1 \
    /path/to/foo.wav

(4) Whisper models

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html

  ./bin/sherpa-mnn-vad-with-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --whisper-encoder=./sherpa-mnn-whisper-base.en/base.en-encoder.int8.onnx \
    --whisper-decoder=./sherpa-mnn-whisper-base.en/base.en-decoder.int8.onnx \
    --tokens=./sherpa-mnn-whisper-base.en/base.en-tokens.txt \
    --num-threads=1 \
    /path/to/foo.wav

(5) NeMo CTC models

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html

  ./bin/sherpa-mnn-vad-with-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --tokens=./sherpa-mnn-nemo-ctc-en-conformer-medium/tokens.txt \
    --nemo-ctc-model=./sherpa-mnn-nemo-ctc-en-conformer-medium/model.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    --debug=false \
    ./sherpa-mnn-nemo-ctc-en-conformer-medium/test_wavs/0.wav

(6) TDNN CTC model for the yesno recipe from icefall

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/yesno/index.html

  ./bin/sherpa-mnn-vad-with-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --sample-rate=8000 \
    --feat-dim=23 \
    --tokens=./sherpa-mnn-tdnn-yesno/tokens.txt \
    --tdnn-model=./sherpa-mnn-tdnn-yesno/model-epoch-14-avg-2.onnx \
    ./sherpa-mnn-tdnn-yesno/test_wavs/0_0_0_1_0_0_0_1.wav

The input wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::OfflineRecognizerConfig asr_config;
  asr_config.Register(&po);

  sherpa_mnn::VadModelConfig vad_config;
  vad_config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 1) {
    fprintf(stderr, "Error: Please provide at only 1 wave file. Given: %d\n\n",
            po.NumArgs());
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", vad_config.ToString().c_str());
  fprintf(stderr, "%s\n", asr_config.ToString().c_str());

  if (!vad_config.Validate()) {
    fprintf(stderr, "Errors in vad_config!\n");
    return -1;
  }

  if (!asr_config.Validate()) {
    fprintf(stderr, "Errors in ASR config!\n");
    return -1;
  }

  fprintf(stderr, "Creating recognizer ...\n");
  sherpa_mnn::OfflineRecognizer recognizer(asr_config);
  fprintf(stderr, "Recognizer created!\n");

  auto vad = std::make_unique<sherpa_mnn::VoiceActivityDetector>(vad_config);

  fprintf(stderr, "Started\n");
  const auto begin = std::chrono::steady_clock::now();

  std::string wave_filename = po.GetArg(1);
  fprintf(stderr, "Reading: %s\n", wave_filename.c_str());
  int32_t sampling_rate = -1;
  bool is_ok = false;
  auto samples = sherpa_mnn::ReadWave(wave_filename, &sampling_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "Failed to read '%s'\n", wave_filename.c_str());
    return -1;
  }

  if (sampling_rate != 16000) {
    fprintf(stderr, "Resampling from %d Hz to 16000 Hz", sampling_rate);
    float min_freq = std::min<int32_t>(sampling_rate, 16000);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    auto resampler = std::make_unique<sherpa_mnn::LinearResample>(
        sampling_rate, 16000, lowpass_cutoff, lowpass_filter_width);
    std::vector<float> out_samples;
    resampler->Resample(samples.data(), samples.size(), true, &out_samples);
    samples = std::move(out_samples);
    fprintf(stderr, "Resampling done\n");
  }

  fprintf(stderr, "Started!\n");
  int32_t window_size = vad_config.silero_vad.window_size;
  int32_t i = 0;
  while (i + window_size < samples.size()) {
    vad->AcceptWaveform(samples.data() + i, window_size);
    i += window_size;
    if (i >= samples.size()) {
      vad->Flush();
    }

    while (!vad->Empty()) {
      const auto &segment = vad->Front();
      float duration = segment.samples.size() / 16000.;
      float start_time = segment.start / 16000.;
      float end_time = start_time + duration;
      if (duration < 0.1) {
        vad->Pop();
        continue;
      }

      auto s = recognizer.CreateStream();
      s->AcceptWaveform(16000, segment.samples.data(), segment.samples.size());
      recognizer.DecodeStream(s.get());
      const auto &result = s->GetResult();
      if (!result.text.empty()) {
        fprintf(stderr, "%.3f -- %.3f: %s\n", start_time, end_time,
                result.text.c_str());
      }
      vad->Pop();
    }
  }

  const auto end = std::chrono::steady_clock::now();

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "num threads: %d\n", asr_config.model_config.num_threads);
  fprintf(stderr, "decoding method: %s\n", asr_config.decoding_method.c_str());
  if (asr_config.decoding_method == "modified_beam_search") {
    fprintf(stderr, "max active paths: %d\n", asr_config.max_active_paths);
  }

  float duration = samples.size() / 16000.;
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
