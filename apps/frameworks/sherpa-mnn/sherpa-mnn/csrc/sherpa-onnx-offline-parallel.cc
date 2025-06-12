// sherpa-mnn/csrc/sherpa-mnn-offline-parallel.cc
//
// Copyright (c)  2022-2023  cuidc

#include <stdio.h>

#include <atomic>
#include <chrono>  // NOLINT
#include <fstream>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "sherpa-mnn/csrc/offline-recognizer.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/wave-reader.h"

std::atomic<int> wav_index(0);
std::mutex mtx;

std::vector<std::vector<std::string>> SplitToBatches(
    const std::vector<std::string> &input, int32_t batch_size) {
  std::vector<std::vector<std::string>> outputs;
  auto itr = input.cbegin();
  int32_t process_num = 0;

  while (process_num + batch_size <= static_cast<int32_t>(input.size())) {
    auto chunk_end = itr + batch_size;
    outputs.emplace_back(itr, chunk_end);
    itr = chunk_end;
    process_num += batch_size;
  }
  if (itr != input.cend()) {
    outputs.emplace_back(itr, input.cend());
  }
  return outputs;
}

std::vector<std::string> LoadScpFile(const std::string &wav_scp_path) {
  std::vector<std::string> wav_paths;
  std::ifstream in(wav_scp_path);
  if (!in.is_open()) {
    fprintf(stderr, "Failed to open file: %s.\n", wav_scp_path.c_str());
    return wav_paths;
  }
  std::string line, column1, column2;
  while (std::getline(in, line)) {
    std::istringstream iss(line);
    iss >> column1 >> column2;
    wav_paths.emplace_back(std::move(column2));
  }

  return wav_paths;
}

void AsrInference(const std::vector<std::vector<std::string>> &chunk_wav_paths,
                  sherpa_mnn::OfflineRecognizer *recognizer,
                  float *total_length, float *total_time) {
  std::vector<std::unique_ptr<sherpa_mnn::OfflineStream>> ss;
  std::vector<sherpa_mnn::OfflineStream *> ss_pointers;
  float duration = 0.0f;
  float elapsed_seconds_batch = 0.0f;

  // warm up
  for (const auto &wav_filename : chunk_wav_paths[0]) {
    int32_t sampling_rate = -1;
    bool is_ok = false;
    const std::vector<float> samples =
        sherpa_mnn::ReadWave(wav_filename, &sampling_rate, &is_ok);
    if (!is_ok) {
      fprintf(stderr, "Failed to read '%s'\n", wav_filename.c_str());
      continue;
    }
    duration += samples.size() / static_cast<float>(sampling_rate);
    auto s = recognizer->CreateStream();
    s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

    ss.push_back(std::move(s));
    ss_pointers.push_back(ss.back().get());
  }
  recognizer->DecodeStreams(ss_pointers.data(), ss_pointers.size());
  ss_pointers.clear();
  ss.clear();

  while (true) {
    int chunk = wav_index.fetch_add(1);
    if (chunk >= static_cast<int32_t>(chunk_wav_paths.size())) {
      break;
    }
    const auto &wav_paths = chunk_wav_paths[chunk];
    const auto begin = std::chrono::steady_clock::now();
    for (const auto &wav_filename : wav_paths) {
      int32_t sampling_rate = -1;
      bool is_ok = false;
      const std::vector<float> samples =
          sherpa_mnn::ReadWave(wav_filename, &sampling_rate, &is_ok);
      if (!is_ok) {
        fprintf(stderr, "Failed to read '%s'\n", wav_filename.c_str());
        continue;
      }
      duration += samples.size() / static_cast<float>(sampling_rate);
      auto s = recognizer->CreateStream();
      s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

      ss.push_back(std::move(s));
      ss_pointers.push_back(ss.back().get());
    }
    recognizer->DecodeStreams(ss_pointers.data(), ss_pointers.size());
    const auto end = std::chrono::steady_clock::now();
    float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count() /
        1000.;
    elapsed_seconds_batch += elapsed_seconds;
    int i = 0;
    for (const auto &wav_filename : wav_paths) {
      fprintf(stderr, "%s\n%s\n----\n", wav_filename.c_str(),
              ss[i]->GetResult().AsJsonString().c_str());
      i = i + 1;
    }
    ss_pointers.clear();
    ss.clear();
  }

  {
    std::lock_guard<std::mutex> guard(mtx);
    *total_length += duration;
    if (*total_time < elapsed_seconds_batch) {
      *total_time = elapsed_seconds_batch;
    }
  }
}

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Speech recognition using non-streaming models with sherpa-mnn.

Usage:

(1) Transducer from icefall

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html

  ./bin/sherpa-mnn-offline-parallel \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    --batch-size=8 \
    --nj=1 \
    --wav-scp=wav.scp

  ./bin/sherpa-mnn-offline-parallel \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    --batch-size=1 \
    --nj=8 \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(2) Paraformer from FunASR

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html

  ./bin/sherpa-mnn-offline-parallel \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(3) Whisper models

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html

  ./bin/sherpa-mnn-offline-parallel \
    --whisper-encoder=./sherpa-mnn-whisper-base.en/base.en-encoder.int8.onnx \
    --whisper-decoder=./sherpa-mnn-whisper-base.en/base.en-decoder.int8.onnx \
    --tokens=./sherpa-mnn-whisper-base.en/base.en-tokens.txt \
    --num-threads=1 \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(4) NeMo CTC models

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html

  ./bin/sherpa-mnn-offline-parallel \
    --tokens=./sherpa-mnn-nemo-ctc-en-conformer-medium/tokens.txt \
    --nemo-ctc-model=./sherpa-mnn-nemo-ctc-en-conformer-medium/model.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    --debug=false \
    ./sherpa-mnn-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
    ./sherpa-mnn-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
    ./sherpa-mnn-nemo-ctc-en-conformer-medium/test_wavs/8k.wav

(5) TDNN CTC model for the yesno recipe from icefall

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/yesno/index.html
      //
  ./bin/sherpa-mnn-offline-parallel \
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
  std::string wav_scp = "";  // file path, kaldi style wav list.
  int32_t nj = 1;            // thread number
  int32_t batch_size = 1;    // number of wav files processed at once.
  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::OfflineRecognizerConfig config;
  config.Register(&po);
  po.Register("wav-scp", &wav_scp,
              "a file including wav-id and wav-path, kaldi style wav list."
              "default="
              ". when it is not empty, wav files which positional "
              "parameters provide are invalid.");
  po.Register("nj", &nj, "multi-thread num for decoding, default=1");
  po.Register("batch-size", &batch_size,
              "number of wav files processed at once during the decoding"
              "process. default=1");

  po.Read(argc, argv);
  if (po.NumArgs() < 1 && wav_scp.empty()) {
    fprintf(stderr, "Error: Please provide at least 1 wave file.\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }
  std::this_thread::sleep_for(std::chrono::seconds(10));  // sleep 10s
  fprintf(stderr, "Creating recognizer ...\n");
  const auto begin = std::chrono::steady_clock::now();
  sherpa_mnn::OfflineRecognizer recognizer(config);
  const auto end = std::chrono::steady_clock::now();
  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  fprintf(stderr,
          "Started nj: %d, batch_size: %d, wav_path: %s. recognizer init time: "
          "%.6f\n",
          nj, batch_size, wav_scp.c_str(), elapsed_seconds);
  std::this_thread::sleep_for(std::chrono::seconds(10));  // sleep 10s
  std::vector<std::string> wav_paths;
  if (!wav_scp.empty()) {
    wav_paths = LoadScpFile(wav_scp);
  } else {
    for (int32_t i = 1; i <= po.NumArgs(); ++i) {
      wav_paths.emplace_back(po.GetArg(i));
    }
  }
  if (wav_paths.empty()) {
    fprintf(stderr, "wav files is empty.\n");
    return -1;
  }
  std::vector<std::thread> threads;
  std::vector<std::vector<std::string>> batch_wav_paths =
      SplitToBatches(wav_paths, batch_size);
  float total_length = 0.0f;
  float total_time = 0.0f;
  for (int i = 0; i < nj; i++) {
    threads.emplace_back(std::thread(AsrInference, batch_wav_paths, &recognizer,
                                     &total_length, &total_time));
  }

  for (auto &thread : threads) {
    thread.join();
  }

  fprintf(stderr, "num threads: %d\n", config.model_config.num_threads);
  fprintf(stderr, "decoding method: %s\n", config.decoding_method.c_str());
  if (config.decoding_method == "modified_beam_search") {
    fprintf(stderr, "max active paths: %d\n", config.max_active_paths);
  }
  fprintf(stderr, "Elapsed seconds: %.3f s\n", total_time);
  float rtf = total_time / total_length;
  fprintf(stderr, "Real time factor (RTF): %.6f / %.6f = %.4f\n", total_time,
          total_length, rtf);
  fprintf(stderr, "SPEEDUP: %.4f\n", 1.0 / rtf);

  return 0;
}
