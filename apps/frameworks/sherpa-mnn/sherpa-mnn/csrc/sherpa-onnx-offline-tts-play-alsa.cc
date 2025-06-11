// sherpa-mnn/csrc/sherpa-mnn-tts-play-alsa.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

// see https://www.alsa-project.org/alsa-doc/alsa-lib/group___p_c_m.html
// https://www.alsa-project.org/alsa-doc/alsa-lib/group___p_c_m___h_w___params.html
// https://www.alsa-project.org/alsa-doc/alsa-lib/group___p_c_m.html

#include <signal.h>

#include <algorithm>
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <fstream>
#include <mutex>  // NOLINT
#include <queue>
#include <thread>  // NOLINT
#include <vector>

#include "sherpa-mnn/csrc/alsa-play.h"
#include "sherpa-mnn/csrc/offline-tts.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/wave-writer.h"

static std::condition_variable g_cv;
static std::mutex g_cv_m;

struct Buffer {
  std::queue<std::vector<float>> samples;
  std::mutex mutex;
};

static Buffer g_buffer;

static bool g_stopped = false;
static bool g_killed = false;

static void Handler(int32_t /*sig*/) {
  if (g_killed) {
    exit(0);
  }

  g_killed = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting\n");
}

static int32_t AudioGeneratedCallback(const float *s, int32_t n,
                                      float /*progress*/) {
  if (n > 0) {
    std::lock_guard<std::mutex> lock(g_buffer.mutex);
    g_buffer.samples.push({s, s + n});
    g_cv.notify_all();
  }

  if (g_killed) {
    return 0;  // stop generating
  }

  // continue generating
  return 1;
}

static void StartPlayback(const std::string &device_name, int32_t sample_rate) {
  sherpa_mnn::AlsaPlay alsa(device_name.c_str(), sample_rate);

  std::unique_lock<std::mutex> lock(g_cv_m);
  while (!g_killed && !g_stopped) {
    while (!g_buffer.samples.empty()) {
      auto &p = g_buffer.samples.front();
      alsa.Play(p);
      g_buffer.samples.pop();
    }

    g_cv.wait(lock);
  }

  if (g_killed) {
    return;
  }

  if (g_stopped) {
    while (!g_buffer.samples.empty()) {
      auto &p = g_buffer.samples.front();
      alsa.Play(p);
      g_buffer.samples.pop();
    }
  }

  alsa.Drain();
}

int main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
Offline text-to-speech with sherpa-mnn.

It plays the generated audio as the model is processing.

Note that it is alsa so it works only on **Linux**. For instance, you can
use it on Raspberry Pi.

Usage example:

wget https://github.com/k2-fsa/sherpa-mnn/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2

./bin/sherpa-mnn-offline-tts-play-alsa \
 --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
 --vits-tokens=./vits-piper-en_US-amy-low/tokens.txt \
 --vits-data-dir=./vits-piper-en_US-amy-low/espeak-ng-data \
 --output-filename=./generated.wav \
 "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."

It will generate a file ./generated.wav as specified by --output-filename.

You can find more models at
https://github.com/k2-fsa/sherpa-mnn/releases/tag/tts-models

Please see
https://k2-fsa.github.io/sherpa/onnx/tts/index.html
or details.
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  std::string device_name = "default";
  std::string output_filename = "./generated.wav";
  int32_t sid = 0;

  po.Register("output-filename", &output_filename,
              "Path to save the generated audio");

  po.Register("device-name", &device_name,
              "Name of the device to play the generated audio");

  po.Register("sid", &sid,
              "Speaker ID. Used only for multi-speaker models, e.g., models "
              "trained using the VCTK dataset. Not used for single-speaker "
              "models, e.g., models trained using the LJSpeech dataset");

  sherpa_mnn::OfflineTtsConfig config;

  config.Register(&po);
  po.Read(argc, argv);

  if (po.NumArgs() == 0) {
    fprintf(stderr, "Error: Please provide the text to generate audio.\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (po.NumArgs() > 1) {
    fprintf(stderr,
            "Error: Accept only one positional argument. Please use single "
            "quotes to wrap your text\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    exit(EXIT_FAILURE);
  }

  if (config.max_num_sentences != 1) {
    fprintf(stderr, "Setting config.max_num_sentences to 1\n");
    config.max_num_sentences = 1;
  }

  fprintf(stderr, "Loading the model\n");
  sherpa_mnn::OfflineTts tts(config);

  fprintf(stderr, "Start the playback thread\n");
  std::thread playback_thread(StartPlayback, device_name, tts.SampleRate());

  float speed = 1.0;

  fprintf(stderr, "Generating ...\n");
  const auto begin = std::chrono::steady_clock::now();
  auto audio = tts.Generate(po.GetArg(1), sid, speed, AudioGeneratedCallback);
  const auto end = std::chrono::steady_clock::now();
  g_stopped = true;
  g_cv.notify_all();
  fprintf(stderr, "Generating done!\n");
  if (audio.samples.empty()) {
    fprintf(
        stderr,
        "Error in generating audio. Please read previous error messages.\n");
    exit(EXIT_FAILURE);
  }

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = audio.samples.size() / static_cast<float>(audio.sample_rate);

  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  fprintf(stderr, "Audio duration: %.3f s\n", duration);
  fprintf(stderr, "Real-time factor (RTF): %.3f/%.3f = %.3f\n", elapsed_seconds,
          duration, rtf);

  bool ok = sherpa_mnn::WriteWave(output_filename, audio.sample_rate,
                                   audio.samples.data(), audio.samples.size());
  if (!ok) {
    fprintf(stderr, "Failed to write wave to %s\n", output_filename.c_str());
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "The text is: %s. Speaker ID: %d\n\n", po.GetArg(1).c_str(),
          sid);
  fprintf(stderr, "\n**** Saved to %s successfully! ****\n",
          output_filename.c_str());

  fprintf(stderr, "\n");
  fprintf(
      stderr,
      "Wait for the playback to finish. You can safely press ctrl + C to stop "
      "the playback.\n");
  playback_thread.join();

  fprintf(stderr, "Done!\n");

  return 0;
}
