// sherpa-mnn/csrc/sherpa-mnn-offline-tts-play.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include <signal.h>

#include <algorithm>
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <fstream>
#include <mutex>  // NOLINT
#include <queue>
#include <thread>  // NOLINT
#include <vector>

#include "portaudio.h"  // NOLINT
#include "sherpa-mnn/csrc/microphone.h"
#include "sherpa-mnn/csrc/offline-tts.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/wave-writer.h"

static std::condition_variable g_cv;
static std::mutex g_cv_m;

struct Samples {
  std::vector<float> data;
  int32_t consumed = 0;
};

struct Buffer {
  std::queue<Samples> samples;
  std::mutex mutex;
};

static Buffer g_buffer;

static bool g_started = false;
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
    Samples samples;
    samples.data = std::vector<float>{s, s + n};

    std::lock_guard<std::mutex> lock(g_buffer.mutex);
    g_buffer.samples.push(std::move(samples));
    g_started = true;
  }
  if (g_killed) {
    return 0;  // stop generating
  }

  // continue generating
  return 1;
}

static int PlayCallback(const void * /*in*/, void *out,
                        unsigned long n,  // NOLINT
                        const PaStreamCallbackTimeInfo * /*time_info*/,
                        PaStreamCallbackFlags /*status_flags*/,
                        void * /*user_data*/) {
  if (g_killed) {
    return paComplete;
  }

  float *pout = reinterpret_cast<float *>(out);
  std::lock_guard<std::mutex> lock(g_buffer.mutex);

  if (g_buffer.samples.empty()) {
    if (g_stopped) {
      // no more data is available and we have processed all of the samples
      return paComplete;
    }

    // The current sentence is so long, though very unlikely, that
    // the model has not finished processing it yet.
    std::fill_n(pout, n, 0);

    return paContinue;
  }

  int32_t k = 0;
  for (; k < static_cast<int32_t>(n) && !g_buffer.samples.empty();) {
    int32_t this_block = n - k;

    auto &p = g_buffer.samples.front();

    int32_t remaining = p.data.size() - p.consumed;

    if (this_block <= remaining) {
      std::copy(p.data.begin() + p.consumed,
                p.data.begin() + p.consumed + this_block, pout + k);
      p.consumed += this_block;

      k = n;

      if (p.consumed == static_cast<int32_t>(p.data.size())) {
        g_buffer.samples.pop();
      }
      break;
    }

    std::copy(p.data.begin() + p.consumed, p.data.end(), pout + k);
    k += p.data.size() - p.consumed;
    g_buffer.samples.pop();
  }

  if (k < static_cast<int32_t>(n)) {
    std::fill_n(pout + k, n - k, 0);
  }

  if (g_stopped && g_buffer.samples.empty()) {
    return paComplete;
  }

  return paContinue;
}

static void PlayCallbackFinished(void * /*userData*/) { g_cv.notify_all(); }

static void StartPlayback(int32_t sample_rate) {
  int32_t frames_per_buffer = 1024;
  PaStreamParameters outputParameters;
  PaStream *stream;
  PaError err;

  outputParameters.device =
      Pa_GetDefaultOutputDevice(); /* default output device */

  outputParameters.channelCount = 1;         /* stereo output */
  outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
  outputParameters.suggestedLatency =
      Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
  outputParameters.hostApiSpecificStreamInfo = nullptr;

  err = Pa_OpenStream(&stream, nullptr, /* no input */
                      &outputParameters, sample_rate, frames_per_buffer,
                      paClipOff,  // we won't output out of range samples so
                                  //   don't bother clipping them
                      PlayCallback, nullptr);
  if (err != paNoError) {
    fprintf(stderr, "%d portaudio error: %s\n", __LINE__, Pa_GetErrorText(err));
    return;
  }

  err = Pa_SetStreamFinishedCallback(stream, &PlayCallbackFinished);
  if (err != paNoError) {
    fprintf(stderr, "%d portaudio error: %s\n", __LINE__, Pa_GetErrorText(err));
    return;
  }

  err = Pa_StartStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "%d portaudio error: %s\n", __LINE__, Pa_GetErrorText(err));
    return;
  }

  std::unique_lock<std::mutex> lock(g_cv_m);
  while (!g_killed && !g_stopped &&
         (!g_started || (g_started && !g_buffer.samples.empty()))) {
    g_cv.wait(lock);
  }

  err = Pa_StopStream(stream);
  if (err != paNoError) {
    return;
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    return;
  }
}

int main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
Offline text-to-speech with sherpa-mnn.

It plays the generated audio as the model is processing.

Usage example:

wget https://github.com/k2-fsa/sherpa-mnn/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2

./bin/sherpa-mnn-offline-tts-play \
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
  std::string output_filename = "./generated.wav";
  int32_t sid = 0;

  po.Register("output-filename", &output_filename,
              "Path to save the generated audio");

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

  sherpa_mnn::Microphone mic;

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "Num devices: %d\n", num_devices);

  PaStreamParameters param;

  param.device = Pa_GetDefaultOutputDevice();
  if (param.device == paNoDevice) {
    fprintf(stderr, "No default output device found\n");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Use default device: %d\n", param.device);

  const PaDeviceInfo *info = Pa_GetDeviceInfo(param.device);
  fprintf(stderr, "  Name: %s\n", info->name);
  fprintf(stderr, "  Max output channels: %d\n", info->maxOutputChannels);

  if (config.max_num_sentences != 1) {
    fprintf(stderr, "Setting config.max_num_sentences to 1\n");
    config.max_num_sentences = 1;
  }

  fprintf(stderr, "Loading the model\n");
  sherpa_mnn::OfflineTts tts(config);

  fprintf(stderr, "Start the playback thread\n");
  std::thread playback_thread(StartPlayback, tts.SampleRate());

  float speed = 1.0;

  fprintf(stderr, "Generating ...\n");
  const auto begin = std::chrono::steady_clock::now();
  auto audio = tts.Generate(po.GetArg(1), sid, speed, AudioGeneratedCallback);
  const auto end = std::chrono::steady_clock::now();
  g_stopped = true;
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
