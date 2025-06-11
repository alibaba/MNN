// sherpa-mnn/csrc/sherpa-mnn-alsa-offline-audio-tagging.cc
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "sherpa-mnn/csrc/alsa.h"
#include "sherpa-mnn/csrc/audio-tagging.h"
#include "sherpa-mnn/csrc/macros.h"

enum class State {
  kIdle,
  kRecording,
  kDecoding,
};

State state = State::kIdle;

// true to stop the program and exit
bool stop = false;

std::vector<float> samples;
std::mutex samples_mutex;

static void DetectKeyPress() {
  SHERPA_ONNX_LOGE("Press Enter to start");
  int32_t key;
  while (!stop && (key = getchar())) {
    if (key != 0x0a) {
      continue;
    }

    switch (state) {
      case State::kIdle:
        SHERPA_ONNX_LOGE("Start recording. Press Enter to stop recording");
        state = State::kRecording;
        {
          std::lock_guard<std::mutex> lock(samples_mutex);
          samples.clear();
        }
        break;
      case State::kRecording:
        SHERPA_ONNX_LOGE("Stop recording. Decoding ...");
        state = State::kDecoding;
        break;
      case State::kDecoding:
        break;
    }
  }
}

static void Record(const char *device_name, int32_t expected_sample_rate) {
  sherpa_mnn::Alsa alsa(device_name);

  if (alsa.GetExpectedSampleRate() != expected_sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            expected_sample_rate);
    exit(-1);
  }

  int32_t chunk = 0.1 * alsa.GetActualSampleRate();
  while (!stop) {
    const std::vector<float> &s = alsa.Read(chunk);
    std::lock_guard<std::mutex> lock(samples_mutex);
    samples.insert(samples.end(), s.begin(), s.end());
  }
}

static void Handler(int32_t sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Press Enter to exit\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
Audio tagging from microphone (Linux only).
Usage:

wget https://github.com/k2-fsa/sherpa-mnn/releases/download/audio-tagging-models/sherpa-mnn-zipformer-audio-tagging-2024-04-09.tar.bz2
tar xvf sherpa-mnn-zipformer-audio-tagging-2024-04-09.tar.bz2
rm sherpa-mnn-zipformer-audio-tagging-2024-04-09.tar.bz2

./bin/sherpa-mnn-alsa-offline-audio-tagging \
  --zipformer-model=./sherpa-mnn-zipformer-audio-tagging-2024-04-09/model.onnx \
  --labels=./sherpa-mnn-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv \
    device_name

Please refer to
https://github.com/k2-fsa/sherpa-mnn/releases/tag/audio-tagging-models
for a list of pre-trained models to download.

The device name specifies which microphone to use in case there are several
on your system. You can use

  arecord -l

to find all available microphones on your computer. For instance, if it outputs

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and device 0 on that card, please use:

  plughw:3,0

as the device_name.
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::AudioTaggingConfig config;
  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 1) {
    fprintf(stderr, "Please provide only 1 argument: the device name\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  SHERPA_ONNX_LOGE("Creating audio tagger ...");
  sherpa_mnn::AudioTagging tagger(config);
  SHERPA_ONNX_LOGE("Audio tagger created created!");

  std::string device_name = po.GetArg(1);
  fprintf(stderr, "Use recording device: %s\n", device_name.c_str());

  int32_t sample_rate = 16000;  // fixed to 16000Hz for all models from icefall

  std::thread t2(Record, device_name.c_str(), sample_rate);
  using namespace std::chrono_literals;  // NOLINT
  std::this_thread::sleep_for(100ms);    // sleep for 100ms
  std::thread t(DetectKeyPress);

  while (!stop) {
    switch (state) {
      case State::kIdle:
        break;
      case State::kRecording:
        break;
      case State::kDecoding: {
        std::vector<float> buf;
        {
          std::lock_guard<std::mutex> lock(samples_mutex);
          buf = std::move(samples);
        }
        SHERPA_ONNX_LOGE("Computing...");
        auto s = tagger.CreateStream();
        s->AcceptWaveform(sample_rate, buf.data(), buf.size());
        auto results = tagger.Compute(s.get());
        SHERPA_ONNX_LOGE("Result is:");

        int32_t i = 0;
        std::ostringstream os;
        for (const auto &event : results) {
          os << i << ": " << event.ToString() << "\n";
          i += 1;
        }

        SHERPA_ONNX_LOGE("\n%s\n", os.str().c_str());

        state = State::kIdle;
        SHERPA_ONNX_LOGE("Press Enter to start");
        break;
      }
    }

    std::this_thread::sleep_for(20ms);  // sleep for 20ms
  }
  t.join();
  t2.join();

  return 0;
}
