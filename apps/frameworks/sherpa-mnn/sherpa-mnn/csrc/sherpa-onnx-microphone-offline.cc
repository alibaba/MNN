// sherpa-mnn/csrc/sherpa-mnn-microphone-offline.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cctype>  // std::tolower
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "portaudio.h"  // NOLINT
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/microphone.h"
#include "sherpa-mnn/csrc/offline-recognizer.h"

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

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void * /*user_data*/) {
  std::lock_guard<std::mutex> lock(samples_mutex);

  auto p = reinterpret_cast<const float *>(input_buffer);
  samples.insert(samples.end(), p, p + frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t /*sig*/) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Press Enter to exit\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program uses non-streaming models with microphone for speech recognition.
Usage:

(1) Transducer from icefall

  ./bin/sherpa-mnn-microphone-offline \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search

(2) Paraformer from FunASR

  ./bin/sherpa-mnn-microphone-offline \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    --num-threads=1

(3) Whisper models

  ./bin/sherpa-mnn-microphone-offline \
    --whisper-encoder=./sherpa-mnn-whisper-base.en/base.en-encoder.int8.onnx \
    --whisper-decoder=./sherpa-mnn-whisper-base.en/base.en-decoder.int8.onnx \
    --tokens=./sherpa-mnn-whisper-base.en/base.en-tokens.txt \
    --num-threads=1

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::OfflineRecognizerConfig config;
  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 0) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  SHERPA_ONNX_LOGE("Creating recognizer ...");
  sherpa_mnn::OfflineRecognizer recognizer(config);
  SHERPA_ONNX_LOGE("Recognizer created!");

  sherpa_mnn::Microphone mic;

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "Num devices: %d\n", num_devices);

  int32_t device_index = Pa_GetDefaultInputDevice();

  if (device_index == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    fprintf(stderr, "If you are using Linux, please switch to \n");
    fprintf(stderr, " ./bin/sherpa-mnn-alsa-offline \n");
    exit(EXIT_FAILURE);
  }

  const char *pDeviceIndex = std::getenv("SHERPA_ONNX_MIC_DEVICE");
  if (pDeviceIndex) {
    fprintf(stderr, "Use specified device: %s\n", pDeviceIndex);
    device_index = atoi(pDeviceIndex);
  }

  for (int32_t i = 0; i != num_devices; ++i) {
    const PaDeviceInfo *info = Pa_GetDeviceInfo(i);
    fprintf(stderr, " %s %d %s\n", (i == device_index) ? "*" : " ", i,
            info->name);
  }

  PaStreamParameters param;
  param.device = device_index;

  fprintf(stderr, "Use device: %d\n", param.device);

  const PaDeviceInfo *info = Pa_GetDeviceInfo(param.device);
  fprintf(stderr, "  Name: %s\n", info->name);
  fprintf(stderr, "  Max input channels: %d\n", info->maxInputChannels);

  param.channelCount = 1;
  param.sampleFormat = paFloat32;

  param.suggestedLatency = info->defaultLowInputLatency;
  param.hostApiSpecificStreamInfo = nullptr;
  float mic_sample_rate = 16000;
  const char *pSampleRateStr = std::getenv("SHERPA_ONNX_MIC_SAMPLE_RATE");
  if (pSampleRateStr) {
    fprintf(stderr, "Use sample rate %f for mic\n", mic_sample_rate);
    mic_sample_rate = atof(pSampleRateStr);
  }

  PaStream *stream;
  PaError err =
      Pa_OpenStream(&stream, &param, nullptr, /* &outputParameters, */
                    mic_sample_rate,
                    0,          // frames per buffer
                    paClipOff,  // we won't output out of range samples
                                // so don't bother clipping them
                    RecordCallback, nullptr);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  err = Pa_StartStream(stream);
  fprintf(stderr, "Started\n");

  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

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

        auto s = recognizer.CreateStream();
        s->AcceptWaveform(mic_sample_rate, buf.data(), buf.size());
        recognizer.DecodeStream(s.get());
        SHERPA_ONNX_LOGE("Decoding Done! Result is:");
        SHERPA_ONNX_LOGE("%s", s->GetResult().text.c_str());

        state = State::kIdle;
        SHERPA_ONNX_LOGE("Press Enter to start");
        break;
      }
    }

    Pa_Sleep(20);  // sleep for 20ms
  }
  t.join();

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  return 0;
}
