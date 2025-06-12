// sherpa-mnn/csrc/sherpa-mnn-keyword-spotter-microphone.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

#include "portaudio.h"  // NOLINT
#include "sherpa-mnn/csrc/display.h"
#include "sherpa-mnn/csrc/keyword-spotter.h"
#include "sherpa-mnn/csrc/microphone.h"

bool stop = false;
float mic_sample_rate = 16000;

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void *user_data) {
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(user_data);

  stream->AcceptWaveform(mic_sample_rate,
                         reinterpret_cast<const float *>(input_buffer),
                         frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t /*sig*/) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program uses streaming models with microphone for keyword spotting.
Usage:

  ./bin/sherpa-mnn-keyword-spotter-microphone \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --provider=cpu \
    --num-threads=1 \
    --keywords-file=keywords.txt

Please refer to
https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::KeywordSpotterConfig config;

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

  sherpa_mnn::KeywordSpotter spotter(config);
  auto s = spotter.CreateStream();

  sherpa_mnn::Microphone mic;

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "Num devices: %d\n", num_devices);

  int32_t device_index = Pa_GetDefaultInputDevice();

  if (device_index == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    fprintf(stderr, "If you are using Linux, please switch to \n");
    fprintf(stderr, " ./bin/sherpa-mnn-keyword-spotter-alsa \n");
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
                    RecordCallback, s.get());
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

  int32_t keyword_index = 0;
  sherpa_mnn::Display display;
  while (!stop) {
    while (spotter.IsReady(s.get())) {
      spotter.DecodeStream(s.get());

      const auto r = spotter.GetResult(s.get());
      if (!r.keyword.empty()) {
        display.Print(keyword_index, r.AsJsonString());
        fflush(stderr);
        keyword_index++;

        spotter.Reset(s.get());
      }
    }

    Pa_Sleep(20);  // sleep for 20ms
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  return 0;
}
