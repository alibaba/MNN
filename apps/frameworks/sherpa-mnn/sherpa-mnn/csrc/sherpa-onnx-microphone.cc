// sherpa-mnn/csrc/sherpa-mnn-microphone.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <clocale>
#include <cwctype>

#include "portaudio.h"  // NOLINT
#include "sherpa-mnn/csrc/display.h"
#include "sherpa-mnn/csrc/microphone.h"
#include "sherpa-mnn/csrc/online-recognizer.h"

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

static std::string tolowerUnicode(const std::string &input_str) {
  // Use system locale
  std::setlocale(LC_ALL, "");

  // From char string to wchar string
  std::wstring input_wstr(input_str.size() + 1, '\0');
  std::mbstowcs(&input_wstr[0], input_str.c_str(), input_str.size());
  std::wstring lowercase_wstr;

  for (wchar_t wc : input_wstr) {
    if (std::iswupper(wc)) {
      lowercase_wstr += std::towlower(wc);
    } else {
      lowercase_wstr += wc;
    }
  }

  // Back to char string
  std::string lowercase_str(input_str.size() + 1, '\0');
  std::wcstombs(&lowercase_str[0], lowercase_wstr.c_str(),
                lowercase_wstr.size());

  return lowercase_str;
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program uses streaming models with microphone for speech recognition.
Usage:

  ./bin/sherpa-mnn-microphone \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --provider=cpu \
    --num-threads=1 \
    --decoding-method=greedy_search

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::OnlineRecognizerConfig config;

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

  sherpa_mnn::OnlineRecognizer recognizer(config);
  auto s = recognizer.CreateStream();

  sherpa_mnn::Microphone mic;

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "Num devices: %d\n", num_devices);

  int32_t device_index = Pa_GetDefaultInputDevice();

  if (device_index == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    fprintf(stderr, "If you are using Linux, please switch to \n");
    fprintf(stderr, " ./bin/sherpa-mnn-alsa \n");
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
  float sample_rate = 16000;

  PaStream *stream;
  PaError err =
      Pa_OpenStream(&stream, &param, nullptr, /* &outputParameters, */
                    sample_rate,
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

  std::string last_text;
  int32_t segment_index = 0;
  sherpa_mnn::Display display(30);
  while (!stop) {
    while (recognizer.IsReady(s.get())) {
      recognizer.DecodeStream(s.get());
    }

    auto text = recognizer.GetResult(s.get()).text;
    bool is_endpoint = recognizer.IsEndpoint(s.get());

    if (is_endpoint && !config.model_config.paraformer.encoder.empty()) {
      // For streaming paraformer models, since it has a large right chunk size
      // we need to pad it on endpointing so that the last character
      // can be recognized
      std::vector<float> tail_paddings(static_cast<int>(1.0 * mic_sample_rate));
      s->AcceptWaveform(mic_sample_rate, tail_paddings.data(),
                        tail_paddings.size());
      while (recognizer.IsReady(s.get())) {
        recognizer.DecodeStream(s.get());
      }
      text = recognizer.GetResult(s.get()).text;
    }

    if (!text.empty() && last_text != text) {
      last_text = text;
      display.Print(segment_index, tolowerUnicode(text));
      fflush(stderr);
    }

    if (is_endpoint) {
      if (!text.empty()) {
        ++segment_index;
      }

      recognizer.Reset(s.get());
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
