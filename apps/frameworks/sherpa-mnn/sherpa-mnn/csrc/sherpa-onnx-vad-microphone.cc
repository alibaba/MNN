// sherpa-mnn/csrc/sherpa-mnn-vad-microphone.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <mutex>  // NOLINT

#include "portaudio.h"  // NOLINT
#include "sherpa-mnn/csrc/circular-buffer.h"
#include "sherpa-mnn/csrc/microphone.h"
#include "sherpa-mnn/csrc/resample.h"
#include "sherpa-mnn/csrc/voice-activity-detector.h"
#include "sherpa-mnn/csrc/wave-writer.h"

bool stop = false;
std::mutex mutex;
sherpa_mnn::CircularBuffer buffer(16000 * 60);

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void * /*user_data*/) {
  std::lock_guard<std::mutex> lock(mutex);
  buffer.Push(reinterpret_cast<const float *>(input_buffer), frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t /*sig*/) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program shows how to use VAD in sherpa-mnn.

  ./bin/sherpa-mnn-vad-microphone \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --vad-provider=cpu \
    --vad-num-threads=1

Please download silero_vad.onnx from
https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

For instance, use
wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::VadModelConfig config;

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

  sherpa_mnn::Microphone mic;

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "Num devices: %d\n", num_devices);

  int32_t device_index = Pa_GetDefaultInputDevice();

  if (device_index == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    fprintf(stderr, "If you are using Linux, please switch to \n");
    fprintf(stderr, " ./bin/sherpa-mnn-vad-alsa \n");
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
  float sample_rate = 16000;

  std::unique_ptr<sherpa_mnn::LinearResample> resampler;
  if (mic_sample_rate != sample_rate) {
    float min_freq = std::min(mic_sample_rate, sample_rate);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler = std::make_unique<sherpa_mnn::LinearResample>(
        mic_sample_rate, sample_rate, lowpass_cutoff, lowpass_filter_width);
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

  auto vad = std::make_unique<sherpa_mnn::VoiceActivityDetector>(config);

  fprintf(stderr, "Started\n");

  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  int32_t window_size = config.silero_vad.window_size;
  bool printed = false;

  int32_t k = 0;
  while (!stop) {
    {
      std::lock_guard<std::mutex> lock(mutex);

      while (buffer.Size() >= window_size) {
        std::vector<float> samples = buffer.Get(buffer.Head(), window_size);
        buffer.Pop(window_size);

        if (resampler) {
          std::vector<float> tmp;
          resampler->Resample(samples.data(), samples.size(), true, &tmp);
          samples = std::move(tmp);
        }

        vad->AcceptWaveform(samples.data(), samples.size());

        if (vad->IsSpeechDetected() && !printed) {
          printed = true;
          fprintf(stderr, "\nDetected speech!\n");
        }
        if (!vad->IsSpeechDetected()) {
          printed = false;
        }

        while (!vad->Empty()) {
          const auto &segment = vad->Front();
          float duration = segment.samples.size() / sample_rate;
          fprintf(stderr, "Duration: %.3f seconds\n", duration);

          char filename[128];
          snprintf(filename, sizeof(filename), "seg-%d-%.3fs.wav", k, duration);
          k += 1;
          sherpa_mnn::WriteWave(filename, sample_rate, segment.samples.data(),
                                 segment.samples.size());
          fprintf(stderr, "Saved to %s\n", filename);
          fprintf(stderr, "----------\n");

          vad->Pop();
        }
      }
    }
    Pa_Sleep(100);  // sleep for 100ms
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  return 0;
}
