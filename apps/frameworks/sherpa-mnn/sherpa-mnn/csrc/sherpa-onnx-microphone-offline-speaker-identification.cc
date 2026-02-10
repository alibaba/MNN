// sherpa-mnn/csrc/sherpa-mnn-microphone-offline-speaker-identification.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <mutex>  // NOLINT
#include <sstream>
#include <thread>  // NOLINT

#include "portaudio.h"  // NOLINT
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/microphone.h"
#include "sherpa-mnn/csrc/speaker-embedding-extractor.h"
#include "sherpa-mnn/csrc/speaker-embedding-manager.h"
#include "sherpa-mnn/csrc/wave-reader.h"

enum class State {
  kIdle,
  kRecording,
  kComputing,
};

State state = State::kIdle;

// true to stop the program and exit
bool stop = false;

std::vector<float> samples;
std::mutex samples_mutex;

static void DetectKeyPress() {
  SHERPA_ONNX_LOGE("\nPress Enter to start");
  int32_t key;
  while (!stop && (key = getchar())) {
    if (key != 0x0a) {
      continue;
    }

    switch (state) {
      case State::kIdle:
        SHERPA_ONNX_LOGE("\nStart recording. Press Enter to stop recording");
        state = State::kRecording;
        {
          std::lock_guard<std::mutex> lock(samples_mutex);
          samples.clear();
        }
        break;
      case State::kRecording:
        SHERPA_ONNX_LOGE("\nStop recording. Computing ...");
        state = State::kComputing;
        break;
      case State::kComputing:
        break;
    }
  }
}

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void *user_data) {
  std::lock_guard<std::mutex> lock(samples_mutex);

  auto p = reinterpret_cast<const float *>(input_buffer);
  samples.insert(samples.end(), p, p + frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Press Enter to exit\n");
}

static std::vector<std::vector<float>> ComputeEmbeddings(
    const std::vector<std::string> &filenames,
    sherpa_mnn::SpeakerEmbeddingExtractor *extractor) {
  std::vector<std::vector<float>> embedding_list;
  embedding_list.reserve(filenames.size());

  for (const auto &f : filenames) {
    int32_t sampling_rate = -1;

    bool is_ok = false;
    const std::vector<float> samples =
        sherpa_mnn::ReadWave(f, &sampling_rate, &is_ok);

    if (!is_ok) {
      fprintf(stderr, "Failed to read '%s'\n", f.c_str());
      exit(-1);
    }

    auto s = extractor->CreateStream();
    s->AcceptWaveform(sampling_rate, samples.data(), samples.size());
    s->InputFinished();
    auto embedding = extractor->Compute(s.get());
    embedding_list.push_back(embedding);
  }
  return embedding_list;
}

static std::unordered_map<std::string, std::vector<std::string>>
ReadSpeakerFile(const std::string &filename) {
  std::unordered_map<std::string, std::vector<std::string>> ans;

  std::ifstream is(filename);
  if (!is) {
    fprintf(stderr, "Failed to open %s", filename.c_str());
    exit(0);
  }

  std::string line;
  std::string name;
  std::string path;

  while (std::getline(is, line)) {
    std::istringstream iss(line);
    name.clear();
    path.clear();

    iss >> name >> path;
    if (!iss || !iss.eof() || name.empty() || path.empty()) {
      fprintf(stderr, "Invalid line: %s\n", line.c_str());
      exit(-1);
    }
    ans[name].push_back(path);
  }

  return ans;
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program shows how to use non-streaming speaker identification.
Usage:

(1) Prepare a text file containing speaker related files.

Each line in the text file contains two columns. The first column is the
speaker name, while the second column contains the wave file of the speaker.

If the text file contains multiple wave files for the same speaker, then the
embeddings of these files are averaged.

An example text file is given below:

    foo /path/to/a.wav
    bar /path/to/b.wav
    foo /path/to/c.wav
    foobar /path/to/d.wav

Each wave file should contain only a single channel; the sample format
should be int16_t; the sample rate can be arbitrary.

(2) Download a model for computing speaker embeddings

Please visit
https://github.com/k2-fsa/sherpa-mnn/releases/tag/speaker-recongition-models
to download a model. An example is given below:

    wget https://github.com/k2-fsa/sherpa-mnn/releases/download/speaker-recongition-models/wespeaker_zh_cnceleb_resnet34.onnx

Note that `zh` means Chinese, while `en` means English.

(3) Run it !

  ./bin/sherpa-mnn-microphone-offline-speaker-identification \
    --model=/path/to/your-model.onnx \
    --speaker-file=/path/to/speaker.txt
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  float threshold = 0.5;
  std::string speaker_file;

  po.Register("threshold", &threshold,
              "Threshold for comparing embedding scores.");

  po.Register("speaker-file", &speaker_file, "Path to speaker.txt");

  sherpa_mnn::SpeakerEmbeddingExtractorConfig config;
  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 0) {
    fprintf(stderr,
            "This program does not support any positional arguments.\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config! Please use --help to view the usage.\n");
    return -1;
  }

  SHERPA_ONNX_LOGE("\nCreating extractor ...");
  sherpa_mnn::SpeakerEmbeddingExtractor extractor(config);
  SHERPA_ONNX_LOGE("\nextractor created!");

  sherpa_mnn::SpeakerEmbeddingManager manager(extractor.Dim());

  auto name2files = ReadSpeakerFile(speaker_file);
  for (const auto &p : name2files) {
    SHERPA_ONNX_LOGE("\nProcessing speaker %s", p.first.c_str());
    auto embedding_list = ComputeEmbeddings(p.second, &extractor);
    manager.Add(p.first, embedding_list);
  }

  sherpa_mnn::Microphone mic;

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "Num devices: %d\n", num_devices);

  int32_t device_index = Pa_GetDefaultInputDevice();
  if (device_index == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    fprintf(stderr, "If you are using Linux, please switch to \n");
    fprintf(stderr,
            " ./bin/sherpa-mnn-alsa-offline-speaker-identification \n");
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
      case State::kComputing: {
        std::vector<float> buf;
        {
          std::lock_guard<std::mutex> lock(samples_mutex);
          buf = std::move(samples);
        }

        auto s = extractor.CreateStream();
        s->AcceptWaveform(mic_sample_rate, buf.data(), buf.size());
        s->InputFinished();
        auto embedding = extractor.Compute(s.get());
        auto name = manager.Search(embedding.data(), threshold);

        if (name.empty()) {
          name = "--Unknown--";
        }

        SHERPA_ONNX_LOGE("\nDone!\nDetected speaker is: %s", name.c_str());

        state = State::kIdle;
        SHERPA_ONNX_LOGE("\nPress Enter to start");
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
