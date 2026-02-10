// sherpa-mnn/csrc/sherpa-mnn-keyword-spotter-alsa.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>

#include "sherpa-mnn/csrc/alsa.h"
#include "sherpa-mnn/csrc/display.h"
#include "sherpa-mnn/csrc/keyword-spotter.h"
#include "sherpa-mnn/csrc/parse-options.h"

bool stop = false;

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
Usage:
  ./bin/sherpa-mnn-keyword-spotter-alsa \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --provider=cpu \
    --num-threads=2 \
    --keywords-file=keywords.txt \
    device_name

Please refer to
https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
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
  sherpa_mnn::KeywordSpotterConfig config;

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
  sherpa_mnn::KeywordSpotter spotter(config);

  int32_t expected_sample_rate = config.feat_config.sampling_rate;

  std::string device_name = po.GetArg(1);
  sherpa_mnn::Alsa alsa(device_name.c_str());
  fprintf(stderr, "Use recording device: %s\n", device_name.c_str());

  if (alsa.GetExpectedSampleRate() != expected_sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            expected_sample_rate);
    exit(-1);
  }

  int32_t chunk = 0.1 * alsa.GetActualSampleRate();

  std::string last_text;

  auto stream = spotter.CreateStream();

  sherpa_mnn::Display display;

  int32_t keyword_index = 0;
  while (!stop) {
    const std::vector<float> &samples = alsa.Read(chunk);

    stream->AcceptWaveform(expected_sample_rate, samples.data(),
                           samples.size());

    while (spotter.IsReady(stream.get())) {
      spotter.DecodeStream(stream.get());

      const auto r = spotter.GetResult(stream.get());
      if (!r.keyword.empty()) {
        display.Print(keyword_index, r.AsJsonString());
        fflush(stderr);
        keyword_index++;

        spotter.Reset(stream.get());
      }
    }
  }

  return 0;
}
