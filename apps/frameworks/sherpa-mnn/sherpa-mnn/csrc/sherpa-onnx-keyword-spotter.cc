// sherpa-mnn/csrc/sherpa-mnn-keyword-spotter.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include <stdio.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/keyword-spotter.h"
#include "sherpa-mnn/csrc/online-stream.h"
#include "sherpa-mnn/csrc/parse-options.h"
#include "sherpa-mnn/csrc/wave-reader.h"

typedef struct {
  std::unique_ptr<sherpa_mnn::OnlineStream> online_stream;
  std::string filename;
} Stream;

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Usage:

(1) Streaming transducer

  ./bin/sherpa-mnn-keyword-spotter \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --provider=cpu \
    --num-threads=2 \
    --keywords-file=keywords.txt \
    /path/to/foo.wav [bar.wav foobar.wav ...]

Note: It supports decoding multiple files in batches

Default value for num_threads is 2.
Valid values for provider: cpu (default), cuda, coreml.
foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::KeywordSpotterConfig config;

  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  sherpa_mnn::KeywordSpotter keyword_spotter(config);

  std::vector<Stream> ss;

  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    const std::string wav_filename = po.GetArg(i);
    int32_t sampling_rate = -1;

    bool is_ok = false;
    const std::vector<float> samples =
        sherpa_mnn::ReadWave(wav_filename, &sampling_rate, &is_ok);

    if (!is_ok) {
      fprintf(stderr, "Failed to read '%s'\n", wav_filename.c_str());
      return -1;
    }

    auto s = keyword_spotter.CreateStream();
    s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

    std::vector<float> tail_paddings(static_cast<int>(0.8 * sampling_rate));
    // Note: We can call AcceptWaveform() multiple times.
    s->AcceptWaveform(sampling_rate, tail_paddings.data(),
                      tail_paddings.size());

    // Call InputFinished() to indicate that no audio samples are available
    s->InputFinished();
    ss.push_back({std::move(s), wav_filename});
  }

  std::vector<sherpa_mnn::OnlineStream *> ready_streams;
  for (;;) {
    ready_streams.clear();
    for (auto &s : ss) {
      const auto p_ss = s.online_stream.get();
      if (keyword_spotter.IsReady(p_ss)) {
        ready_streams.push_back(p_ss);
      }
      std::ostringstream os;
      const auto r = keyword_spotter.GetResult(p_ss);
      if (!r.keyword.empty()) {
        os << s.filename << "\n";
        os << r.AsJsonString() << "\n\n";
        fprintf(stderr, "%s", os.str().c_str());
      }
    }

    if (ready_streams.empty()) {
      break;
    }
    keyword_spotter.DecodeStreams(ready_streams.data(), ready_streams.size());
  }
  return 0;
}
