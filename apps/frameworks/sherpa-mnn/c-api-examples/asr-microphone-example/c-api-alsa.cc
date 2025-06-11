// c-api-examples/asr-microphone-example/c-api-alsa.cc
// Copyright (c)  2022-2024  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cctype>  // std::tolower
#include <cstdint>
#include <string>

#include "c-api-examples/asr-microphone-example/alsa.h"

// NOTE: You don't need to use cargs.h in your own project.
// We use it in this file to parse commandline arguments
#include "cargs.h"  // NOLINT
#include "sherpa-mnn/c-api/c-api.h"

static struct cag_option options[] = {
    {/*.identifier =*/'h',
     /*.access_letters =*/"h",
     /*.access_name =*/"help",
     /*.value_name =*/"help",
     /*.description =*/"Show help"},
    {/*.identifier =*/'t',
     /*.access_letters =*/NULL,
     /*.access_name =*/"tokens",
     /*.value_name =*/"tokens",
     /*.description =*/"Tokens file"},
    {/*.identifier =*/'e',
     /*.access_letters =*/NULL,
     /*.access_name =*/"encoder",
     /*.value_name =*/"encoder",
     /*.description =*/"Encoder ONNX file"},
    {/*.identifier =*/'d',
     /*.access_letters =*/NULL,
     /*.access_name =*/"decoder",
     /*.value_name =*/"decoder",
     /*.description =*/"Decoder ONNX file"},
    {/*.identifier =*/'j',
     /*.access_letters =*/NULL,
     /*.access_name =*/"joiner",
     /*.value_name =*/"joiner",
     /*.description =*/"Joiner ONNX file"},
    {/*.identifier =*/'n',
     /*.access_letters =*/NULL,
     /*.access_name =*/"num-threads",
     /*.value_name =*/"num-threads",
     /*.description =*/"Number of threads"},
    {/*.identifier =*/'p',
     /*.access_letters =*/NULL,
     /*.access_name =*/"provider",
     /*.value_name =*/"provider",
     /*.description =*/"Provider: cpu (default), cuda, coreml"},
    {/*.identifier =*/'m',
     /*.access_letters =*/NULL,
     /*.access_name =*/"decoding-method",
     /*.value_name =*/"decoding-method",
     /*.description =*/
     "Decoding method: greedy_search (default), modified_beam_search"},
    {/*.identifier =*/'f',
     /*.access_letters =*/NULL,
     /*.access_name =*/"hotwords-file",
     /*.value_name =*/"hotwords-file",
     /*.description =*/
     "The file containing hotwords, one words/phrases per line, and for each "
     "phrase the bpe/cjkchar are separated by a space. For example: ▁HE LL O "
     "▁WORLD, 你 好 世 界"},
    {/*.identifier =*/'s',
     /*.access_letters =*/NULL,
     /*.access_name =*/"hotwords-score",
     /*.value_name =*/"hotwords-score",
     /*.description =*/
     "The bonus score for each token in hotwords. Used only when "
     "decoding_method is modified_beam_search"},
};

const char *kUsage =
    R"(
Usage:
  ./bin/c-api-alsa \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/decoder.onnx \
    device_name

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
)";

bool stop = false;

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  if (argc < 6) {
    fprintf(stderr, "%s\n", kUsage);
    exit(0);
  }

  signal(SIGINT, Handler);

  SherpaMnnOnlineRecognizerConfig config;
  memset(&config, 0, sizeof(config));

  config.model_config.debug = 0;
  config.model_config.num_threads = 1;
  config.model_config.provider = "cpu";

  config.decoding_method = "greedy_search";

  config.max_active_paths = 4;

  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;

  config.enable_endpoint = 1;
  config.rule1_min_trailing_silence = 2.4;
  config.rule2_min_trailing_silence = 1.2;
  config.rule3_min_utterance_length = 300;

  cag_option_context context;
  char identifier;
  const char *value;

  cag_option_prepare(&context, options, CAG_ARRAY_SIZE(options), argc, argv);

  while (cag_option_fetch(&context)) {
    identifier = cag_option_get(&context);
    value = cag_option_get_value(&context);
    switch (identifier) {
      case 't':
        config.model_config.tokens = value;
        break;
      case 'e':
        config.model_config.transducer.encoder = value;
        break;
      case 'd':
        config.model_config.transducer.decoder = value;
        break;
      case 'j':
        config.model_config.transducer.joiner = value;
        break;
      case 'n':
        config.model_config.num_threads = atoi(value);
        break;
      case 'p':
        config.model_config.provider = value;
        break;
      case 'm':
        config.decoding_method = value;
        break;
      case 'f':
        config.hotwords_file = value;
        break;
      case 's':
        config.hotwords_score = atof(value);
        break;
      case 'h': {
        fprintf(stderr, "%s\n", kUsage);
        exit(0);
        break;
      }
      default:
        // do nothing as config already has valid default values
        break;
    }
  }

  const SherpaMnnOnlineRecognizer *recognizer =
      SherpaMnnCreateOnlineRecognizer(&config);
  const SherpaMnnOnlineStream *stream =
      SherpaMnnCreateOnlineStream(recognizer);

  const SherpaMnnDisplay *display = SherpaMnnCreateDisplay(50);
  int32_t segment_id = 0;

  const char *device_name = argv[context.index];
  sherpa_mnn::Alsa alsa(device_name);
  fprintf(stderr, "Use recording device: %s\n", device_name);
  fprintf(stderr,
          "Please \033[32m\033[1mspeak\033[0m! Press \033[31m\033[1mCtrl + "
          "C\033[0m to exit\n");

  int32_t expected_sample_rate = 16000;

  if (alsa.GetExpectedSampleRate() != expected_sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            expected_sample_rate);
    exit(-1);
  }

  int32_t chunk = 0.1 * alsa.GetActualSampleRate();

  std::string last_text;

  int32_t segment_index = 0;

  while (!stop) {
    const std::vector<float> &samples = alsa.Read(chunk);
    SherpaMnnOnlineStreamAcceptWaveform(stream, expected_sample_rate,
                                         samples.data(), samples.size());
    while (SherpaMnnIsOnlineStreamReady(recognizer, stream)) {
      SherpaMnnDecodeOnlineStream(recognizer, stream);
    }

    const SherpaMnnOnlineRecognizerResult *r =
        SherpaMnnGetOnlineStreamResult(recognizer, stream);

    std::string text = r->text;
    SherpaMnnDestroyOnlineRecognizerResult(r);

    if (!text.empty() && last_text != text) {
      last_text = text;

      std::transform(text.begin(), text.end(), text.begin(),
                     [](auto c) { return std::tolower(c); });

      SherpaMnnPrint(display, segment_index, text.c_str());
      fflush(stderr);
    }

    if (SherpaMnnOnlineStreamIsEndpoint(recognizer, stream)) {
      if (!text.empty()) {
        ++segment_index;
      }
      SherpaMnnOnlineStreamReset(recognizer, stream);
    }
  }

  // free allocated resources
  SherpaMnnDestroyDisplay(display);
  SherpaMnnDestroyOnlineStream(stream);
  SherpaMnnDestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
