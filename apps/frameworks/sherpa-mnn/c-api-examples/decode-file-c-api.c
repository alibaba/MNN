// c-api-examples/decode-file-c-api.c
//
// Copyright (c)  2023  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// to decode a file.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cargs.h"
#include "sherpa-mnn/c-api/c-api.h"

static struct cag_option options[] = {
    {.identifier = 'h',
     .access_letters = "h",
     .access_name = "help",
     .description = "Show help"},
    {.identifier = 't',
     .access_letters = NULL,
     .access_name = "tokens",
     .value_name = "tokens",
     .description = "Tokens file"},
    {.identifier = 'e',
     .access_letters = NULL,
     .access_name = "encoder",
     .value_name = "encoder",
     .description = "Encoder ONNX file"},
    {.identifier = 'd',
     .access_letters = NULL,
     .access_name = "decoder",
     .value_name = "decoder",
     .description = "Decoder ONNX file"},
    {.identifier = 'j',
     .access_letters = NULL,
     .access_name = "joiner",
     .value_name = "joiner",
     .description = "Joiner ONNX file"},
    {.identifier = 'n',
     .access_letters = NULL,
     .access_name = "num-threads",
     .value_name = "num-threads",
     .description = "Number of threads"},
    {.identifier = 'p',
     .access_letters = NULL,
     .access_name = "provider",
     .value_name = "provider",
     .description = "Provider: cpu (default), cuda, coreml"},
    {.identifier = 'm',
     .access_letters = NULL,
     .access_name = "decoding-method",
     .value_name = "decoding-method",
     .description =
         "Decoding method: greedy_search (default), modified_beam_search"},
    {.identifier = 'f',
     .access_letters = NULL,
     .access_name = "hotwords-file",
     .value_name = "hotwords-file",
     .description = "The file containing hotwords, one words/phrases per line, "
                    "and for each phrase the bpe/cjkchar are separated by a "
                    "space. For example: ▁HE LL O ▁WORLD, 你 好 世 界"},
    {.identifier = 's',
     .access_letters = NULL,
     .access_name = "hotwords-score",
     .value_name = "hotwords-score",
     .description = "The bonus score for each token in hotwords. Used only "
                    "when decoding_method is modified_beam_search"},
};

const char *kUsage =
    "\n"
    "Usage:\n "
    "  ./bin/decode-file-c-api \\\n"
    "    --tokens=/path/to/tokens.txt \\\n"
    "    --encoder=/path/to/encoder.onnx \\\n"
    "    --decoder=/path/to/decoder.onnx \\\n"
    "    --joiner=/path/to/joiner.onnx \\\n"
    "    --provider=cpu \\\n"
    "    /path/to/foo.wav\n"
    "\n\n"
    "Default num_threads is 1.\n"
    "Valid decoding_method: greedy_search (default), modified_beam_search\n\n"
    "Valid provider: cpu (default), cuda, coreml\n\n"
    "Please refer to \n"
    "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/"
    "index.html\n"
    "for a list of pre-trained models to download.\n"
    "\n"
    "Note that this file supports only streaming transducer models.\n";

int32_t main(int32_t argc, char *argv[]) {
  if (argc < 6) {
    fprintf(stderr, "%s\n", kUsage);
    exit(0);
  }

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

  const char *wav_filename = argv[context.index];
  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }
  // simulate streaming

#define N 3200  // 0.2 s. Sample rate is fixed to 16 kHz

  fprintf(stderr, "sample rate: %d, num samples: %d, duration: %.2f s\n",
          wave->sample_rate, wave->num_samples,
          (float)wave->num_samples / wave->sample_rate);

  int32_t k = 0;
  while (k < wave->num_samples) {
    int32_t start = k;
    int32_t end =
        (start + N > wave->num_samples) ? wave->num_samples : (start + N);
    k += N;

    SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate,
                                         wave->samples + start, end - start);
    while (SherpaMnnIsOnlineStreamReady(recognizer, stream)) {
      SherpaMnnDecodeOnlineStream(recognizer, stream);
    }

    const SherpaMnnOnlineRecognizerResult *r =
        SherpaMnnGetOnlineStreamResult(recognizer, stream);

    if (strlen(r->text)) {
      SherpaMnnPrint(display, segment_id, r->text);
    }

    if (SherpaMnnOnlineStreamIsEndpoint(recognizer, stream)) {
      if (strlen(r->text)) {
        ++segment_id;
      }
      SherpaMnnOnlineStreamReset(recognizer, stream);
    }

    SherpaMnnDestroyOnlineRecognizerResult(r);
  }

  // add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail_paddings,
                                       4800);

  SherpaMnnFreeWave(wave);

  SherpaMnnOnlineStreamInputFinished(stream);
  while (SherpaMnnIsOnlineStreamReady(recognizer, stream)) {
    SherpaMnnDecodeOnlineStream(recognizer, stream);
  }

  const SherpaMnnOnlineRecognizerResult *r =
      SherpaMnnGetOnlineStreamResult(recognizer, stream);

  if (strlen(r->text)) {
    SherpaMnnPrint(display, segment_id, r->text);
  }

  SherpaMnnDestroyOnlineRecognizerResult(r);

  SherpaMnnDestroyDisplay(display);
  SherpaMnnDestroyOnlineStream(stream);
  SherpaMnnDestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
