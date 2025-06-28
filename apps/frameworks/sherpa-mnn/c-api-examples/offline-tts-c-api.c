// c-api-examples/offline-tts-c-api.c
//
// Copyright (c)  2023  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// to convert text to speech using an offline model.

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
    {.access_name = "vits-model",
     .value_name = "/path/to/xxx.onnx",
     .identifier = '0',
     .description = "Path to VITS model"},
    {.access_name = "vits-lexicon",
     .value_name = "/path/to/lexicon.txt",
     .identifier = '1',
     .description = "Path to lexicon.txt for VITS models"},
    {.access_name = "vits-tokens",
     .value_name = "/path/to/tokens.txt",
     .identifier = '2',
     .description = "Path to tokens.txt for VITS models"},
    {.access_name = "vits-noise-scale",
     .value_name = "0.667",
     .identifier = '3',
     .description = "noise_scale for VITS models"},
    {.access_name = "vits-noise-scale-w",
     .value_name = "0.8",
     .identifier = '4',
     .description = "noise_scale_w for VITS models"},
    {.access_name = "vits-length-scale",
     .value_name = "1.0",
     .identifier = '5',
     .description =
         "length_scale for VITS models. Default to 1. You can tune it "
         "to change the speech speed. small -> faster; large -> slower. "},
    {.access_name = "num-threads",
     .value_name = "1",
     .identifier = '6',
     .description = "Number of threads"},
    {.access_name = "provider",
     .value_name = "cpu",
     .identifier = '7',
     .description = "Provider: cpu (default), cuda, coreml"},
    {.access_name = "debug",
     .value_name = "0",
     .identifier = '8',
     .description = "1 to show debug messages while loading the model"},
    {.access_name = "sid",
     .value_name = "0",
     .identifier = '9',
     .description = "Speaker ID. Default to 0. Note it is not used for "
                    "single-speaker models."},
    {.access_name = "output-filename",
     .value_name = "./generated.wav",
     .identifier = 'a',
     .description =
         "Filename to save the generated audio. Default to ./generated.wav"},

    {.access_name = "tts-rule-fsts",
     .value_name = "/path/to/rule.fst",
     .identifier = 'b',
     .description = "It not empty, it contains a list of rule FST filenames."
                    "Multiple filenames are separated by a comma and they are "
                    "applied from left to right. An example value: "
                    "rule1.fst,rule2,fst,rule3.fst"},

    {.access_name = "max-num-sentences",
     .value_name = "2",
     .identifier = 'c',
     .description = "Maximum number of sentences that we process at a time. "
                    "This is to avoid OOM for very long input text. "
                    "If you set it to -1, then we process all sentences in a "
                    "single batch."},

    {.access_name = "vits-data-dir",
     .value_name = "/path/to/espeak-ng-data",
     .identifier = 'd',
     .description =
         "Path to espeak-ng-data. If it is given, --vits-lexicon is ignored"},

};

static void ShowUsage() {
  const char *kUsageMessage =
      "Offline text-to-speech with sherpa-onnx C API"
      "\n"
      "./offline-tts-c-api \\\n"
      " --vits-model=/path/to/model.onnx \\\n"
      " --vits-lexicon=/path/to/lexicon.txt \\\n"
      " --vits-tokens=/path/to/tokens.txt \\\n"
      " --sid=0 \\\n"
      " --output-filename=./generated.wav \\\n"
      " 'some text within single quotes on linux/macos or use double quotes on "
      "windows'\n"
      "\n"
      "It will generate a file ./generated.wav as specified by "
      "--output-filename.\n"
      "\n"
      "You can download a test model from\n"
      "https://huggingface.co/csukuangfj/vits-ljs\n"
      "\n"
      "For instance, you can use:\n"
      "wget "
      "https://huggingface.co/csukuangfj/vits-ljs/resolve/main/vits-ljs.onnx\n"
      "wget "
      "https://huggingface.co/csukuangfj/vits-ljs/resolve/main/lexicon.txt\n"
      "wget "
      "https://huggingface.co/csukuangfj/vits-ljs/resolve/main/tokens.txt\n"
      "\n"
      "./offline-tts-c-api \\\n"
      "  --vits-model=./vits-ljs.onnx \\\n"
      "  --vits-lexicon=./lexicon.txt \\\n"
      "  --vits-tokens=./tokens.txt \\\n"
      "  --sid=0 \\\n"
      "  --output-filename=./generated.wav \\\n"
      "  'liliana, the most beautiful and lovely assistant of our team!'\n"
      "\n"
      "Please see\n"
      "https://k2-fsa.github.io/sherpa/onnx/tts/index.html\n"
      "or details.\n\n";

  fprintf(stderr, "%s", kUsageMessage);
  cag_option_print(options, CAG_ARRAY_SIZE(options), stderr);
  exit(0);
}

int32_t main(int32_t argc, char *argv[]) {
  cag_option_context context;
  char identifier;
  const char *value;

  cag_option_prepare(&context, options, CAG_ARRAY_SIZE(options), argc, argv);

  SherpaMnnOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));

  int32_t sid = 0;
  const char *filename = strdup("./generated.wav");
  const char *text;

  while (cag_option_fetch(&context)) {
    identifier = cag_option_get(&context);
    value = cag_option_get_value(&context);
    switch (identifier) {
      case '0':
        config.model.vits.model = value;
        break;
      case '1':
        config.model.vits.lexicon = value;
        break;
      case '2':
        config.model.vits.tokens = value;
        break;
      case '3':
        config.model.vits.noise_scale = atof(value);
        break;
      case '4':
        config.model.vits.noise_scale_w = atof(value);
        break;
      case '5':
        config.model.vits.length_scale = atof(value);
        break;
      case '6':
        config.model.num_threads = atoi(value);
        break;
      case '7':
        config.model.provider = value;
        break;
      case '8':
        config.model.debug = atoi(value);
        break;
      case '9':
        sid = atoi(value);
        break;
      case 'a':
        free((void *)filename);
        filename = strdup(value);
        break;
      case 'b':
        config.rule_fsts = value;
        break;
      case 'c':
        config.max_num_sentences = atoi(value);
        break;
      case 'd':
        config.model.vits.data_dir = value;
        break;
      case '?':
        fprintf(stderr, "Unknown option\n");
        // fall through
      case 'h':
        // fall through
      default:
        ShowUsage();
    }
  }
  fprintf(stderr, "here\n");

  if (!config.model.vits.model) {
    fprintf(stderr, "Please provide --vits-model\n");
    ShowUsage();
  }

  if (!config.model.vits.tokens) {
    fprintf(stderr, "Please provide --vits-tokens\n");
    ShowUsage();
  }

  if (!config.model.vits.data_dir && !config.model.vits.lexicon) {
    fprintf(stderr, "Please provide --vits-data-dir or --vits-lexicon\n");
    ShowUsage();
  }

  // the last arg is the text
  text = argv[argc - 1];
  if (text[0] == '-') {
    fprintf(stderr, "\n***Please input your text!***\n\n");
    fprintf(stderr, "\n---------------Usage---------------\n\n");
    ShowUsage();
  }

  const SherpaMnnOfflineTts *tts = SherpaMnnCreateOfflineTts(&config);

  const SherpaMnnGeneratedAudio *audio =
      SherpaMnnOfflineTtsGenerate(tts, text, sid, 1.0);

  SherpaMnnWriteWave(audio->samples, audio->n, audio->sample_rate, filename);

  SherpaMnnDestroyOfflineTtsGeneratedAudio(audio);
  SherpaMnnDestroyOfflineTts(tts);

  fprintf(stderr, "Input text is: %s\n", text);
  fprintf(stderr, "Speaker ID is is: %d\n", sid);
  fprintf(stderr, "Saved to: %s\n", filename);

  free((void *)filename);

  return 0;
}
