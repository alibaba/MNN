// c-api-examples/add-punctuation-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation

// We assume you have pre-downloaded the model files for testing
// from https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models
//
// An example is given below:
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
// tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
// rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-mnn/c-api/c-api.h"

int32_t main() {
  SherpaMnnOfflinePunctuationConfig config;
  memset(&config, 0, sizeof(config));

  // clang-format off
  config.model.ct_transformer = "./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx";
  // clang-format on
  config.model.num_threads = 1;
  config.model.debug = 1;
  config.model.provider = "cpu";

  const SherpaMnnOfflinePunctuation *punct =
      SherpaMnnCreateOfflinePunctuation(&config);
  if (!punct) {
    fprintf(stderr,
            "Failed to create OfflinePunctuation. Please check your config");
    return -1;
  }

  const char *texts[] = {
      "这是一个测试你好吗How are you我很好thank you are you ok谢谢你",
      "我们都是木头人不会说话不会动",
      ("The African blogosphere is rapidly expanding bringing more voices "
       "online in the form of commentaries opinions analyses rants and poetry"),
  };

  int32_t n = sizeof(texts) / sizeof(const char *);
  fprintf(stderr, "n: %d\n", n);

  fprintf(stderr, "--------------------\n");
  for (int32_t i = 0; i != n; ++i) {
    const char *text_with_punct =
        SherpaOfflinePunctuationAddPunct(punct, texts[i]);

    fprintf(stderr, "Input text: %s\n", texts[i]);
    fprintf(stderr, "Output text: %s\n", text_with_punct);
    SherpaOfflinePunctuationFreeText(text_with_punct);
    fprintf(stderr, "--------------------\n");
  }

  SherpaMnnDestroyOfflinePunctuation(punct);

  return 0;
};
