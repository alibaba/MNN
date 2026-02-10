// sherpa-mnn/csrc/sherpa-mnn-online-punctuation.cc
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#include <stdio.h>

#include <chrono>  // NOLINT
#include <iostream>

#include "sherpa-mnn/csrc/online-punctuation.h"
#include "sherpa-mnn/csrc/parse-options.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Add punctuations to the input text.

The input text can contain English words.

Usage:

Please download the model from:
https://github.com/k2-fsa/sherpa-mnn/releases/download/punctuation-models/sherpa-mnn-online-punct-en-2024-08-06.tar.bz2

./bin/Release/sherpa-mnn-online-punctuation \
  --cnn-bilstm=/path/to/model.onnx \
  --bpe-vocab=/path/to/bpe.vocab \
  "how are you i am fine thank you"

The output text should look like below:
  "How are you? I am fine. Thank you."
)usage";

  sherpa_mnn::ParseOptions po(kUsageMessage);
  sherpa_mnn::OnlinePunctuationConfig config;
  config.Register(&po);
  po.Read(argc, argv);
  if (po.NumArgs() != 1) {
    fprintf(stderr,
            "Error: Please provide only 1 positional argument containing the "
            "input text.\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  fprintf(stderr, "Creating OnlinePunctuation ...\n");
  sherpa_mnn::OnlinePunctuation punct(config);
  fprintf(stderr, "Started\n");
  const auto begin = std::chrono::steady_clock::now();

  std::string text = po.GetArg(1);

  std::string text_with_punct_case = punct.AddPunctuationWithCase(text);

  const auto end = std::chrono::steady_clock::now();
  fprintf(stderr, "Done\n");

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "Num threads: %d\n", config.model.num_threads);
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  fprintf(stderr, "Input text: %s\n", text.c_str());
  fprintf(stderr, "Output text: %s\n", text_with_punct_case.c_str());
}
