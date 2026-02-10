// c-api-examples/speech-enhancement-gtcrn-c-api.c
//
// Copyright (c)  2025  Xiaomi Corporation
//
// We assume you have pre-downloaded model
// from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
//
// An example command to download
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
*/
// clang-format on
#include <stdio.h>
#include <string.h>

#include "sherpa-mnn/c-api/c-api.h"

int32_t main() {
  SherpaMnnOfflineSpeechDenoiserConfig config;
  const char *wav_filename = "./inp_16k.wav";
  const char *out_wave_filename = "./enhanced_16k.wav";

  memset(&config, 0, sizeof(config));
  config.model.gtcrn.model = "./gtcrn_simple.onnx";

  const SherpaMnnOfflineSpeechDenoiser *sd =
      SherpaMnnCreateOfflineSpeechDenoiser(&config);
  if (!sd) {
    fprintf(stderr, "Please check your config");
    return -1;
  }

  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    SherpaMnnDestroyOfflineSpeechDenoiser(sd);
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  const SherpaMnnDenoisedAudio *denoised = SherpaMnnOfflineSpeechDenoiserRun(
      sd, wave->samples, wave->num_samples, wave->sample_rate);

  SherpaMnnWriteWave(denoised->samples, denoised->n, denoised->sample_rate,
                      out_wave_filename);

  SherpaMnnDestroyDenoisedAudio(denoised);
  SherpaMnnFreeWave(wave);
  SherpaMnnDestroyOfflineSpeechDenoiser(sd);

  fprintf(stdout, "Saved to %s\n", out_wave_filename);
}
