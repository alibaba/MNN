// c-api-examples/speaker-identification-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation

// We assume you have pre-downloaded the speaker embedding extractor model
// from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
//
// An example command to download
// "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
// is given below:
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
//
// clang-format on
//
// Also, please download the test wave files from
//
// https://github.com/csukuangfj/sr-data

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-mnn/c-api/c-api.h"

static const float *ComputeEmbedding(
    const SherpaMnnSpeakerEmbeddingExtractor *ex, const char *wav_filename) {
  const SherpaMnnWave *wave = SherpaMnnReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    exit(-1);
  }

  const SherpaMnnOnlineStream *stream =
      SherpaMnnSpeakerEmbeddingExtractorCreateStream(ex);

  SherpaMnnOnlineStreamAcceptWaveform(stream, wave->sample_rate, wave->samples,
                                       wave->num_samples);
  SherpaMnnOnlineStreamInputFinished(stream);

  if (!SherpaMnnSpeakerEmbeddingExtractorIsReady(ex, stream)) {
    fprintf(stderr, "The input wave file %s is too short!\n", wav_filename);
    exit(-1);
  }

  // we will free `v` outside of this function
  const float *v =
      SherpaMnnSpeakerEmbeddingExtractorComputeEmbedding(ex, stream);

  SherpaMnnDestroyOnlineStream(stream);
  SherpaMnnFreeWave(wave);

  // Remeber to free v to avoid memory leak
  return v;
}

int32_t main() {
  SherpaMnnSpeakerEmbeddingExtractorConfig config;

  memset(&config, 0, sizeof(config));

  // please download the model from
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
  config.model = "./3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx";

  config.num_threads = 1;
  config.debug = 0;
  config.provider = "cpu";

  const SherpaMnnSpeakerEmbeddingExtractor *ex =
      SherpaMnnCreateSpeakerEmbeddingExtractor(&config);
  if (!ex) {
    fprintf(stderr, "Failed to create speaker embedding extractor");
    return -1;
  }

  int32_t dim = SherpaMnnSpeakerEmbeddingExtractorDim(ex);

  const SherpaMnnSpeakerEmbeddingManager *manager =
      SherpaMnnCreateSpeakerEmbeddingManager(dim);

  // Please download the test data from
  // https://github.com/csukuangfj/sr-data
  const char *spk1_1 = "./sr-data/enroll/fangjun-sr-1.wav";
  const char *spk1_2 = "./sr-data/enroll/fangjun-sr-2.wav";
  const char *spk1_3 = "./sr-data/enroll/fangjun-sr-3.wav";

  const char *spk2_1 = "./sr-data/enroll/leijun-sr-1.wav";
  const char *spk2_2 = "./sr-data/enroll/leijun-sr-2.wav";

  const float *spk1_vec[4] = {NULL};
  spk1_vec[0] = ComputeEmbedding(ex, spk1_1);
  spk1_vec[1] = ComputeEmbedding(ex, spk1_2);
  spk1_vec[2] = ComputeEmbedding(ex, spk1_3);

  const float *spk2_vec[3] = {NULL};
  spk2_vec[0] = ComputeEmbedding(ex, spk2_1);
  spk2_vec[1] = ComputeEmbedding(ex, spk2_2);

  if (!SherpaMnnSpeakerEmbeddingManagerAddList(manager, "fangjun", spk1_vec)) {
    fprintf(stderr, "Failed to register fangjun\n");
    exit(-1);
  }

  if (!SherpaMnnSpeakerEmbeddingManagerContains(manager, "fangjun")) {
    fprintf(stderr, "Failed to find fangjun\n");
    exit(-1);
  }

  if (!SherpaMnnSpeakerEmbeddingManagerAddList(manager, "leijun", spk2_vec)) {
    fprintf(stderr, "Failed to register leijun\n");
    exit(-1);
  }

  if (!SherpaMnnSpeakerEmbeddingManagerContains(manager, "leijun")) {
    fprintf(stderr, "Failed to find leijun\n");
    exit(-1);
  }

  if (SherpaMnnSpeakerEmbeddingManagerNumSpeakers(manager) != 2) {
    fprintf(stderr, "There should be two speakers: fangjun and leijun\n");
    exit(-1);
  }

  const char *const *all_speakers =
      SherpaMnnSpeakerEmbeddingManagerGetAllSpeakers(manager);
  const char *const *p = all_speakers;
  fprintf(stderr, "list of registered speakers\n-----\n");
  while (p[0]) {
    fprintf(stderr, "speaker: %s\n", p[0]);
    ++p;
  }
  fprintf(stderr, "----\n");

  SherpaMnnSpeakerEmbeddingManagerFreeAllSpeakers(all_speakers);

  const char *test1 = "./sr-data/test/fangjun-test-sr-1.wav";
  const char *test2 = "./sr-data/test/leijun-test-sr-1.wav";
  const char *test3 = "./sr-data/test/liudehua-test-sr-1.wav";

  const float *v1 = ComputeEmbedding(ex, test1);
  const float *v2 = ComputeEmbedding(ex, test2);
  const float *v3 = ComputeEmbedding(ex, test3);

  float threshold = 0.6;

  const char *name1 =
      SherpaMnnSpeakerEmbeddingManagerSearch(manager, v1, threshold);
  if (name1) {
    fprintf(stderr, "%s: Found %s\n", test1, name1);
    SherpaMnnSpeakerEmbeddingManagerFreeSearch(name1);
  } else {
    fprintf(stderr, "%s: Not found\n", test1);
  }

  const char *name2 =
      SherpaMnnSpeakerEmbeddingManagerSearch(manager, v2, threshold);
  if (name2) {
    fprintf(stderr, "%s: Found %s\n", test2, name2);
    SherpaMnnSpeakerEmbeddingManagerFreeSearch(name2);
  } else {
    fprintf(stderr, "%s: Not found\n", test2);
  }

  const char *name3 =
      SherpaMnnSpeakerEmbeddingManagerSearch(manager, v3, threshold);
  if (name3) {
    fprintf(stderr, "%s: Found %s\n", test3, name3);
    SherpaMnnSpeakerEmbeddingManagerFreeSearch(name3);
  } else {
    fprintf(stderr, "%s: Not found\n", test3);
  }

  int32_t ok = SherpaMnnSpeakerEmbeddingManagerVerify(manager, "fangjun", v1,
                                                       threshold);
  if (ok) {
    fprintf(stderr, "%s matches fangjun\n", test1);
  } else {
    fprintf(stderr, "%s does NOT match fangjun\n", test1);
  }

  ok = SherpaMnnSpeakerEmbeddingManagerVerify(manager, "fangjun", v2,
                                               threshold);
  if (ok) {
    fprintf(stderr, "%s matches fangjun\n", test2);
  } else {
    fprintf(stderr, "%s does NOT match fangjun\n", test2);
  }

  fprintf(stderr, "Removing fangjun\n");
  if (!SherpaMnnSpeakerEmbeddingManagerRemove(manager, "fangjun")) {
    fprintf(stderr, "Failed to remove fangjun\n");
    exit(-1);
  }

  if (SherpaMnnSpeakerEmbeddingManagerNumSpeakers(manager) != 1) {
    fprintf(stderr, "There should be only 1 speaker left\n");
    exit(-1);
  }

  name1 = SherpaMnnSpeakerEmbeddingManagerSearch(manager, v1, threshold);
  if (name1) {
    fprintf(stderr, "%s: Found %s\n", test1, name1);
    SherpaMnnSpeakerEmbeddingManagerFreeSearch(name1);
  } else {
    fprintf(stderr, "%s: Not found\n", test1);
  }

  fprintf(stderr, "Removing leijun\n");
  if (!SherpaMnnSpeakerEmbeddingManagerRemove(manager, "leijun")) {
    fprintf(stderr, "Failed to remove leijun\n");
    exit(-1);
  }

  if (SherpaMnnSpeakerEmbeddingManagerNumSpeakers(manager) != 0) {
    fprintf(stderr, "There should be only 1 speaker left\n");
    exit(-1);
  }

  name2 = SherpaMnnSpeakerEmbeddingManagerSearch(manager, v2, threshold);
  if (name2) {
    fprintf(stderr, "%s: Found %s\n", test2, name2);
    SherpaMnnSpeakerEmbeddingManagerFreeSearch(name2);
  } else {
    fprintf(stderr, "%s: Not found\n", test2);
  }

  all_speakers = SherpaMnnSpeakerEmbeddingManagerGetAllSpeakers(manager);

  p = all_speakers;
  fprintf(stderr, "list of registered speakers\n-----\n");
  while (p[0]) {
    fprintf(stderr, "speaker: %s\n", p[0]);
    ++p;
  }
  fprintf(stderr, "----\n");

  SherpaMnnSpeakerEmbeddingManagerFreeAllSpeakers(all_speakers);
  SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding(v1);
  SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding(v2);
  SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding(v3);

  SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding(spk1_vec[0]);
  SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding(spk1_vec[1]);
  SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding(spk1_vec[2]);

  SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding(spk2_vec[0]);
  SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding(spk2_vec[1]);

  SherpaMnnDestroySpeakerEmbeddingManager(manager);
  SherpaMnnDestroySpeakerEmbeddingExtractor(ex);

  return 0;
}
