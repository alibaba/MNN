// sherpa-mnn/jni/wave-writer.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-mnn/csrc/wave-writer.h"

#include "sherpa-mnn/jni/common.h"

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_mnn_WaveWriter_writeWaveToFile(
    JNIEnv *env, jclass /*obj*/, jstring filename, jfloatArray samples,
    jint sample_rate) {
  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  const char *p_filename = env->GetStringUTFChars(filename, nullptr);

  bool ok = sherpa_mnn::WriteWave(p_filename, sample_rate, p, n);

  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
  env->ReleaseStringUTFChars(filename, p_filename);

  return ok;
}
