// sherpa-mnn/jni/online-stream.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-stream.h"

#include "sherpa-mnn/jni/common.h"

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_OnlineStream_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_mnn::OnlineStream *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_OnlineStream_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jint sample_rate) {
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  stream->AcceptWaveform(sample_rate, p, n);
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_OnlineStream_inputFinished(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(ptr);
  stream->InputFinished();
}
