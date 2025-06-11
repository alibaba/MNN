// sherpa-mnn/jni/jni.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation
//                2022       Pingfeng Luo
//                2023       Zhaoming

#include <fstream>

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/csrc/onnx-utils.h"
#include "sherpa-mnn/csrc/wave-writer.h"
#include "sherpa-mnn/jni/common.h"

// see
// https://stackoverflow.com/questions/29043872/android-jni-return-multiple-variables
jobject NewInteger(JNIEnv *env, int32_t value) {
  jclass cls = env->FindClass("java/lang/Integer");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(I)V");
  return env->NewObject(cls, constructor, value);
}

jobject NewFloat(JNIEnv *env, float value) {
  jclass cls = env->FindClass("java/lang/Float");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(F)V");
  return env->NewObject(cls, constructor, value);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL Java_com_k2fsa_sherpa_mnn_GeneratedAudio_saveImpl(
    JNIEnv *env, jobject /*obj*/, jstring filename, jfloatArray samples,
    jint sample_rate) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  bool ok = sherpa_mnn::WriteWave(p_filename, sample_rate, p, n);

  env->ReleaseStringUTFChars(filename, p_filename);
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);

  return ok;
}

#if 0
SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_OnlineRecognizer_decodeStreams(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr,
                                                          jlongArray ss_ptr,
                                                          jint stream_size) {
  sherpa_mnn::OnlineRecognizer *model =
      reinterpret_cast<sherpa_mnn::OnlineRecognizer *>(ptr);
  jlong *p = env->GetLongArrayElements(ss_ptr, nullptr);
  jsize n = env->GetArrayLength(ss_ptr);
  std::vector<sherpa_mnn::OnlineStream *> p_ss(n);
  for (int32_t i = 0; i != n; ++i) {
    p_ss[i] = reinterpret_cast<sherpa_mnn::OnlineStream *>(p[i]);
  }

  model->DecodeStreams(p_ss.data(), n);
  env->ReleaseLongArrayElements(ss_ptr, p, JNI_ABORT);
}
#endif
