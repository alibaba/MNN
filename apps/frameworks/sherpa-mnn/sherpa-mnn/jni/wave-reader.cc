// sherpa-mnn/jni/wave-reader.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-mnn/csrc/wave-reader.h"

#include <fstream>

#include "sherpa-mnn/csrc/file-utils.h"
#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/jni/common.h"

static jobjectArray ReadWaveImpl(JNIEnv *env, std::istream &is,
                                 const char *p_filename) {
  bool is_ok = false;
  int32_t sampling_rate = -1;
  std::vector<float> samples =
      sherpa_mnn::ReadWave(is, &sampling_rate, &is_ok);

  if (!is_ok) {
    SHERPA_ONNX_LOGE("Failed to read '%s'", p_filename);
    jclass exception_class = env->FindClass("java/lang/Exception");
    env->ThrowNew(exception_class, "Failed to read wave file.");
    return nullptr;
  }

  jfloatArray samples_arr = env->NewFloatArray(samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, samples.size(), samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, sampling_rate));

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_WaveReader_00024Companion_readWaveFromFile(
    JNIEnv *env, jclass /*cls*/, jstring filename) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);
  std::ifstream is(p_filename, std::ios::binary);

  auto obj_arr = ReadWaveImpl(env, is, p_filename);

  env->ReleaseStringUTFChars(filename, p_filename);

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_WaveReader_readWaveFromFile(JNIEnv *env,
                                                       jclass /*obj*/,
                                                       jstring filename) {
  return Java_com_k2fsa_sherpa_mnn_WaveReader_00024Companion_readWaveFromFile(
      env, nullptr, filename);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_WaveReader_00024Companion_readWaveFromAsset(
    JNIEnv *env, jclass /*cls*/, jobject asset_manager, jstring filename) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    exit(-1);
  }
  std::vector<char> buffer = sherpa_mnn::ReadFile(mgr, p_filename);

  std::istrstream is(buffer.data(), buffer.size());
#else
  std::ifstream is(p_filename, std::ios::binary);
#endif

  auto obj_arr = ReadWaveImpl(env, is, p_filename);

  env->ReleaseStringUTFChars(filename, p_filename);

  return obj_arr;
}
