// sherpa-mnn/jni/spoken-language-identification.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/spoken-language-identification.h"

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/jni/common.h"

namespace sherpa_mnn {

static SpokenLanguageIdentificationConfig GetSpokenLanguageIdentificationConfig(
    JNIEnv *env, jobject config) {
  SpokenLanguageIdentificationConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid = env->GetFieldID(
      cls, "whisper",
      "Lcom/k2fsa/sherpa/onnx/SpokenLanguageIdentificationWhisperConfig;");

  jobject whisper = env->GetObjectField(config, fid);
  jclass whisper_cls = env->GetObjectClass(whisper);

  fid = env->GetFieldID(whisper_cls, "encoder", "Ljava/lang/String;");

  jstring s = (jstring)env->GetObjectField(whisper, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.whisper.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(whisper, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.whisper.decoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_cls, "tailPaddings", "I");
  ans.whisper.tail_paddings = env->GetIntField(whisper, fid);

  fid = env->GetFieldID(cls, "numThreads", "I");
  ans.num_threads = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "debug", "Z");
  ans.debug = env->GetBooleanField(config, fid);

  fid = env->GetFieldID(cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.provider = p;
  env->ReleaseStringUTFChars(s, p);

  return ans;
}

}  // namespace sherpa_mnn

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_SpokenLanguageIdentification_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif

  auto config =
      sherpa_mnn::GetSpokenLanguageIdentificationConfig(env, _config);
  SHERPA_ONNX_LOGE("spoken language identification newFromAsset config:\n%s",
                   config.ToString().c_str());

  auto slid = new sherpa_mnn::SpokenLanguageIdentification(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);
  SHERPA_ONNX_LOGE("slid %p", slid);

  return (jlong)slid;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_SpokenLanguageIdentification_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config =
      sherpa_mnn::GetSpokenLanguageIdentificationConfig(env, _config);
  SHERPA_ONNX_LOGE("SpokenLanguageIdentification newFromFile config:\n%s",
                   config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto tagger = new sherpa_mnn::SpokenLanguageIdentification(config);

  return (jlong)tagger;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_SpokenLanguageIdentification_delete(JNIEnv * /*env*/,
                                                               jobject /*obj*/,
                                                               jlong ptr) {
  delete reinterpret_cast<sherpa_mnn::SpokenLanguageIdentification *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_SpokenLanguageIdentification_createStream(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  auto slid =
      reinterpret_cast<sherpa_mnn::SpokenLanguageIdentification *>(ptr);
  std::unique_ptr<sherpa_mnn::OfflineStream> s = slid->CreateStream();

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_mnn_OfflineStream_delete() from
  // ./offline-stream.cc
  sherpa_mnn::OfflineStream *p = s.release();
  return (jlong)p;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_mnn_SpokenLanguageIdentification_compute(JNIEnv *env,
                                                                jobject /*obj*/,
                                                                jlong ptr,
                                                                jlong s_ptr) {
  sherpa_mnn::SpokenLanguageIdentification *slid =
      reinterpret_cast<sherpa_mnn::SpokenLanguageIdentification *>(ptr);
  sherpa_mnn::OfflineStream *s =
      reinterpret_cast<sherpa_mnn::OfflineStream *>(s_ptr);
  std::string lang = slid->Compute(s);
  return env->NewStringUTF(lang.c_str());
}
