// sherpa-mnn/csrc/voice-activity-detector.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-mnn/csrc/voice-activity-detector.h"

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/jni/common.h"

namespace sherpa_mnn {

static VadModelConfig GetVadModelConfig(JNIEnv *env, jobject config) {
  VadModelConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  // silero_vad
  fid = env->GetFieldID(cls, "sileroVadModelConfig",
                        "Lcom/k2fsa/sherpa/onnx/SileroVadModelConfig;");
  jobject silero_vad_config = env->GetObjectField(config, fid);
  jclass silero_vad_config_cls = env->GetObjectClass(silero_vad_config);

  fid = env->GetFieldID(silero_vad_config_cls, "model", "Ljava/lang/String;");
  auto s = (jstring)env->GetObjectField(silero_vad_config, fid);
  auto p = env->GetStringUTFChars(s, nullptr);
  ans.silero_vad.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(silero_vad_config_cls, "threshold", "F");
  ans.silero_vad.threshold = env->GetFloatField(silero_vad_config, fid);

  fid = env->GetFieldID(silero_vad_config_cls, "minSilenceDuration", "F");
  ans.silero_vad.min_silence_duration =
      env->GetFloatField(silero_vad_config, fid);

  fid = env->GetFieldID(silero_vad_config_cls, "minSpeechDuration", "F");
  ans.silero_vad.min_speech_duration =
      env->GetFloatField(silero_vad_config, fid);

  fid = env->GetFieldID(silero_vad_config_cls, "windowSize", "I");
  ans.silero_vad.window_size = env->GetIntField(silero_vad_config, fid);

  fid = env->GetFieldID(silero_vad_config_cls, "maxSpeechDuration", "F");
  ans.silero_vad.max_speech_duration =
      env->GetFloatField(silero_vad_config, fid);

  fid = env->GetFieldID(cls, "sampleRate", "I");
  ans.sample_rate = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "numThreads", "I");
  ans.num_threads = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.provider = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "debug", "Z");
  ans.debug = env->GetBooleanField(config, fid);

  return ans;
}

}  // namespace sherpa_mnn

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_mnn_Vad_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  auto config = sherpa_mnn::GetVadModelConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto model = new sherpa_mnn::VoiceActivityDetector(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_mnn_Vad_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config = sherpa_mnn::GetVadModelConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto model = new sherpa_mnn::VoiceActivityDetector(config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_Vad_delete(JNIEnv * /*env*/,
                                                             jobject /*obj*/,
                                                             jlong ptr) {
  delete reinterpret_cast<sherpa_mnn::VoiceActivityDetector *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_Vad_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples) {
  SafeJNI(env, "Vad_acceptWaveform", [&] {
    if (!ValidatePointer(env, ptr, "Vad_acceptWaveform",
                         "VoiceActivityDetector pointer is null.")) {
      return;
    }

    auto model = reinterpret_cast<sherpa_mnn::VoiceActivityDetector *>(ptr);
    jfloat *p = env->GetFloatArrayElements(samples, nullptr);
    jsize n = env->GetArrayLength(samples);

    model->AcceptWaveform(p, n);

    env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
  });
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_mnn_Vad_empty(JNIEnv * /*env*/,
                                                            jobject /*obj*/,
                                                            jlong ptr) {
  auto model = reinterpret_cast<sherpa_mnn::VoiceActivityDetector *>(ptr);
  return model->Empty();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_Vad_pop(JNIEnv * /*env*/,
                                                          jobject /*obj*/,
                                                          jlong ptr) {
  auto model = reinterpret_cast<sherpa_mnn::VoiceActivityDetector *>(ptr);
  model->Pop();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_Vad_clear(JNIEnv * /*env*/,
                                                            jobject /*obj*/,
                                                            jlong ptr) {
  auto model = reinterpret_cast<sherpa_mnn::VoiceActivityDetector *>(ptr);
  model->Clear();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_front(JNIEnv *env, jobject /*obj*/, jlong ptr) {
  const auto &front =
      reinterpret_cast<sherpa_mnn::VoiceActivityDetector *>(ptr)->Front();

  jfloatArray samples_arr = env->NewFloatArray(front.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, front.samples.size(),
                           front.samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, NewInteger(env, front.start));
  env->SetObjectArrayElement(obj_arr, 1, samples_arr);

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_mnn_Vad_isSpeechDetected(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_mnn::VoiceActivityDetector *>(ptr);
  return model->IsSpeechDetected();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_Vad_reset(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  SafeJNI(env, "Vad_reset", [&] {
    if (!ValidatePointer(env, ptr, "Vad_reset",
                         "VoiceActivityDetector pointer is null.")) {
      return;
    }

    auto model = reinterpret_cast<sherpa_mnn::VoiceActivityDetector *>(ptr);
    model->Reset();
  });
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_Vad_flush(JNIEnv * /*env*/,
                                                            jobject /*obj*/,
                                                            jlong ptr) {
  auto model = reinterpret_cast<sherpa_mnn::VoiceActivityDetector *>(ptr);
  model->Flush();
}
