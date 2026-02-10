// sherpa-mnn/jni/offline-punctuation.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-punctuation.h"

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/jni/common.h"

namespace sherpa_mnn {

static OfflinePunctuationConfig GetOfflinePunctuationConfig(JNIEnv *env,
                                                            jobject config) {
  OfflinePunctuationConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  fid = env->GetFieldID(
      cls, "model", "Lcom/k2fsa/sherpa/mnn/OfflinePunctuationModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  fid =
      env->GetFieldID(model_config_cls, "ctTransformer", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(model_config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.model.ct_transformer = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "numThreads", "I");
  ans.model.num_threads = env->GetIntField(model_config, fid);

  fid = env->GetFieldID(model_config_cls, "debug", "Z");
  ans.model.debug = env->GetBooleanField(model_config, fid);

  fid = env->GetFieldID(model_config_cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.provider = p;
  env->ReleaseStringUTFChars(s, p);

  return ans;
}

}  // namespace sherpa_mnn

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_OfflinePunctuation_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  auto config = sherpa_mnn::GetOfflinePunctuationConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto model = new sherpa_mnn::OfflinePunctuation(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_OfflinePunctuation_newFromFile(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jobject _config) {
  auto config = sherpa_mnn::GetOfflinePunctuationConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto model = new sherpa_mnn::OfflinePunctuation(config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_OfflinePunctuation_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_mnn::OfflinePunctuation *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_mnn_OfflinePunctuation_addPunctuation(JNIEnv *env,
                                                             jobject /*obj*/,
                                                             jlong ptr,
                                                             jstring text) {
  auto punct = reinterpret_cast<const sherpa_mnn::OfflinePunctuation *>(ptr);

  const char *ptext = env->GetStringUTFChars(text, nullptr);

  std::string result = punct->AddPunctuation(ptext);

  env->ReleaseStringUTFChars(text, ptext);

  return env->NewStringUTF(result.c_str());
}
