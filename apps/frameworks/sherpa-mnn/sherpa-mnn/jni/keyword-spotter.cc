// sherpa-mnn/jni/keyword-spotter.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/keyword-spotter.h"

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/jni/common.h"

namespace sherpa_mnn {

static KeywordSpotterConfig GetKwsConfig(JNIEnv *env, jobject config) {
  KeywordSpotterConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  // https://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/types.html
  // https://courses.cs.washington.edu/courses/cse341/99wi/java/tutorial/native1.1/implementing/field.html

  //---------- decoding ----------
  fid = env->GetFieldID(cls, "maxActivePaths", "I");
  ans.max_active_paths = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "keywordsFile", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.keywords_file = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "keywordsScore", "F");
  ans.keywords_score = env->GetFloatField(config, fid);

  fid = env->GetFieldID(cls, "keywordsThreshold", "F");
  ans.keywords_threshold = env->GetFloatField(config, fid);

  fid = env->GetFieldID(cls, "numTrailingBlanks", "I");
  ans.num_trailing_blanks = env->GetIntField(config, fid);

  //---------- feat config ----------
  fid = env->GetFieldID(cls, "featConfig",
                        "Lcom/k2fsa/sherpa/mnn/FeatureConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  fid = env->GetFieldID(feat_config_cls, "sampleRate", "I");
  ans.feat_config.sampling_rate = env->GetIntField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "featureDim", "I");
  ans.feat_config.feature_dim = env->GetIntField(feat_config, fid);

  //---------- model config ----------
  fid = env->GetFieldID(cls, "modelConfig",
                        "Lcom/k2fsa/sherpa/mnn/OnlineModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  // transducer
  fid = env->GetFieldID(model_config_cls, "transducer",
                        "Lcom/k2fsa/sherpa/mnn/OnlineTransducerModelConfig;");
  jobject transducer_config = env->GetObjectField(model_config, fid);
  jclass transducer_config_cls = env->GetObjectClass(transducer_config);

  fid = env->GetFieldID(transducer_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(transducer_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.decoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(transducer_config_cls, "joiner", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.joiner = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "numThreads", "I");
  ans.model_config.num_threads = env->GetIntField(model_config, fid);

  fid = env->GetFieldID(model_config_cls, "debug", "Z");
  ans.model_config.debug = env->GetBooleanField(model_config, fid);

  fid = env->GetFieldID(model_config_cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.provider_config.provider = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "modelType", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.model_type = p;
  env->ReleaseStringUTFChars(s, p);

  return ans;
}

}  // namespace sherpa_mnn

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_mnn_KeywordSpotter_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  auto config = sherpa_mnn::GetKwsConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto kws = new sherpa_mnn::KeywordSpotter(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)kws;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_mnn_KeywordSpotter_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config = sherpa_mnn::GetKwsConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto kws = new sherpa_mnn::KeywordSpotter(config);

  return (jlong)kws;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_KeywordSpotter_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_mnn::KeywordSpotter *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_KeywordSpotter_decode(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto kws = reinterpret_cast<sherpa_mnn::KeywordSpotter *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);

  kws->DecodeStream(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_KeywordSpotter_reset(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto kws = reinterpret_cast<sherpa_mnn::KeywordSpotter *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);

  kws->Reset(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_mnn_KeywordSpotter_createStream(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring keywords) {
  auto kws = reinterpret_cast<sherpa_mnn::KeywordSpotter *>(ptr);

  const char *p = env->GetStringUTFChars(keywords, nullptr);
  std::unique_ptr<sherpa_mnn::OnlineStream> stream;

  if (strlen(p) == 0) {
    stream = kws->CreateStream();
  } else {
    stream = kws->CreateStream(p);
  }

  env->ReleaseStringUTFChars(keywords, p);

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_mnn_OfflineStream_delete() from
  // ./offline-stream.cc
  sherpa_mnn::OnlineStream *ans = stream.release();
  return (jlong)ans;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_mnn_KeywordSpotter_isReady(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto kws = reinterpret_cast<sherpa_mnn::KeywordSpotter *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);

  return kws->IsReady(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_KeywordSpotter_getResult(JNIEnv *env,
                                                    jobject /*obj*/, jlong ptr,
                                                    jlong stream_ptr) {
  auto kws = reinterpret_cast<sherpa_mnn::KeywordSpotter *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);

  sherpa_mnn::KeywordResult result = kws->GetResult(stream);

  // [0]: keyword, jstring
  // [1]: tokens, array of jstring
  // [2]: timestamps, array of float
  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      3, env->FindClass("java/lang/Object"), nullptr);

  jstring keyword = env->NewStringUTF(result.keyword.c_str());
  env->SetObjectArrayElement(obj_arr, 0, keyword);

  jobjectArray tokens_arr = (jobjectArray)env->NewObjectArray(
      result.tokens.size(), env->FindClass("java/lang/String"), nullptr);

  int32_t i = 0;
  for (const auto &t : result.tokens) {
    jstring jtext = env->NewStringUTF(t.c_str());
    env->SetObjectArrayElement(tokens_arr, i, jtext);
    i += 1;
  }

  env->SetObjectArrayElement(obj_arr, 1, tokens_arr);

  jfloatArray timestamps_arr = env->NewFloatArray(result.timestamps.size());
  env->SetFloatArrayRegion(timestamps_arr, 0, result.timestamps.size(),
                           result.timestamps.data());

  env->SetObjectArrayElement(obj_arr, 2, timestamps_arr);

  return obj_arr;
}
