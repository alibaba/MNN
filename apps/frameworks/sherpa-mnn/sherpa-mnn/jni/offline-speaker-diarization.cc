// sherpa-mnn/jni/offline-speaker-diarization.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-speaker-diarization.h"

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/jni/common.h"

namespace sherpa_mnn {

static OfflineSpeakerDiarizationConfig GetOfflineSpeakerDiarizationConfig(
    JNIEnv *env, jobject config) {
  OfflineSpeakerDiarizationConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  //---------- segmentation ----------
  fid = env->GetFieldID(
      cls, "segmentation",
      "Lcom/k2fsa/sherpa/mnn/OfflineSpeakerSegmentationModelConfig;");
  jobject segmentation_config = env->GetObjectField(config, fid);
  jclass segmentation_config_cls = env->GetObjectClass(segmentation_config);

  fid = env->GetFieldID(
      segmentation_config_cls, "pyannote",
      "Lcom/k2fsa/sherpa/mnn/OfflineSpeakerSegmentationPyannoteModelConfig;");
  jobject pyannote_config = env->GetObjectField(segmentation_config, fid);
  jclass pyannote_config_cls = env->GetObjectClass(pyannote_config);

  fid = env->GetFieldID(pyannote_config_cls, "model", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(pyannote_config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.segmentation.pyannote.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(segmentation_config_cls, "numThreads", "I");
  ans.segmentation.num_threads = env->GetIntField(segmentation_config, fid);

  fid = env->GetFieldID(segmentation_config_cls, "debug", "Z");
  ans.segmentation.debug = env->GetBooleanField(segmentation_config, fid);

  fid = env->GetFieldID(segmentation_config_cls, "provider",
                        "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(segmentation_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.segmentation.provider = p;
  env->ReleaseStringUTFChars(s, p);

  //---------- embedding ----------
  fid = env->GetFieldID(
      cls, "embedding",
      "Lcom/k2fsa/sherpa/mnn/SpeakerEmbeddingExtractorConfig;");
  jobject embedding_config = env->GetObjectField(config, fid);
  jclass embedding_config_cls = env->GetObjectClass(embedding_config);

  fid = env->GetFieldID(embedding_config_cls, "model", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(embedding_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.embedding.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(embedding_config_cls, "numThreads", "I");
  ans.embedding.num_threads = env->GetIntField(embedding_config, fid);

  fid = env->GetFieldID(embedding_config_cls, "debug", "Z");
  ans.embedding.debug = env->GetBooleanField(embedding_config, fid);

  fid = env->GetFieldID(embedding_config_cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(embedding_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.embedding.provider = p;
  env->ReleaseStringUTFChars(s, p);

  //---------- clustering ----------
  fid = env->GetFieldID(cls, "clustering",
                        "Lcom/k2fsa/sherpa/mnn/FastClusteringConfig;");
  jobject clustering_config = env->GetObjectField(config, fid);
  jclass clustering_config_cls = env->GetObjectClass(clustering_config);

  fid = env->GetFieldID(clustering_config_cls, "numClusters", "I");
  ans.clustering.num_clusters = env->GetIntField(clustering_config, fid);

  fid = env->GetFieldID(clustering_config_cls, "threshold", "F");
  ans.clustering.threshold = env->GetFloatField(clustering_config, fid);

  // its own fields
  fid = env->GetFieldID(cls, "minDurationOn", "F");
  ans.min_duration_on = env->GetFloatField(config, fid);

  fid = env->GetFieldID(cls, "minDurationOff", "F");
  ans.min_duration_off = env->GetFloatField(config, fid);

  return ans;
}

}  // namespace sherpa_mnn

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_OfflineSpeakerDiarization_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif

  auto config = sherpa_mnn::GetOfflineSpeakerDiarizationConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto sd = new sherpa_mnn::OfflineSpeakerDiarization(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)sd;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_OfflineSpeakerDiarization_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config = sherpa_mnn::GetOfflineSpeakerDiarizationConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto sd = new sherpa_mnn::OfflineSpeakerDiarization(config);

  return (jlong)sd;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_OfflineSpeakerDiarization_setConfig(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jobject _config) {
  auto config = sherpa_mnn::GetOfflineSpeakerDiarizationConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto sd = reinterpret_cast<sherpa_mnn::OfflineSpeakerDiarization *>(ptr);
  sd->SetConfig(config);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_OfflineSpeakerDiarization_delete(JNIEnv * /*env*/,
                                                            jobject /*obj*/,
                                                            jlong ptr) {
  delete reinterpret_cast<sherpa_mnn::OfflineSpeakerDiarization *>(ptr);
}

static jobjectArray ProcessImpl(
    JNIEnv *env,
    const std::vector<sherpa_mnn::OfflineSpeakerDiarizationSegment>
        &segments) {
  jclass cls =
      env->FindClass("com/k2fsa/sherpa/mnn/OfflineSpeakerDiarizationSegment");

  jobjectArray obj_arr =
      (jobjectArray)env->NewObjectArray(segments.size(), cls, nullptr);

  jmethodID constructor = env->GetMethodID(cls, "<init>", "(FFI)V");

  for (int32_t i = 0; i != segments.size(); ++i) {
    const auto &s = segments[i];
    jobject segment =
        env->NewObject(cls, constructor, s.Start(), s.End(), s.Speaker());
    env->SetObjectArrayElement(obj_arr, i, segment);
  }

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_OfflineSpeakerDiarization_process(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples) {
  auto sd = reinterpret_cast<sherpa_mnn::OfflineSpeakerDiarization *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  auto segments = sd->Process(p, n).SortByStartTime();
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);

  return ProcessImpl(env, segments);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_OfflineSpeakerDiarization_processWithCallback(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jobject callback, jlong arg) {
  std::function<int32_t(int32_t, int32_t, void *)> callback_wrapper =
      [env, callback](int32_t num_processed_chunks, int32_t num_total_chunks,
                      void *data) -> int {
    jclass cls = env->GetObjectClass(callback);

    jmethodID mid = env->GetMethodID(cls, "invoke", "(IIJ)Ljava/lang/Integer;");
    if (mid == nullptr) {
      SHERPA_ONNX_LOGE("Failed to get the callback. Ignore it.");
      return 0;
    }

    jobject ret = env->CallObjectMethod(callback, mid, num_processed_chunks,
                                        num_total_chunks, (jlong)data);
    jclass jklass = env->GetObjectClass(ret);
    jmethodID int_value_mid = env->GetMethodID(jklass, "intValue", "()I");
    return env->CallIntMethod(ret, int_value_mid);
  };

  auto sd = reinterpret_cast<sherpa_mnn::OfflineSpeakerDiarization *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  auto segments =
      sd->Process(p, n, callback_wrapper, reinterpret_cast<void *>(arg))
          .SortByStartTime();
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);

  return ProcessImpl(env, segments);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL
Java_com_k2fsa_sherpa_mnn_OfflineSpeakerDiarization_getSampleRate(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_mnn::OfflineSpeakerDiarization *>(ptr)
      ->SampleRate();
}
