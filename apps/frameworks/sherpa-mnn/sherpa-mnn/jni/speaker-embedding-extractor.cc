// sherpa-mnn/jni/speaker-embedding-extractor.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-mnn/csrc/speaker-embedding-extractor.h"

#include "sherpa-mnn/jni/common.h"

namespace sherpa_mnn {

static SpeakerEmbeddingExtractorConfig GetSpeakerEmbeddingExtractorConfig(
    JNIEnv *env, jobject config) {
  SpeakerEmbeddingExtractorConfig ans;

  jclass cls = env->GetObjectClass(config);

  jfieldID fid = env->GetFieldID(cls, "model", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);

  ans.model = p;
  env->ReleaseStringUTFChars(s, p);

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
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingExtractor_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  auto config = sherpa_mnn::GetSpeakerEmbeddingExtractorConfig(env, _config);
  SHERPA_ONNX_LOGE("new config:\n%s", config.ToString().c_str());

  auto extractor = new sherpa_mnn::SpeakerEmbeddingExtractor(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)extractor;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingExtractor_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config = sherpa_mnn::GetSpeakerEmbeddingExtractorConfig(env, _config);
  SHERPA_ONNX_LOGE("newFromFile config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
  }

  auto extractor = new sherpa_mnn::SpeakerEmbeddingExtractor(config);

  return (jlong)extractor;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingExtractor_delete(JNIEnv * /*env*/,
                                                            jobject /*obj*/,
                                                            jlong ptr) {
  delete reinterpret_cast<sherpa_mnn::SpeakerEmbeddingExtractor *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingExtractor_createStream(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  std::unique_ptr<sherpa_mnn::OnlineStream> s =
      reinterpret_cast<sherpa_mnn::SpeakerEmbeddingExtractor *>(ptr)
          ->CreateStream();

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_mnn_OnlineStream_delete() from
  // ./online-stream.cc
  sherpa_mnn::OnlineStream *p = s.release();
  return (jlong)p;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingExtractor_isReady(JNIEnv * /*env*/,
                                                             jobject /*obj*/,
                                                             jlong ptr,
                                                             jlong stream_ptr) {
  auto extractor =
      reinterpret_cast<sherpa_mnn::SpeakerEmbeddingExtractor *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);
  return extractor->IsReady(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jfloatArray JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingExtractor_compute(JNIEnv *env,
                                                             jobject /*obj*/,
                                                             jlong ptr,
                                                             jlong stream_ptr) {
  auto extractor =
      reinterpret_cast<sherpa_mnn::SpeakerEmbeddingExtractor *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);

  std::vector<float> embedding = extractor->Compute(stream);
  jfloatArray embedding_arr = env->NewFloatArray(embedding.size());
  env->SetFloatArrayRegion(embedding_arr, 0, embedding.size(),
                           embedding.data());
  return embedding_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingExtractor_dim(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  auto extractor =
      reinterpret_cast<sherpa_mnn::SpeakerEmbeddingExtractor *>(ptr);
  return extractor->Dim();
}
