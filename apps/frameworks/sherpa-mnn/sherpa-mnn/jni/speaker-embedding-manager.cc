// sherpa-mnn/jni/speaker-embedding-manager.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-mnn/csrc/speaker-embedding-manager.h"

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/jni/common.h"

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingManager_create(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jint dim) {
  auto p = new sherpa_mnn::SpeakerEmbeddingManager(dim);
  return (jlong)p;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingManager_delete(JNIEnv * /*env*/,
                                                          jobject /*obj*/,
                                                          jlong ptr) {
  auto manager = reinterpret_cast<sherpa_mnn::SpeakerEmbeddingManager *>(ptr);
  delete manager;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingManager_add(JNIEnv *env,
                                                       jobject /*obj*/,
                                                       jlong ptr, jstring name,
                                                       jfloatArray embedding) {
  auto manager = reinterpret_cast<sherpa_mnn::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize n = env->GetArrayLength(embedding);

  if (n != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(n));
    exit(-1);
  }

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Add(p_name, p);
  env->ReleaseStringUTFChars(name, p_name);
  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingManager_addList(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring name,
    jobjectArray embedding_arr) {
  auto manager = reinterpret_cast<sherpa_mnn::SpeakerEmbeddingManager *>(ptr);

  int num_embeddings = env->GetArrayLength(embedding_arr);
  if (num_embeddings == 0) {
    return false;
  }

  std::vector<std::vector<float>> embedding_list;
  embedding_list.reserve(num_embeddings);
  for (int32_t i = 0; i != num_embeddings; ++i) {
    jfloatArray embedding =
        (jfloatArray)env->GetObjectArrayElement(embedding_arr, i);

    jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
    jsize n = env->GetArrayLength(embedding);

    if (n != manager->Dim()) {
      SHERPA_ONNX_LOGE("i: %d. Expected dim %d, given %d", i, manager->Dim(),
                       static_cast<int32_t>(n));
      exit(-1);
    }

    embedding_list.push_back({p, p + n});
    env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);
  }

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Add(p_name, embedding_list);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingManager_remove(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr,
                                                          jstring name) {
  auto manager = reinterpret_cast<sherpa_mnn::SpeakerEmbeddingManager *>(ptr);

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Remove(p_name);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingManager_search(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr,
                                                          jfloatArray embedding,
                                                          jfloat threshold) {
  auto manager = reinterpret_cast<sherpa_mnn::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize n = env->GetArrayLength(embedding);

  if (n != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(n));
    exit(-1);
  }

  std::string name = manager->Search(p, threshold);

  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  return env->NewStringUTF(name.c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingManager_verify(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring name,
    jfloatArray embedding, jfloat threshold) {
  auto manager = reinterpret_cast<sherpa_mnn::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize n = env->GetArrayLength(embedding);

  if (n != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(n));
    exit(-1);
  }

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Verify(p_name, p, threshold);

  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingManager_contains(JNIEnv *env,
                                                            jobject /*obj*/,
                                                            jlong ptr,
                                                            jstring name) {
  auto manager = reinterpret_cast<sherpa_mnn::SpeakerEmbeddingManager *>(ptr);

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Contains(p_name);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingManager_numSpeakers(JNIEnv * /*env*/,
                                                               jobject /*obj*/,
                                                               jlong ptr) {
  auto manager = reinterpret_cast<sherpa_mnn::SpeakerEmbeddingManager *>(ptr);
  return manager->NumSpeakers();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_SpeakerEmbeddingManager_allSpeakerNames(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto manager = reinterpret_cast<sherpa_mnn::SpeakerEmbeddingManager *>(ptr);
  std::vector<std::string> all_speakers = manager->GetAllSpeakers();

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      all_speakers.size(), env->FindClass("java/lang/String"), nullptr);

  int32_t i = 0;
  for (auto &s : all_speakers) {
    jstring js = env->NewStringUTF(s.c_str());
    env->SetObjectArrayElement(obj_arr, i, js);

    ++i;
  }

  return obj_arr;
}
