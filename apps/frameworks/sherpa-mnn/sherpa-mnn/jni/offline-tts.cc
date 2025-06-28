// sherpa-mnn/jni/offline-tts.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/offline-tts.h"

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/jni/common.h"

namespace sherpa_mnn {

static OfflineTtsConfig GetOfflineTtsConfig(JNIEnv *env, jobject config) {
  OfflineTtsConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  fid = env->GetFieldID(cls, "model",
                        "Lcom/k2fsa/sherpa/mnn/OfflineTtsModelConfig;");
  jobject model = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model);

  // vits
  fid = env->GetFieldID(model_config_cls, "vits",
                        "Lcom/k2fsa/sherpa/mnn/OfflineTtsVitsModelConfig;");
  jobject vits = env->GetObjectField(model, fid);
  jclass vits_cls = env->GetObjectClass(vits);

  fid = env->GetFieldID(vits_cls, "model", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(vits, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.model.vits.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(vits_cls, "lexicon", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(vits, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.vits.lexicon = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(vits_cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(vits, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.vits.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(vits_cls, "dataDir", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(vits, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.vits.data_dir = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(vits_cls, "dictDir", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(vits, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.vits.dict_dir = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(vits_cls, "noiseScale", "F");
  ans.model.vits.noise_scale = env->GetFloatField(vits, fid);

  fid = env->GetFieldID(vits_cls, "noiseScaleW", "F");
  ans.model.vits.noise_scale_w = env->GetFloatField(vits, fid);

  fid = env->GetFieldID(vits_cls, "lengthScale", "F");
  ans.model.vits.length_scale = env->GetFloatField(vits, fid);

  // matcha
  fid = env->GetFieldID(model_config_cls, "matcha",
                        "Lcom/k2fsa/sherpa/mnn/OfflineTtsMatchaModelConfig;");
  jobject matcha = env->GetObjectField(model, fid);
  jclass matcha_cls = env->GetObjectClass(matcha);

  fid = env->GetFieldID(matcha_cls, "acousticModel", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(matcha, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.matcha.acoustic_model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(matcha_cls, "vocoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(matcha, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.matcha.vocoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(matcha_cls, "lexicon", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(matcha, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.matcha.lexicon = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(matcha_cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(matcha, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.matcha.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(matcha_cls, "dataDir", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(matcha, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.matcha.data_dir = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(matcha_cls, "dictDir", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(matcha, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.matcha.dict_dir = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(matcha_cls, "noiseScale", "F");
  ans.model.matcha.noise_scale = env->GetFloatField(matcha, fid);

  fid = env->GetFieldID(matcha_cls, "lengthScale", "F");
  ans.model.matcha.length_scale = env->GetFloatField(matcha, fid);

  // kokoro
  fid = env->GetFieldID(model_config_cls, "kokoro",
                        "Lcom/k2fsa/sherpa/mnn/OfflineTtsKokoroModelConfig;");
  jobject kokoro = env->GetObjectField(model, fid);
  jclass kokoro_cls = env->GetObjectClass(kokoro);

  fid = env->GetFieldID(kokoro_cls, "model", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(kokoro, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.kokoro.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(kokoro_cls, "voices", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(kokoro, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.kokoro.voices = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(kokoro_cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(kokoro, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.kokoro.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(kokoro_cls, "lexicon", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(kokoro, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.kokoro.lexicon = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(kokoro_cls, "dataDir", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(kokoro, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.kokoro.data_dir = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(kokoro_cls, "dictDir", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(kokoro, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.kokoro.dict_dir = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(kokoro_cls, "lengthScale", "F");
  ans.model.kokoro.length_scale = env->GetFloatField(kokoro, fid);

  fid = env->GetFieldID(model_config_cls, "numThreads", "I");
  ans.model.num_threads = env->GetIntField(model, fid);

  fid = env->GetFieldID(model_config_cls, "debug", "Z");
  ans.model.debug = env->GetBooleanField(model, fid);

  fid = env->GetFieldID(model_config_cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.provider = p;
  env->ReleaseStringUTFChars(s, p);

  // for ruleFsts
  fid = env->GetFieldID(cls, "ruleFsts", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.rule_fsts = p;
  env->ReleaseStringUTFChars(s, p);

  // for ruleFars
  fid = env->GetFieldID(cls, "ruleFars", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.rule_fars = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "maxNumSentences", "I");
  ans.max_num_sentences = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "silenceScale", "F");
  ans.silence_scale = env->GetFloatField(config, fid);

  return ans;
}

}  // namespace sherpa_mnn

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_mnn_OfflineTts_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  auto config = sherpa_mnn::GetOfflineTtsConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto tts = new sherpa_mnn::OfflineTts(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)tts;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_mnn_OfflineTts_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  return SafeJNI(
      env, "OfflineTts_newFromFile",
      [&]() -> jlong {
        auto config = sherpa_mnn::GetOfflineTtsConfig(env, _config);
        SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

        if (!config.Validate()) {
          SHERPA_ONNX_LOGE("Errors found in config!");
        }

        auto tts = new sherpa_mnn::OfflineTts(config);
        return reinterpret_cast<jlong>(tts);
      },
      0L);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_OfflineTts_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_mnn::OfflineTts *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_mnn_OfflineTts_getSampleRate(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_mnn::OfflineTts *>(ptr)->SampleRate();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_mnn_OfflineTts_getNumSpeakers(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_mnn::OfflineTts *>(ptr)->NumSpeakers();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_OfflineTts_generateImpl(JNIEnv *env, jobject /*obj*/,
                                                   jlong ptr, jstring text,
                                                   jint sid, jfloat speed) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);

  auto audio = reinterpret_cast<sherpa_mnn::OfflineTts *>(ptr)->Generate(
      p_text, sid, speed);

  jfloatArray samples_arr = env->NewFloatArray(audio.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, audio.samples.size(),
                           audio.samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, audio.sample_rate));

  env->ReleaseStringUTFChars(text, p_text);

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_OfflineTts_generateWithCallbackImpl(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring text, jint sid,
    jfloat speed, jobject callback) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);

  std::function<int32_t(const float *, int32_t, float)> callback_wrapper =
      [env, callback](const float *samples, int32_t n,
                      float /*progress*/) -> int {
    jclass cls = env->GetObjectClass(callback);

#if 0
        // this block is for debugging only
        // see also
        // https://jnjosh.com/posts/kotlinfromcpp/
        jmethodID classMethodId =
            env->GetMethodID(cls, "getClass", "()Ljava/lang/Class;");
        jobject klassObj = env->CallObjectMethod(callback, classMethodId);
        auto klassObject = env->GetObjectClass(klassObj);
        auto nameMethodId =
            env->GetMethodID(klassObject, "getName", "()Ljava/lang/String;");
        jstring classString =
            (jstring)env->CallObjectMethod(klassObj, nameMethodId);
        auto className = env->GetStringUTFChars(classString, NULL);
        SHERPA_ONNX_LOGE("name is: %s", className);
        env->ReleaseStringUTFChars(classString, className);
#endif

    jmethodID mid = env->GetMethodID(cls, "invoke", "([F)Ljava/lang/Integer;");
    if (mid == nullptr) {
      SHERPA_ONNX_LOGE("Failed to get the callback. Ignore it.");
      return 1;
    }

    jfloatArray samples_arr = env->NewFloatArray(n);
    env->SetFloatArrayRegion(samples_arr, 0, n, samples);

    jobject should_continue = env->CallObjectMethod(callback, mid, samples_arr);
    jclass jklass = env->GetObjectClass(should_continue);
    jmethodID int_value_mid = env->GetMethodID(jklass, "intValue", "()I");
    return env->CallIntMethod(should_continue, int_value_mid);
  };

  auto tts = reinterpret_cast<sherpa_mnn::OfflineTts *>(ptr);
  auto audio = tts->Generate(p_text, sid, speed, callback_wrapper);

  jfloatArray samples_arr = env->NewFloatArray(audio.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, audio.samples.size(),
                           audio.samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, audio.sample_rate));

  env->ReleaseStringUTFChars(text, p_text);

  return obj_arr;
}
