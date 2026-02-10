// sherpa-mnn/jni/online-recognizer.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-mnn/csrc/online-recognizer.h"

#include "sherpa-mnn/csrc/macros.h"
#include "sherpa-mnn/jni/common.h"

namespace sherpa_mnn {

static OnlineRecognizerConfig GetConfig(JNIEnv *env, jobject config) {
  OnlineRecognizerConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  // https://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/types.html
  // https://courses.cs.washington.edu/courses/cse341/99wi/java/tutorial/native1.1/implementing/field.html

  //---------- decoding ----------
  fid = env->GetFieldID(cls, "decodingMethod", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.decoding_method = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "maxActivePaths", "I");
  ans.max_active_paths = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "hotwordsFile", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.hotwords_file = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "hotwordsScore", "F");
  ans.hotwords_score = env->GetFloatField(config, fid);

  fid = env->GetFieldID(cls, "ruleFsts", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.rule_fsts = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "ruleFars", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.rule_fars = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "blankPenalty", "F");
  ans.blank_penalty = env->GetFloatField(config, fid);

  //---------- feat config ----------
  fid = env->GetFieldID(cls, "featConfig",
                        "Lcom/k2fsa/sherpa/mnn/FeatureConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  fid = env->GetFieldID(feat_config_cls, "sampleRate", "I");
  ans.feat_config.sampling_rate = env->GetIntField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "featureDim", "I");
  ans.feat_config.feature_dim = env->GetIntField(feat_config, fid);

  //---------- enable endpoint ----------
  fid = env->GetFieldID(cls, "enableEndpoint", "Z");
  ans.enable_endpoint = env->GetBooleanField(config, fid);

  //---------- endpoint_config ----------

  fid = env->GetFieldID(cls, "endpointConfig",
                        "Lcom/k2fsa/sherpa/mnn/EndpointConfig;");
  jobject endpoint_config = env->GetObjectField(config, fid);
  jclass endpoint_config_cls = env->GetObjectClass(endpoint_config);

  fid = env->GetFieldID(endpoint_config_cls, "rule1",
                        "Lcom/k2fsa/sherpa/mnn/EndpointRule;");
  jobject rule1 = env->GetObjectField(endpoint_config, fid);
  jclass rule_class = env->GetObjectClass(rule1);

  fid = env->GetFieldID(endpoint_config_cls, "rule2",
                        "Lcom/k2fsa/sherpa/mnn/EndpointRule;");
  jobject rule2 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(endpoint_config_cls, "rule3",
                        "Lcom/k2fsa/sherpa/mnn/EndpointRule;");
  jobject rule3 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(rule_class, "mustContainNonSilence", "Z");
  ans.endpoint_config.rule1.must_contain_nonsilence =
      env->GetBooleanField(rule1, fid);
  ans.endpoint_config.rule2.must_contain_nonsilence =
      env->GetBooleanField(rule2, fid);
  ans.endpoint_config.rule3.must_contain_nonsilence =
      env->GetBooleanField(rule3, fid);

  fid = env->GetFieldID(rule_class, "minTrailingSilence", "F");
  ans.endpoint_config.rule1.min_trailing_silence =
      env->GetFloatField(rule1, fid);
  ans.endpoint_config.rule2.min_trailing_silence =
      env->GetFloatField(rule2, fid);
  ans.endpoint_config.rule3.min_trailing_silence =
      env->GetFloatField(rule3, fid);

  fid = env->GetFieldID(rule_class, "minUtteranceLength", "F");
  ans.endpoint_config.rule1.min_utterance_length =
      env->GetFloatField(rule1, fid);
  ans.endpoint_config.rule2.min_utterance_length =
      env->GetFloatField(rule2, fid);
  ans.endpoint_config.rule3.min_utterance_length =
      env->GetFloatField(rule3, fid);

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

  // paraformer
  fid = env->GetFieldID(model_config_cls, "paraformer",
                        "Lcom/k2fsa/sherpa/mnn/OnlineParaformerModelConfig;");
  jobject paraformer_config = env->GetObjectField(model_config, fid);
  jclass paraformer_config_cls = env->GetObjectClass(paraformer_config);

  fid = env->GetFieldID(paraformer_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(paraformer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.paraformer.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(paraformer_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(paraformer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.paraformer.decoder = p;
  env->ReleaseStringUTFChars(s, p);

  // streaming zipformer2 CTC
  fid =
      env->GetFieldID(model_config_cls, "zipformer2Ctc",
                      "Lcom/k2fsa/sherpa/mnn/OnlineZipformer2CtcModelConfig;");
  jobject zipformer2_ctc_config = env->GetObjectField(model_config, fid);
  jclass zipformer2_ctc_config_cls = env->GetObjectClass(zipformer2_ctc_config);

  fid =
      env->GetFieldID(zipformer2_ctc_config_cls, "model", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(zipformer2_ctc_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.zipformer2_ctc.model = p;
  env->ReleaseStringUTFChars(s, p);

  // streaming NeMo CTC
  fid = env->GetFieldID(model_config_cls, "neMoCtc",
                        "Lcom/k2fsa/sherpa/mnn/OnlineNeMoCtcModelConfig;");
  jobject nemo_ctc_config = env->GetObjectField(model_config, fid);
  jclass nemo_ctc_config_cls = env->GetObjectClass(nemo_ctc_config);

  fid = env->GetFieldID(nemo_ctc_config_cls, "model", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(nemo_ctc_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.nemo_ctc.model = p;
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

  fid = env->GetFieldID(model_config_cls, "modelingUnit", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.modeling_unit = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "bpeVocab", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.bpe_vocab = p;
  env->ReleaseStringUTFChars(s, p);

  //---------- rnn lm model config ----------
  fid = env->GetFieldID(cls, "lmConfig",
                        "Lcom/k2fsa/sherpa/mnn/OnlineLMConfig;");
  jobject lm_model_config = env->GetObjectField(config, fid);
  jclass lm_model_config_cls = env->GetObjectClass(lm_model_config);

  fid = env->GetFieldID(lm_model_config_cls, "model", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(lm_model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.lm_config.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(lm_model_config_cls, "scale", "F");
  ans.lm_config.scale = env->GetFloatField(lm_model_config, fid);

  fid = env->GetFieldID(cls, "ctcFstDecoderConfig",
                        "Lcom/k2fsa/sherpa/mnn/OnlineCtcFstDecoderConfig;");

  jobject fst_decoder_config = env->GetObjectField(config, fid);
  jclass fst_decoder_config_cls = env->GetObjectClass(fst_decoder_config);

  fid = env->GetFieldID(fst_decoder_config_cls, "graph", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(fst_decoder_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.ctc_fst_decoder_config.graph = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(fst_decoder_config_cls, "maxActive", "I");
  ans.ctc_fst_decoder_config.max_active =
      env->GetIntField(fst_decoder_config, fid);

  return ans;
}
}  // namespace sherpa_mnn

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_OnlineRecognizer_newFromAsset(JNIEnv *env,
                                                         jobject /*obj*/,
                                                         jobject asset_manager,
                                                         jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  auto config = sherpa_mnn::GetConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto recognizer = new sherpa_mnn::OnlineRecognizer(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)recognizer;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_mnn_OnlineRecognizer_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config = sherpa_mnn::GetConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto recognizer = new sherpa_mnn::OnlineRecognizer(config);

  return (jlong)recognizer;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_OnlineRecognizer_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_mnn::OnlineRecognizer *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_OnlineRecognizer_reset(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto recognizer = reinterpret_cast<sherpa_mnn::OnlineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);
  recognizer->Reset(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_mnn_OnlineRecognizer_isReady(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto recognizer = reinterpret_cast<sherpa_mnn::OnlineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);

  return recognizer->IsReady(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_mnn_OnlineRecognizer_isEndpoint(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto recognizer = reinterpret_cast<sherpa_mnn::OnlineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);

  return recognizer->IsEndpoint(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_mnn_OnlineRecognizer_decode(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto recognizer = reinterpret_cast<sherpa_mnn::OnlineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);

  recognizer->DecodeStream(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_OnlineRecognizer_createStream(JNIEnv *env,
                                                         jobject /*obj*/,
                                                         jlong ptr,
                                                         jstring hotwords) {
  auto recognizer = reinterpret_cast<sherpa_mnn::OnlineRecognizer *>(ptr);

  const char *p = env->GetStringUTFChars(hotwords, nullptr);
  std::unique_ptr<sherpa_mnn::OnlineStream> stream;

  if (strlen(p) == 0) {
    stream = recognizer->CreateStream();
  } else {
    stream = recognizer->CreateStream(p);
  }

  env->ReleaseStringUTFChars(hotwords, p);

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_mnn_OfflineStream_delete() from
  // ./offline-stream.cc
  sherpa_mnn::OnlineStream *ans = stream.release();
  return (jlong)ans;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_OnlineRecognizer_getResult(JNIEnv *env,
                                                      jobject /*obj*/,
                                                      jlong ptr,
                                                      jlong stream_ptr) {
  auto recognizer = reinterpret_cast<sherpa_mnn::OnlineRecognizer *>(ptr);
  auto stream = reinterpret_cast<sherpa_mnn::OnlineStream *>(stream_ptr);

  sherpa_mnn::OnlineRecognizerResult result = recognizer->GetResult(stream);

  // [0]: text, jstring
  // [1]: tokens, array of jstring
  // [2]: timestamps, array of float
  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      3, env->FindClass("java/lang/Object"), nullptr);

  jstring text = env->NewStringUTF(result.text.c_str());
  env->SetObjectArrayElement(obj_arr, 0, text);

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
