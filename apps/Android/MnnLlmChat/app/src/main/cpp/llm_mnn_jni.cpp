#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>
#include <ostream>
#include <sstream>
#include <chrono>
#include "mls_log.h"
#include "MNN/expr/ExecutorScope.hpp"
#include "nlohmann/json.hpp"
#include "llm_stream_buffer.hpp"
#include "utf8_stream_processor.hpp"
#include "llm_session.h"

using MNN::Transformer::Llm;
using json = nlohmann::json;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnUnload");
}

JNIEXPORT jlong JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_initNative(JNIEnv *env,
                                                                                  jobject thiz,
                                                                                  jstring modelDir,
                                                                                  jobject chat_history,
                                                                                  jstring mergeConfigStr,
                                                                                  jstring configJsonStr) {
    const char *model_dir = env->GetStringUTFChars(modelDir, nullptr);
    auto model_dir_str = std::string(model_dir);
    const char *config_json_cstr = env->GetStringUTFChars(configJsonStr, nullptr);
    const char *merged_config_cstr = env->GetStringUTFChars(mergeConfigStr, nullptr);
    json merged_config = json::parse(merged_config_cstr);
    json extra_json_config = json::parse(config_json_cstr);
    env->ReleaseStringUTFChars(modelDir, model_dir);
    env->ReleaseStringUTFChars(configJsonStr, config_json_cstr);
    env->ReleaseStringUTFChars(mergeConfigStr, merged_config_cstr);
    MNN_DEBUG("createLLM BeginLoad %s", model_dir);
    std::vector<std::string> history;
    history.clear();
    if (chat_history != nullptr) {
        jclass listClass = env->GetObjectClass(chat_history);
        jmethodID sizeMethod = env->GetMethodID(listClass, "size", "()I");
        jmethodID getMethod = env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");
        jint listSize = env->CallIntMethod(chat_history, sizeMethod);
        for (jint i = 0; i < listSize; i++) {
            jobject element = env->CallObjectMethod(chat_history, getMethod, i);
            const char *elementCStr = env->GetStringUTFChars((jstring) element, nullptr);
            history.emplace_back(elementCStr);
            env->ReleaseStringUTFChars((jstring) element, elementCStr);
            env->DeleteLocalRef(element);
        }
    }
    auto llm_session = new mls::LlmSession(model_dir_str, merged_config, extra_json_config,
                                           history);
    llm_session->Load();
    MNN_DEBUG("LIFECYCLE: LlmSession CREATED at %p", llm_session);
    MNN_DEBUG("createLLM EndLoad %ld ", reinterpret_cast<jlong>(llm_session));
    return reinterpret_cast<jlong>(llm_session);
}


JNIEXPORT jobject JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_submitNative(JNIEnv *env,
                                                                                      jobject thiz,
                                                                                      jlong llmPtr,
                                                                                      jstring inputStr,
                                                                                      jboolean keepHistory,
                                                                                      jobject
                                                                                      progressListener) {
    auto *llm = reinterpret_cast<mls::LlmSession *>(llmPtr);
    if (!llm) {
        return env->NewStringUTF("Failed, Chat is not ready!");
    }
    const char *input_str = env->GetStringUTFChars(inputStr, nullptr);
    jclass progressListenerClass = env->GetObjectClass(progressListener);
    jmethodID onProgressMethod = env->GetMethodID(progressListenerClass, "onProgress",
                                                  "(Ljava/lang/String;)Z");
    if (!onProgressMethod) {
        MNN_DEBUG("ProgressListener onProgress method not found.");
    }
    auto *context = llm->Response(input_str, [&, progressListener, onProgressMethod](
            const std::string &response, bool is_eop) {
        if (progressListener && onProgressMethod) {
            jstring javaString = is_eop ? nullptr : env->NewStringUTF(response.c_str());
            jboolean user_stop_requested = env->CallBooleanMethod(progressListener,
                                                                  onProgressMethod, javaString);
            env->DeleteLocalRef(javaString);
            return (bool) user_stop_requested;
        } else {
            return true;
        }
    });
    int64_t prompt_len = 0;
    int64_t decode_len = 0;
    int64_t vision_time = 0;
    int64_t audio_time = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    prompt_len += context->prompt_len;
    decode_len += context->gen_seq_len;
    vision_time += context->vision_us;
    audio_time += context->audio_us;
    prefill_time += context->prefill_us;
    decode_time += context->decode_us;
    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
    jmethodID putMethod = env->GetMethodID(hashMapClass, "put",
                                           "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject hashMap = env->NewObject(hashMapClass, hashMapInit);

    // Add metrics to the HashMap
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("prompt_len"),
                          env->NewObject(env->FindClass("java/lang/Long"),
                                         env->GetMethodID(env->FindClass("java/lang/Long"),
                                                          "<init>", "(J)V"), prompt_len));
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("decode_len"),
                          env->NewObject(env->FindClass("java/lang/Long"),
                                         env->GetMethodID(env->FindClass("java/lang/Long"),
                                                          "<init>", "(J)V"), decode_len));
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("vision_time"),
                          env->NewObject(env->FindClass("java/lang/Long"),
                                         env->GetMethodID(env->FindClass("java/lang/Long"),
                                                          "<init>", "(J)V"), vision_time));
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("audio_time"),
                          env->NewObject(env->FindClass("java/lang/Long"),
                                         env->GetMethodID(env->FindClass("java/lang/Long"),
                                                          "<init>", "(J)V"), audio_time));
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("prefill_time"),
                          env->NewObject(env->FindClass("java/lang/Long"),
                                         env->GetMethodID(env->FindClass("java/lang/Long"),
                                                          "<init>", "(J)V"), prefill_time));
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("decode_time"),
                          env->NewObject(env->FindClass("java/lang/Long"),
                                         env->GetMethodID(env->FindClass("java/lang/Long"),
                                                          "<init>", "(J)V"), decode_time));
    return hashMap;
}



// 新增：支持完整历史消息的JNI方法
JNIEXPORT jobject JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_submitFullHistoryNative(
        JNIEnv *env,
        jobject thiz,
        jlong llmPtr,
        jobject historyList,  // List<Pair<String, String>>
        jobject progressListener
) {
    auto *llm = reinterpret_cast<mls::LlmSession *>(llmPtr);
    if (!llm) {
        return env->NewStringUTF("Failed, Chat is not ready!");
    }

    // 解析 Java List<Pair<String, String>> 到 C++ vector
    std::vector<mls::PromptItem> history;

    // 获取List类和相关方法
    jclass listClass = env->GetObjectClass(historyList);
    jmethodID sizeMethod = env->GetMethodID(listClass, "size", "()I");
    jmethodID getMethod = env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");

    jint listSize = env->CallIntMethod(historyList, sizeMethod);

    // 获取 Pair 类和相关方法，修改为 android.util.Pair
    jclass pairClass = env->FindClass("android/util/Pair");
    if (pairClass == nullptr) {
        MNN_DEBUG("Failed to find android.util.Pair class");
        return env->NewStringUTF("Failed to find android.util.Pair class");
    }
    // 使用 GetFieldID 访问 first 字段
    jfieldID firstField = env->GetFieldID(pairClass, "first", "Ljava/lang/Object;");
    // 使用 GetFieldID 访问 second 字段
    jfieldID secondField = env->GetFieldID(pairClass, "second", "Ljava/lang/Object;");

    // 遍历List，提取每个Pair
    for (jint i = 0; i < listSize; i++) {
        jobject pairObj = env->CallObjectMethod(historyList, getMethod, i);
        if (pairObj == nullptr) {
            continue;
        }
        // 使用 GetObjectField 访问 first 字段
        jobject roleObj = env->GetObjectField(pairObj, firstField);
        // 使用 GetObjectField 访问 second 字段
        jobject contentObj = env->GetObjectField(pairObj, secondField);

        const char *role = nullptr;
        const char *content = nullptr;
        if (roleObj != nullptr) {
            role = env->GetStringUTFChars((jstring) roleObj, nullptr);
        }
        if (contentObj != nullptr) {
            content = env->GetStringUTFChars((jstring) contentObj, nullptr);
        }

        if (role && content) {
            history.emplace_back(std::string(role), std::string(content));
        }

        if (role) {
            env->ReleaseStringUTFChars((jstring) roleObj, role);
        }
        if (content) {
            env->ReleaseStringUTFChars((jstring) contentObj, content);
        }
        env->DeleteLocalRef(pairObj);
        if (roleObj) {
            env->DeleteLocalRef(roleObj);
        }
        if (contentObj) {
            env->DeleteLocalRef(contentObj);
        }
    }

    // 设置进度回调
    jclass progressListenerClass = env->GetObjectClass(progressListener);
    jmethodID onProgressMethod = env->GetMethodID(progressListenerClass, "onProgress",
                                                  "(Ljava/lang/String;)Z");

    if (!onProgressMethod) {
        MNN_DEBUG("ProgressListener onProgress method not found.");
    }

    // 调用API服务推理方法
    auto *context = llm->ResponseWithHistory(history, [&, progressListener, onProgressMethod](
            const std::string &response, bool is_eop) {
        if (progressListener && onProgressMethod) {
            jstring javaString = is_eop ? nullptr : env->NewStringUTF(response.c_str());
            jboolean user_stop_requested = env->CallBooleanMethod(progressListener,
                                                                  onProgressMethod, javaString);
            if (javaString) {
                env->DeleteLocalRef(javaString);
            }
            return (bool) user_stop_requested;
        } else {
            return true;
        }
    });

    // 构建返回结果
    int64_t prompt_len = 0;
    int64_t decode_len = 0;
    int64_t vision_time = 0;
    int64_t audio_time = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;

    if (context) {
        prompt_len += context->prompt_len;
        decode_len += context->gen_seq_len;
        vision_time += context->vision_us;
        audio_time += context->audio_us;
        prefill_time += context->prefill_us;
        decode_time += context->decode_us;
    }

    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
    jmethodID hashMapPut = env->GetMethodID(hashMapClass, "put",
                                            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject hashMap = env->NewObject(hashMapClass, hashMapInit);

    // 添加统计信息
    jclass longClass = env->FindClass("java/lang/Long");
    jmethodID longInit = env->GetMethodID(longClass, "<init>", "(J)V");

    env->CallObjectMethod(hashMap, hashMapPut, env->NewStringUTF("prompt_len"),
                          env->NewObject(longClass, longInit, prompt_len));
    env->CallObjectMethod(hashMap, hashMapPut, env->NewStringUTF("decode_len"),
                          env->NewObject(longClass, longInit, decode_len));
    env->CallObjectMethod(hashMap, hashMapPut, env->NewStringUTF("vision_time"),
                          env->NewObject(longClass, longInit, vision_time));
    env->CallObjectMethod(hashMap, hashMapPut, env->NewStringUTF("audio_time"),
                          env->NewObject(longClass, longInit, audio_time));
    env->CallObjectMethod(hashMap, hashMapPut, env->NewStringUTF("prefill_time"),
                          env->NewObject(longClass, longInit, prefill_time));
    env->CallObjectMethod(hashMap, hashMapPut, env->NewStringUTF("decode_time"),
                          env->NewObject(longClass, longInit, decode_time));

    return hashMap;
}



JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_resetNative(JNIEnv *env, jobject thiz,
                                                           jlong object_ptr) {
    auto *llm = reinterpret_cast<mls::LlmSession *>(object_ptr);
    if (llm) {
        MNN_DEBUG("RESET");
        llm->Reset();
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_setWavformCallbackNative(
        JNIEnv *env, jobject thiz, jlong instance_id, jobject listener) {

    if (instance_id == 0 || !listener) {
        return JNI_FALSE;
    }
    auto *session = reinterpret_cast<mls::LlmSession *>(instance_id);
    jobject global_ref = env->NewGlobalRef(listener);
    JavaVM *jvm;
    env->GetJavaVM(&jvm);
    session->SetWavformCallback(
            [jvm, global_ref](const float *data, size_t size, bool is_end) -> bool {
                bool needDetach = false;
                JNIEnv *env;
                if (jvm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
                    jvm->AttachCurrentThread(&env, NULL);
                    needDetach = true;
                }
                jvm->AttachCurrentThread(&env, NULL);
                jclass listenerClass = env->GetObjectClass(global_ref);
                jmethodID onAudioDataMethod = env->GetMethodID(listenerClass, "onAudioData",
                                                               "([FZ)Z");
                jfloatArray audioDataArray = env->NewFloatArray(size);
                env->SetFloatArrayRegion(audioDataArray, 0, size, data);
                jboolean result = env->CallBooleanMethod(global_ref, onAudioDataMethod,
                                                         audioDataArray, is_end);
                env->DeleteLocalRef(audioDataArray);
                env->DeleteLocalRef(listenerClass);
                if (needDetach) {
                    jvm->DetachCurrentThread();
                }
                return result == JNI_TRUE;
            });

    return JNI_TRUE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_getDebugInfoNative(JNIEnv *env, jobject thiz,
                                                                  jlong objecPtr) {
    auto *llm = reinterpret_cast<mls::LlmSession *>(objecPtr);
    if (llm == nullptr) {
        return env->NewStringUTF("");
    }
    return env->NewStringUTF(llm->getDebugInfo().c_str());
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_releaseNative(JNIEnv *env,
                                                                                    jobject thiz,
                                                                                    jlong objecPtr) {
    MNN_DEBUG("LIFECYCLE: About to DESTROY LlmSession at %p", reinterpret_cast<void*>(objecPtr));
    auto *llm = reinterpret_cast<mls::LlmSession *>(objecPtr);
    delete llm;
    MNN_DEBUG("LIFECYCLE: LlmSessionInner DESTROYED at %p", reinterpret_cast<void*>(objecPtr));
}

extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_updateMaxNewTokensNative(JNIEnv *env, jobject thiz,
                                                                        jlong llm_ptr,
                                                                        jint max_new_tokens) {
    auto *llm = reinterpret_cast<mls::LlmSession *>(llm_ptr);
    if (llm) {
        llm->SetMaxNewTokens(max_new_tokens);
    }

}
extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_updateSystemPromptNative(JNIEnv *env, jobject thiz,
                                                                        jlong llm_ptr,
                                                                        jstring system_promp_j) {
    auto *llm = reinterpret_cast<mls::LlmSession *>(llm_ptr);
    const char *system_prompt_cstr = env->GetStringUTFChars(system_promp_j, nullptr);
    if (llm) {
        llm->setSystemPrompt(system_prompt_cstr);
    }
    env->ReleaseStringUTFChars(system_promp_j, system_prompt_cstr);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_updateAssistantPromptNative(JNIEnv *env,
                                                                           jobject thiz,
                                                                           jlong llm_ptr,
                                                                           jstring assistant_prompt_j) {
    auto *llm = reinterpret_cast<mls::LlmSession *>(llm_ptr);
    const char *assistant_prompt_cstr = env->GetStringUTFChars(assistant_prompt_j, nullptr);
    if (llm) {
        llm->SetAssistantPrompt(assistant_prompt_cstr);
    }
    env->ReleaseStringUTFChars(assistant_prompt_j, assistant_prompt_cstr);
}
}
extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_updateEnableAudioOutputNative(JNIEnv *env,jobject thiz, jlong llm_ptr, jboolean enable) {
    auto *llm = reinterpret_cast<mls::LlmSession *>(llm_ptr);
    if (llm) {
        llm->enableAudioOutput((bool)enable);
    }
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_getSystemPromptNative(JNIEnv *env, jobject thiz,
                                                                     jlong llm_ptr) {
    auto *llm = reinterpret_cast<mls::LlmSession *>(llm_ptr);
    if (llm) {
        std::string system_prompt = llm->getSystemPrompt();
        return env->NewStringUTF(system_prompt.c_str());
    }
    return nullptr;
}


extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_clearHistoryNative(JNIEnv *env, jobject thiz,
                                                                  jlong llm_ptr) {
    auto *llm = reinterpret_cast<mls::LlmSession *>(llm_ptr);
    if (llm) {
        llm->clearHistory();
    }
}

// JNI glue layer - converts Java objects and delegates to pure C++ implementation
extern "C"
JNIEXPORT jobject JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_runBenchmarkNative(
        JNIEnv *env,
        jobject thiz,
        jlong llmPtr,
        jint backend,
        jint threads,
        jboolean useMmap,
        jint power,
        jint precision,
        jint memory,
        jint dynamicOption,
        jint nPrompt,
        jint nGenerate,
        jint nRepeat,
        jboolean kvCache,
        jobject testInstance,
        jobject callback
) {
    auto *llm_session = reinterpret_cast<mls::LlmSession *>(llmPtr);
    if (!llm_session) {
        // Return failure result
        jclass resultClass = env->FindClass("com/alibaba/mnnllm/android/benchmark/BenchmarkResult");
        jmethodID resultCtor = env->GetMethodID(resultClass, "<init>", 
            "(Lcom/alibaba/mnnllm/android/benchmark/TestInstance;ZLjava/lang/String;)V");
        return env->NewObject(resultClass, resultCtor, testInstance, JNI_FALSE, 
            env->NewStringUTF("LLM session is not initialized"));
    }

    // Debug: Print pointer values for analysis
    MNN_DEBUG("BENCHMARK: Attempting to use LlmSession at %p", llm_session);
    
    // CRITICAL: The real issue might be that llm_session object has been destroyed
    // but the pointer still points to the freed memory. Let's add extensive logging
    // to track the object lifecycle and try to detect use-after-free conditions.
    
    // Try to safely access the object - if this crashes, we know it's use-after-free
    MNN_DEBUG("BENCHMARK: About to call getLlm() on %p", llm_session);
    
    auto *underlying_llm = llm_session->getLlm();
    MNN_DEBUG("BENCHMARK: getLlm() returned %p", underlying_llm);
    
    // Now try to call the actual benchmark - this is where the crash likely occurs
    MNN_DEBUG("BENCHMARK: About to call runBenchmark() - this may crash");
    MNN_DEBUG("BENCHMARK: Parameters - nPrompt=%d, nGenerate=%d, nRepeat=%d, kvCache=%s", 
              (int)nPrompt, (int)nGenerate, (int)nRepeat, kvCache ? "true" : "false");

    // Create global references for callback usage in pure C++ code
    jobject globalCallback = env->NewGlobalRef(callback);
    jobject globalTestInstance = env->NewGlobalRef(testInstance);
    
    // Get callback method IDs
    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onProgressMethod = env->GetMethodID(callbackClass, "onProgress", 
        "(Lcom/alibaba/mnnllm/android/benchmark/BenchmarkProgress;)V");
    jmethodID onErrorMethod = env->GetMethodID(callbackClass, "onBenchmarkError", "(ILjava/lang/String;)V");
    
    // Get BenchmarkProgress class and constructor - updated for structured data
    jclass progressClass = env->FindClass("com/alibaba/mnnllm/android/benchmark/BenchmarkProgress");
    jmethodID progressCtor = env->GetMethodID(progressClass, "<init>", 
        "(ILjava/lang/String;IIIIIFFFFF)V");
    
    // Get TestInstance field accessor methods for updating results
    jclass testInstanceClass = env->GetObjectClass(testInstance);
    jmethodID getPrefillUsMethod = env->GetMethodID(testInstanceClass, "getPrefillUs", "()Ljava/util/List;");
    if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception: GetMethodID for getPrefillUs failed."); env->ExceptionDescribe(); env->ExceptionClear(); }
    jmethodID getDecodeUsMethod = env->GetMethodID(testInstanceClass, "getDecodeUs", "()Ljava/util/List;");
    if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception: GetMethodID for getDecodeUs failed."); env->ExceptionDescribe(); env->ExceptionClear(); }
    jmethodID getSamplesUsMethod = env->GetMethodID(testInstanceClass, "getSamplesUs", "()Ljava/util/List;");
    if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception: GetMethodID for getSamplesUs failed."); env->ExceptionDescribe(); env->ExceptionClear(); }

    // Get List.add method for updating results
    jclass listClass = env->FindClass("java/util/List");
    if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception: FindClass for java/util/List failed."); env->ExceptionDescribe(); env->ExceptionClear(); }
    jmethodID addMethod = env->GetMethodID(listClass, "add", "(Ljava/lang/Object;)Z");
    if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception: GetMethodID for List.add failed."); env->ExceptionDescribe(); env->ExceptionClear(); }
    
    // Get the mutable lists from TestInstance
    jobject prefillUsList = env->CallObjectMethod(testInstance, getPrefillUsMethod);
    if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception: CallObjectMethod for getPrefillUs failed."); env->ExceptionDescribe(); env->ExceptionClear(); }
    jobject decodeUsList = env->CallObjectMethod(testInstance, getDecodeUsMethod);
    if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception: CallObjectMethod for getDecodeUs failed."); env->ExceptionDescribe(); env->ExceptionClear(); }
    jobject samplesUsList = env->CallObjectMethod(testInstance, getSamplesUsMethod);
    if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception: CallObjectMethod for getSamplesUs failed."); env->ExceptionDescribe(); env->ExceptionClear(); }

    // Long class for boxing
    jclass longClass = env->FindClass("java/lang/Long");
    if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception: FindClass for java/lang/Long failed."); env->ExceptionDescribe(); env->ExceptionClear(); }
    jmethodID longCtor = env->GetMethodID(longClass, "<init>", "(J)V");
    if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception: GetMethodID for Long constructor failed."); env->ExceptionDescribe(); env->ExceptionClear(); }

    // Create callback adapter for pure C++ code
    mls::LlmSession::BenchmarkCallback cppCallback;
    
    cppCallback.onProgress = [env, globalCallback, onProgressMethod, progressClass, progressCtor](
        const mls::LlmSession::BenchmarkProgressInfo& progressInfo) {
        if (onProgressMethod && progressClass && progressCtor) {
            jobject progressObj = env->NewObject(progressClass, progressCtor, 
                (jint)progressInfo.progress, 
                env->NewStringUTF(progressInfo.statusMessage.c_str()),
                (jint)progressInfo.progressType,
                (jint)progressInfo.currentIteration,
                (jint)progressInfo.totalIterations,
                (jint)progressInfo.nPrompt,
                (jint)progressInfo.nGenerate,
                (jfloat)progressInfo.runTimeSeconds,
                (jfloat)progressInfo.prefillTimeSeconds,
                (jfloat)progressInfo.decodeTimeSeconds,
                (jfloat)progressInfo.prefillSpeed,
                (jfloat)progressInfo.decodeSpeed);
            env->CallVoidMethod(globalCallback, onProgressMethod, progressObj);
            env->DeleteLocalRef(progressObj);
        }
    };
    
    cppCallback.onError = [env, globalCallback, onErrorMethod](const std::string& error) {
        if (onErrorMethod) {
            env->CallVoidMethod(globalCallback, onErrorMethod, 0, env->NewStringUTF(error.c_str()));
        }
    };
    
    cppCallback.onIterationComplete = [env, globalCallback, onProgressMethod, progressClass, progressCtor](
        const std::string& detailed_stats) {
        if (onProgressMethod && progressClass && progressCtor) {
            // Report detailed stats as progress update with minimal structured data
            jobject progressObj = env->NewObject(progressClass, progressCtor, 
                (jint)-1, env->NewStringUTF(detailed_stats.c_str()), // -1 indicates detailed stats
                (jint)0, (jint)0, (jint)0, (jint)0, (jint)0, // default values for structured fields
                (jfloat)0.0f, (jfloat)0.0f, (jfloat)0.0f, (jfloat)0.0f, (jfloat)0.0f);
            env->CallVoidMethod(globalCallback, onProgressMethod, progressObj);
            env->DeleteLocalRef(progressObj);
        }
    };

    // Call pure C++ benchmark implementation
    auto result = llm_session->runBenchmark(
        (int)backend, (int)threads, (bool)useMmap, (int)power, (int)precision, 
        (int)memory, (int)dynamicOption, (int)nPrompt, (int)nGenerate, 
        (int)nRepeat, (bool)kvCache, cppCallback
    );
    
    // If we reach here, runBenchmark completed without crashing
    MNN_DEBUG("BENCHMARK: runBenchmark() completed successfully! Result success=%s", 
              result.success ? "true" : "false");
    
    // Add detailed logging before returning to Kotlin
    MNN_DEBUG("BENCHMARK_JNI: Handing off to Kotlin. Prefill times count: %d, Decode times count: %d",
              (int)result.prefill_times_us.size(),
              (int)result.decode_times_us.size());

    // Convert C++ result to Java objects and update TestInstance
    if (result.success) {
        if (!prefillUsList || !decodeUsList || !samplesUsList) {
            MNN_ERROR("JNI Error: One or more list objects are null!");
        } else {
            // Update TestInstance lists with timing data
            for (long prefill_time : result.prefill_times_us) {
                jobject prefillLong = env->NewObject(longClass, longCtor, prefill_time);
                if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception after NewObject for prefill."); break; }
                env->CallBooleanMethod(prefillUsList, addMethod, prefillLong);
                if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception after add for prefill."); break; }
                env->DeleteLocalRef(prefillLong);
            }
            
            for (long decode_time : result.decode_times_us) {
                jobject decodeLong = env->NewObject(longClass, longCtor, decode_time);
                if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception after NewObject for decode."); break; }
                env->CallBooleanMethod(decodeUsList, addMethod, decodeLong);
                if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception after add for decode."); break; }
                env->DeleteLocalRef(decodeLong);
            }
            
            for (long sample_time : result.sample_times_us) {
                jobject sampleLong = env->NewObject(longClass, longCtor, sample_time);
                if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception after NewObject for sample."); break; }
                env->CallBooleanMethod(samplesUsList, addMethod, sampleLong);
                if (env->ExceptionCheck()) { MNN_ERROR("JNI Exception after add for sample."); break; }
                env->DeleteLocalRef(sampleLong);
            }
        }
    }
    
    // Cleanup global references
    env->DeleteGlobalRef(globalCallback);
    env->DeleteGlobalRef(globalTestInstance);
    
    // Create and return BenchmarkResult
    jclass resultClass = env->FindClass("com/alibaba/mnnllm/android/benchmark/BenchmarkResult");
    jmethodID resultCtor = env->GetMethodID(resultClass, "<init>", 
        "(Lcom/alibaba/mnnllm/android/benchmark/TestInstance;ZLjava/lang/String;)V");
    
    jstring errorMessage = result.success ? nullptr : env->NewStringUTF(result.error_message.c_str());
    return env->NewObject(resultClass, resultCtor, testInstance, (jboolean)result.success, errorMessage);
}

