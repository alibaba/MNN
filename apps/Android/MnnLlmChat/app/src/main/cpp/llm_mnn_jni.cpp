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
#include <mutex>
#include <ostream>
#include <sstream>
#include <mutex>
#include <string>
#include "diffusion_session.h"
#include <chrono>
#include "mls_log.h"
#include "MNN/expr/ExecutorScope.hpp"
#include "nlohmann/json.hpp"
#include "llm_stream_buffer.hpp"
#include "utf8_stream_processor.hpp"
#include "llm_session.h"

using MNN::Transformer::Llm;
using mls::DiffusionSession;
using json = nlohmann::json;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnUnload");
}

JNIEXPORT jlong JNICALL Java_com_alibaba_mnnllm_android_ChatSession_initNative(JNIEnv* env,
                                                                                    jobject thiz,
                                                                                    jstring modelDir,
                                                                                    jobject chat_history,
                                                                                    jstring mergeConfigStr,
                                                                                    jstring configJsonStr) {
    const char* model_dir = env->GetStringUTFChars(modelDir, nullptr);
    auto model_dir_str = std::string(model_dir);
    const char* config_json_cstr = env->GetStringUTFChars(configJsonStr, nullptr);
    const char* merged_config_cstr = env->GetStringUTFChars(mergeConfigStr, nullptr);
    json merged_config  = json::parse(merged_config_cstr);
    json extra_json_config = json::parse(config_json_cstr);
    bool is_diffusion = extra_json_config["is_diffusion"];
    env->ReleaseStringUTFChars(modelDir, model_dir);
    env->ReleaseStringUTFChars(configJsonStr, config_json_cstr);
    env->ReleaseStringUTFChars(mergeConfigStr, merged_config_cstr);
    if (is_diffusion) {
        std::string diffusion_memory_mode = extra_json_config["diffusion_memory_mode"];
        int diffusion_memory_mode_int = std::stoi(diffusion_memory_mode);
        auto diffusion = new DiffusionSession(model_dir, diffusion_memory_mode_int);
        return reinterpret_cast<jlong>(diffusion);
    }
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
            const char *elementCStr = env->GetStringUTFChars((jstring)element, nullptr);
            history.emplace_back(elementCStr);
            env->ReleaseStringUTFChars((jstring)element, elementCStr);
            env->DeleteLocalRef(element);
        }
    }
    auto llm_session = new mls::LlmSession(model_dir_str, merged_config, extra_json_config, history);
    llm_session->Load();
    MNN_DEBUG("createLLM EndLoad %ld ", reinterpret_cast<jlong>(llm_session));
    return reinterpret_cast<jlong>(llm_session);
}


JNIEXPORT jobject JNICALL Java_com_alibaba_mnnllm_android_ChatSession_submitNative(JNIEnv* env,
                                                                                   jobject thiz,
                                                                                   jlong llmPtr,
                                                                                   jstring inputStr,
                                                                                   jboolean keepHistory,
                                                                                   jobject
                                                                                   progressListener) {
    auto* llm = reinterpret_cast<mls::LlmSession*>(llmPtr);
    if (!llm) {
        return env->NewStringUTF("Failed, Chat is not ready!");
    }
    const char* input_str = env->GetStringUTFChars(inputStr, nullptr);
    jclass progressListenerClass = env->GetObjectClass(progressListener);
    jmethodID onProgressMethod = env->GetMethodID(progressListenerClass, "onProgress", "(Ljava/lang/String;)Z");
    if (!onProgressMethod) {
        MNN_DEBUG("ProgressListener onProgress method not found.");
    }
    auto* context = llm->Response(input_str, [&, progressListener, onProgressMethod](const std::string& response, bool is_eop) {
        if (progressListener && onProgressMethod) {
            jstring javaString = is_eop ? nullptr : env->NewStringUTF(response.c_str());
            jboolean user_stop_requested = env->CallBooleanMethod(progressListener, onProgressMethod,  javaString);
            env->DeleteLocalRef(javaString);
            return (bool)user_stop_requested;
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
    jmethodID putMethod = env->GetMethodID(hashMapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject hashMap = env->NewObject(hashMapClass, hashMapInit);

    // Add metrics to the HashMap
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("prompt_len"), env->NewObject(env->FindClass("java/lang/Long"), env->GetMethodID(env->FindClass("java/lang/Long"), "<init>", "(J)V"), prompt_len));
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("decode_len"), env->NewObject(env->FindClass("java/lang/Long"), env->GetMethodID(env->FindClass("java/lang/Long"), "<init>", "(J)V"), decode_len));
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("vision_time"), env->NewObject(env->FindClass("java/lang/Long"), env->GetMethodID(env->FindClass("java/lang/Long"), "<init>", "(J)V"), vision_time));
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("audio_time"), env->NewObject(env->FindClass("java/lang/Long"), env->GetMethodID(env->FindClass("java/lang/Long"), "<init>", "(J)V"), audio_time));
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("prefill_time"), env->NewObject(env->FindClass("java/lang/Long"), env->GetMethodID(env->FindClass("java/lang/Long"), "<init>", "(J)V"), prefill_time));
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("decode_time"), env->NewObject(env->FindClass("java/lang/Long"), env->GetMethodID(env->FindClass("java/lang/Long"), "<init>", "(J)V"), decode_time));
    return hashMap;
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_ChatSession_resetNative(JNIEnv* env,
                                                                               jobject thiz,
                                                                               jlong object_ptr,
                                                                               jboolean is_diffusion) {
    if (is_diffusion) {
        return;
    }
    auto* llm = reinterpret_cast<mls::LlmSession*>(object_ptr);
    if (llm) {
        MNN_DEBUG("RESET");
        llm->Reset();
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_alibaba_mnnllm_android_ChatSession_setWavformCallbackNative(
        JNIEnv *env, jobject thiz, jlong instance_id, jobject listener) {

    if (instance_id == 0 || !listener) {
        return JNI_FALSE;
    }
    auto *session = reinterpret_cast<mls::LlmSession *>(instance_id);
    jobject global_ref = env->NewGlobalRef(listener);
    session->SetWavformCallback([env, global_ref](const float* data, size_t size, bool is_end) -> bool {
        jclass listenerClass = env->GetObjectClass(global_ref);
        jmethodID onAudioDataMethod = env->GetMethodID(listenerClass, "onAudioData", "([FZ)Z");
        jfloatArray audioDataArray = env->NewFloatArray(size);
        env->SetFloatArrayRegion(audioDataArray, 0, size, data);
        jboolean result = env->CallBooleanMethod(global_ref, onAudioDataMethod, audioDataArray, is_end);
        env->DeleteLocalRef(audioDataArray);
        env->DeleteLocalRef(listenerClass);

        return result == JNI_TRUE;
    });

    return JNI_TRUE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_alibaba_mnnllm_android_ChatSession_getDebugInfoNative(JNIEnv *env, jobject thiz, jlong objecPtr) {
    auto* llm = reinterpret_cast<mls::LlmSession*>(objecPtr);
    if (llm == nullptr) {
        return env->NewStringUTF("");
    }
    return env->NewStringUTF(llm->getDebugInfo().c_str());
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_ChatSession_releaseNative(JNIEnv* env,
                                                                                      jobject thiz,
                                                                                      jlong objecPtr,
                                                                                      jboolean isDiffusion) {
    MNN_DEBUG("Java_com_alibaba_mnnllm_android_ChatSession_releaseNative\n");
    if (isDiffusion) {
        auto* diffusion = reinterpret_cast<DiffusionSession*>(objecPtr);
        delete diffusion;
    } else {
        auto* llm = reinterpret_cast<mls::LlmSession*>(objecPtr);
        delete llm;
    }
}

JNIEXPORT jobject JNICALL
Java_com_alibaba_mnnllm_android_ChatSession_submitDiffusionNative(JNIEnv *env, jobject thiz,
                                                                       jlong instance_id,
                                                                       jstring input,
                                                                       jstring joutput_path,
                                                                       jint iter_num,
                                                                       jint random_seed,
                                                                       jobject progressListener) {
    auto* diffusion = reinterpret_cast<DiffusionSession*>(instance_id); // Cast back to Llm*
    if (!diffusion) {
        return nullptr;
    }
    jclass progressListenerClass = env->GetObjectClass(progressListener);
    jmethodID onProgressMethod = env->GetMethodID(progressListenerClass, "onProgress", "(Ljava/lang/String;)Z");
    if (!onProgressMethod) {
        MNN_DEBUG("ProgressListener onProgress method not found.");
    }
    std::string prompt = env->GetStringUTFChars(input, nullptr);
    std::string output_path = env->GetStringUTFChars(joutput_path, nullptr);
    auto start = std::chrono::high_resolution_clock::now();
    diffusion->Run(prompt,
                   output_path,
                   iter_num,
                   random_seed,
                   [env, progressListener, onProgressMethod](int progress) {
        if (progressListener && onProgressMethod) {
            jstring javaString =  env->NewStringUTF(std::to_string(progress).c_str());
            env->CallBooleanMethod(progressListener, onProgressMethod,  javaString);
            env->DeleteLocalRef(javaString);
        }
    });
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
    jmethodID putMethod = env->GetMethodID(hashMapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject hashMap = env->NewObject(hashMapClass, hashMapInit);
    env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("total_timeus"), env->NewObject(env->FindClass("java/lang/Long"), env->GetMethodID(env->FindClass("java/lang/Long"), "<init>", "(J)V"), duration));
    return hashMap;
}
}

extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_ChatSession_updateMaxNewTokensNative(JNIEnv *env, jobject thiz,
                                                                     jlong llm_ptr,
                                                                     jint max_new_tokens) {
    auto* llm = reinterpret_cast<mls::LlmSession*>(llm_ptr);
    if (llm) {
        llm->SetMaxNewTokens(max_new_tokens);
    }

}
extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_ChatSession_updateSystemPromptNative(JNIEnv *env, jobject thiz,
                                                                     jlong llm_ptr,
                                                                     jstring system_promp_j) {
    auto* llm = reinterpret_cast<mls::LlmSession*>(llm_ptr);
    const char* system_prompt_cstr = env->GetStringUTFChars(system_promp_j, nullptr);
    if (llm) {
        llm->setSystemPrompt(system_prompt_cstr);
    }
    env->ReleaseStringUTFChars(system_promp_j, system_prompt_cstr);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_ChatSession_updateAssistantPromptNative(JNIEnv *env, jobject thiz,
                                                                        jlong llm_ptr,
                                                                        jstring assistant_prompt_j) {
    auto* llm = reinterpret_cast<mls::LlmSession*>(llm_ptr);
    const char* assistant_prompt_cstr = env->GetStringUTFChars(assistant_prompt_j, nullptr);
    if (llm) {
        llm->SetAssistantPrompt(assistant_prompt_cstr);
    }
    env->ReleaseStringUTFChars(assistant_prompt_j, assistant_prompt_cstr);
}