#include <jni.h>
#include "diffusion_session.h"
#include "nlohmann/json.hpp"
#include "mls_log.h"

using namespace mls;
using namespace nlohmann;

namespace {
// A missing or wrongly typed key must not throw: an uncaught C++ exception crossing the JNI
// boundary terminates the whole process instead of reaching the app's error handling.
int ReadMemoryMode(const char* extra_config) {
    if (extra_config == nullptr) {
        return 0;
    }
    json config = json::parse(extra_config, nullptr, false);
    if (config.is_discarded() || !config.contains("diffusion_memory_mode")) {
        MNN_ERROR("DiffusionSession: unusable extra config, falling back to memory mode 0");
        return 0;
    }
    const auto& value = config["diffusion_memory_mode"];
    try {
        if (value.is_number_integer()) {
            return value.get<int>();
        }
        if (value.is_string()) {
            return std::stoi(value.get<std::string>());
        }
    } catch (const std::exception& e) {
        MNN_ERROR("DiffusionSession: bad diffusion_memory_mode: %s", e.what());
    }
    return 0;
}
}

extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_DiffusionSession_resetNative(JNIEnv *env, jobject thiz,
                                                                 jlong instance_id) {

}

extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_DiffusionSession_releaseNative(JNIEnv *env, jobject thiz,
                               jlong instance_id) {
    auto* diffusion = reinterpret_cast<DiffusionSession*>(instance_id);
    delete diffusion;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_alibaba_mnnllm_android_llm_DiffusionSession_initNative(JNIEnv *env,
                                                                jobject thiz,
                                                                jstring config_path,
                                                                jstring extra_config_j) {
    MNN_DEBUG("DiffusionSession::initNative");
    if (config_path == nullptr) {
        MNN_ERROR("DiffusionSession::initNative got a null config path");
        return 0;
    }
    const char* config_path_cstr = env->GetStringUTFChars(config_path, nullptr);
    if (config_path_cstr == nullptr) {
        return 0;
    }
    const char* extra_json_config_cstr =
        extra_config_j == nullptr ? nullptr : env->GetStringUTFChars(extra_config_j, nullptr);
    MNN_DEBUG("DiffusionSession::initNative config_path_cstr : %s extra_json_config_cstr: %s",
              config_path_cstr, extra_json_config_cstr);
    int diffusion_memory_mode_int = ReadMemoryMode(extra_json_config_cstr);
    jlong result = 0;
    try {
        auto* diffusion = new DiffusionSession(config_path_cstr, diffusion_memory_mode_int);
        if (diffusion->IsLoaded()) {
            result = reinterpret_cast<jlong>(diffusion);
        } else {
            MNN_ERROR("DiffusionSession::initNative could not load the model under %s", config_path_cstr);
            delete diffusion;
        }
    } catch (const std::exception& e) {
        MNN_ERROR("DiffusionSession::initNative failed: %s", e.what());
    } catch (...) {
        MNN_ERROR("DiffusionSession::initNative failed with an unknown exception");
    }
    if (extra_json_config_cstr != nullptr) {
        env->ReleaseStringUTFChars(extra_config_j, extra_json_config_cstr);
    }
    env->ReleaseStringUTFChars(config_path, config_path_cstr);
    return result;
}
extern "C"
JNIEXPORT jobject JNICALL
Java_com_alibaba_mnnllm_android_llm_DiffusionSession_submitDiffusionNative(JNIEnv *env,
                                                                           jobject thiz,
                                                                           jlong instance_id,
                                                                           jstring input,
                                                                           jstring joutput_path,
                                                                           jint iter_num,
                                                                           jint random_seed,
                                                                           jobject progress_listener) {
    auto* diffusion = reinterpret_cast<DiffusionSession*>(instance_id); // Cast back to Llm*
    if (!diffusion) {
        return nullptr;
    }
    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
    jmethodID putMethod = env->GetMethodID(hashMapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject hashMap = env->NewObject(hashMapClass, hashMapInit);
    auto putString = [&](const char* key, const char* value) {
        jstring jkey = env->NewStringUTF(key);
        jstring jvalue = env->NewStringUTF(value);
        env->CallObjectMethod(hashMap, putMethod, jkey, jvalue);
        env->DeleteLocalRef(jkey);
        env->DeleteLocalRef(jvalue);
    };
    if (input == nullptr || joutput_path == nullptr) {
        putString("error", "true");
        putString("message", "Diffusion request is missing the prompt or the output path");
        return hashMap;
    }
    jclass progressListenerClass = progress_listener == nullptr ? nullptr : env->GetObjectClass(progress_listener);
    jmethodID onProgressMethod = progressListenerClass == nullptr
                                     ? nullptr
                                     : env->GetMethodID(progressListenerClass, "onProgress", "(Ljava/lang/String;)Z");
    if (progressListenerClass != nullptr && onProgressMethod == nullptr) {
        MNN_ERROR("ProgressListener onProgress method not found.");
    }
    const char* input_cstr = env->GetStringUTFChars(input, nullptr);
    const char* output_cstr = env->GetStringUTFChars(joutput_path, nullptr);
    std::string prompt(input_cstr);
    std::string output_path(output_cstr);
    env->ReleaseStringUTFChars(input, input_cstr);
    env->ReleaseStringUTFChars(joutput_path, output_cstr);
    auto start = std::chrono::high_resolution_clock::now();
    bool success = false;
    try {
        success = diffusion->Run(prompt,
                                 output_path,
                                 iter_num,
                                 random_seed,
                                 [env, progress_listener, onProgressMethod](int progress) {
                                     if (progress_listener && onProgressMethod) {
                                         std::string progress_str = std::to_string(progress);
                                         jstring javaString = env->NewStringUTF(progress_str.c_str());
                                         env->CallBooleanMethod(progress_listener, onProgressMethod, javaString);
                                         env->DeleteLocalRef(javaString);
                                     }
                                 });
    } catch (const std::exception& e) {
        MNN_ERROR("DiffusionSession::submitDiffusionNative failed: %s", e.what());
    } catch (...) {
        MNN_ERROR("DiffusionSession::submitDiffusionNative failed with an unknown exception");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    if (!success) {
        putString("error", "true");
        putString("message", "Image generation failed, see logcat for the native error");
    }
    jclass longClass = env->FindClass("java/lang/Long");
    jmethodID longInit = env->GetMethodID(longClass, "<init>", "(J)V");
    jobject jduration = env->NewObject(longClass, longInit, static_cast<jlong>(duration));
    jstring jkey = env->NewStringUTF("total_timeus");
    env->CallObjectMethod(hashMap, putMethod, jkey, jduration);
    env->DeleteLocalRef(jkey);
    env->DeleteLocalRef(jduration);
    env->DeleteLocalRef(longClass);
    if (progressListenerClass != nullptr) {
        env->DeleteLocalRef(progressListenerClass);
    }
    env->DeleteLocalRef(hashMapClass);
    return hashMap;
}