
#include <jni.h>
#include "sana_session.h"
#include "nlohmann/json.hpp"
#include "mls_log.h"

using namespace mls;
using namespace nlohmann;

extern "C"
JNIEXPORT jlong JNICALL
Java_com_alibaba_mnnllm_android_llm_SanaSession_initNative(JNIEnv *env,
                                                       jobject thiz,
                                                       jstring resource_path,
                                                       jstring config_json) {
    if (resource_path == nullptr) {
        MNN_ERROR("SanaSession_initNative got a null resource path");
        return 0;
    }
    const char* resource_path_cstr = env->GetStringUTFChars(resource_path, nullptr);
    if (resource_path_cstr == nullptr) {
        return 0;
    }
    const char* config_cstr = config_json == nullptr ? nullptr : env->GetStringUTFChars(config_json, nullptr);

    MNN_DEBUG("SanaSession_initNative resource: %s", resource_path_cstr);

    // Every read below stays inside one try block: json::parse, the implicit json -> std::string
    // / int conversions and std::stoi all throw, and an exception crossing the JNI boundary kills
    // the process instead of reaching the app's error handling.
    SanaSession* session = nullptr;
    try {
        json config = config_cstr == nullptr ? json::parse("", nullptr, false) : json::parse(config_cstr, nullptr, false);
        if (config.is_discarded()) {
            config = json::object();
            MNN_ERROR("SanaSession_initNative got an unusable config, using defaults");
        }
        int memory_mode = 0;
        if (config.contains("diffusion_memory_mode")) {
            const auto& value = config["diffusion_memory_mode"];
            memory_mode = value.is_number_integer() ? value.get<int>() : std::stoi(value.get<std::string>());
        }

        int backend_type = MNN_FORWARD_OPENCL;
        if (config.contains("backend_type") && config["backend_type"].is_string()) {
            std::string backend_str = config["backend_type"];
            if (backend_str == "cpu") {
                backend_type = MNN_FORWARD_CPU;
            } else if (backend_str == "opencl") {
                backend_type = MNN_FORWARD_OPENCL;
            } else if (backend_str == "vulkan") {
                backend_type = MNN_FORWARD_VULKAN;
            }
        }

        int width = 512;
        int height = 512;
        int grid_size = 1;
        if (config.contains("image_width") && config["image_width"].is_number_integer()) width = config["image_width"];
        if (config.contains("image_height") && config["image_height"].is_number_integer()) height = config["image_height"];
        if (config.contains("grid_size") && config["grid_size"].is_number_integer()) grid_size = config["grid_size"];

        session = new SanaSession(resource_path_cstr, memory_mode, backend_type, width, height, grid_size);
        session->Load(); // Load immediately or lazy load? Let's load here to be safe.
    } catch (const std::exception& e) {
        MNN_ERROR("SanaSession_initNative failed: %s", e.what());
        delete session;
        session = nullptr;
    } catch (...) {
        MNN_ERROR("SanaSession_initNative failed with an unknown exception");
        delete session;
        session = nullptr;
    }

    if (config_cstr != nullptr) {
        env->ReleaseStringUTFChars(config_json, config_cstr);
    }
    env->ReleaseStringUTFChars(resource_path, resource_path_cstr);

    return reinterpret_cast<jlong>(session);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_SanaSession_releaseNative(JNIEnv *env, jobject thiz, jlong instance_id) {
    auto* session = reinterpret_cast<SanaSession*>(instance_id);
    if (session) {
        delete session;
    }
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_alibaba_mnnllm_android_llm_SanaSession_generateNative(JNIEnv *env,
                                                           jobject thiz,
                                                           jlong instance_id,
                                                           jstring prompt,
                                                           jstring image_path,
                                                           jstring output_path,
                                                           jint steps,
                                                           jint seed,
                                                           jboolean use_cfg,
                                                           jfloat cfg_scale,
                                                           jobject progress_listener) {
    try {
        MNN_DEBUG("generateNative: ENTRY - instance_id=%lld", (long long)instance_id);
        auto* session = reinterpret_cast<SanaSession*>(instance_id);
        if (!session) {
            MNN_ERROR("generateNative: session is null!");
            return nullptr;
        }
        MNN_DEBUG("generateNative: session is valid");

        jclass progressListenerClass = env->GetObjectClass(progress_listener);
        jmethodID onProgressMethod = env->GetMethodID(progressListenerClass, "onProgress", "(Ljava/lang/String;)Z");

        const char* prompt_cstr = env->GetStringUTFChars(prompt, nullptr);
        const char* image_path_cstr = env->GetStringUTFChars(image_path, nullptr);
        const char* output_path_cstr = env->GetStringUTFChars(output_path, nullptr);
        
        std::string prompt_str = prompt_cstr;
        std::string image_path_str = image_path_cstr;
        std::string output_path_str = output_path_cstr;
        
        env->ReleaseStringUTFChars(prompt, prompt_cstr);
        env->ReleaseStringUTFChars(image_path, image_path_cstr);
        env->ReleaseStringUTFChars(output_path, output_path_cstr);

        MNN_DEBUG("generateNative: prompt=%s, image_path=%s, output=%s, steps=%d, seed=%d, use_cfg=%d, cfg_scale=%.2f",
                  prompt_str.c_str(), image_path_str.c_str(), output_path_str.c_str(), steps, seed, use_cfg, cfg_scale);

        auto start = std::chrono::high_resolution_clock::now();

        MNN_DEBUG("generateNative: Calling session->Run()...");
        bool success = session->Run(prompt_str, image_path_str, output_path_str, steps, seed, use_cfg, cfg_scale,
            [env, progress_listener, onProgressMethod](int progress) {
                 if (progress_listener && onProgressMethod) {
                       std::string progress_str = std::to_string(progress);
                       jstring javaString = env->NewStringUTF(progress_str.c_str());
                       env->CallBooleanMethod(progress_listener, onProgressMethod, javaString);
                       env->DeleteLocalRef(javaString);
                 }
            }
        );

        MNN_DEBUG("generateNative: session->Run() returned, success=%d", success);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        jclass hashMapClass = env->FindClass("java/util/HashMap");
        jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
        jmethodID putMethod = env->GetMethodID(hashMapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
        jobject hashMap = env->NewObject(hashMapClass, hashMapInit);
        env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("total_timeus"), env->NewObject(env->FindClass("java/lang/Long"), env->GetMethodID(env->FindClass("java/lang/Long"), "<init>", "(J)V"), duration));

        // Add success flag
        jclass booleanClass = env->FindClass("java/lang/Boolean");
        jmethodID booleanInit = env->GetMethodID(booleanClass, "<init>", "(Z)V");
        env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("success"), env->NewObject(booleanClass, booleanInit, success));

        // Add error information if failed
        if (!success) {
            env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("error"), env->NewObject(booleanClass, booleanInit, true));
            env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("message"), env->NewStringUTF("Native generation failed. Check logcat for MNN_ERROR messages."));
        }

        return hashMap;
    } catch (const std::exception& e) {
        MNN_ERROR("Exception in generateNative: %s", e.what());
        jclass hashMapClass = env->FindClass("java/util/HashMap");
        jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
        jmethodID putMethod = env->GetMethodID(hashMapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
        jobject hashMap = env->NewObject(hashMapClass, hashMapInit);
        jclass booleanClass = env->FindClass("java/lang/Boolean");
        jmethodID booleanInit = env->GetMethodID(booleanClass, "<init>", "(Z)V");
        env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("error"), env->NewObject(booleanClass, booleanInit, true));
        env->CallObjectMethod(hashMap, putMethod, env->NewStringUTF("message"), env->NewStringUTF(e.what()));
        return hashMap;
    } catch (...) {
        MNN_ERROR("Unknown exception in generateNative");
        return nullptr;
    }
}
