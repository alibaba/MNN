
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
    const char* resource_path_cstr = env->GetStringUTFChars(resource_path, nullptr);
    const char* config_cstr = env->GetStringUTFChars(config_json, nullptr);
    
    MNN_DEBUG("SanaSession_initNative resource: %s", resource_path_cstr);
    
    // Parse config if needed
    json config = json::parse(config_cstr);
    int memory_mode = 0;
    if (config.contains("diffusion_memory_mode")) {
        std::string mode_str = config["diffusion_memory_mode"];
        memory_mode = std::stoi(mode_str);
    }
    
    int backend_type = MNN_FORWARD_OPENCL;
    if (config.contains("backend_type")) {
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
    if (config.contains("image_width")) width = config["image_width"];
    if (config.contains("image_height")) height = config["image_height"];
    if (config.contains("grid_size")) grid_size = config["grid_size"];

    auto session = new SanaSession(resource_path_cstr, memory_mode, backend_type, width, height, grid_size);
    session->Load(); // Load immediately or lazy load? Let's load here to be safe.

    env->ReleaseStringUTFChars(resource_path, resource_path_cstr);
    env->ReleaseStringUTFChars(config_json, config_cstr);
    
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
