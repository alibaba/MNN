#include <jni.h>
#include "diffusion_session.h"
#include "nlohmann/json.hpp"
#include "mls_log.h"

using namespace mls;
using namespace nlohmann;

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
    const char* config_path_cstr = env->GetStringUTFChars(config_path, nullptr);
    const char* extra_json_config_cstr = env->GetStringUTFChars(extra_config_j, nullptr);
    MNN_DEBUG("DiffusionSession::initNative config_path_cstr : %s extra_json_config_cstr: %s", config_path_cstr, extra_json_config_cstr);
    json extra_json_config = json::parse(extra_json_config_cstr);
    std::string diffusion_memory_mode = extra_json_config["diffusion_memory_mode"];
    int diffusion_memory_mode_int = std::stoi(diffusion_memory_mode);
    auto diffusion = new DiffusionSession(config_path_cstr, diffusion_memory_mode_int);
    env->ReleaseStringUTFChars(extra_config_j, extra_json_config_cstr);
    env->ReleaseStringUTFChars(config_path, config_path_cstr);
    return reinterpret_cast<jlong>(diffusion);
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
    jclass progressListenerClass = env->GetObjectClass(progress_listener);
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
                   [env, progress_listener, onProgressMethod](int progress) {
                       if (progress_listener && onProgressMethod) {
                           jstring javaString =  env->NewStringUTF(std::to_string(progress).c_str());
                           env->CallBooleanMethod(progress_listener, onProgressMethod,  javaString);
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