#include <jni.h>
#include <memory>
#include "diffusion_session.h"
#include "nlohmann/json.hpp"
#include "mls_log.h"
#include "jni_utils.hpp"

using namespace mls;
using namespace mls::jni;
using namespace nlohmann;

namespace {
int ReadMemoryMode(const std::string& extra_config) {
    json config = json::parse(extra_config, nullptr, false);
    if (!config.is_object() || !config.contains("diffusion_memory_mode")) return 0;
    const auto& value = config["diffusion_memory_mode"];
    if (value.is_number_integer()) return value.get<int>();
    if (value.is_string()) {
        try {
            return std::stoi(value.get<std::string>());
        } catch (const std::invalid_argument&) {
            return 0;
        } catch (const std::out_of_range&) {
            return 0;
        }
    }
    return 0;
}
}

extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_DiffusionSession_resetNative(JNIEnv* env, jobject thiz, jlong instance_id) {
}

extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_DiffusionSession_releaseNative(JNIEnv* env, jobject thiz, jlong instance_id) {
    delete reinterpret_cast<DiffusionSession*>(instance_id);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_alibaba_mnnllm_android_llm_DiffusionSession_initNative(
        JNIEnv* env, jobject thiz, jstring config_path, jstring extra_config_j) {
    return guarded(env, [&]() -> jlong {
        if (!config_path) return 0;
        const auto path = toStdString(env, config_path);
        if (env->ExceptionCheck()) return 0;
        const auto extra = toStdString(env, extra_config_j);
        if (env->ExceptionCheck()) return 0;
        auto session = std::make_unique<DiffusionSession>(path, ReadMemoryMode(extra));
        if (!session->IsLoaded()) {
            MNN_ERROR("DiffusionSession: model load failed under %s", path.c_str());
            return 0;
        }
        return reinterpret_cast<jlong>(session.release());
    });
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_alibaba_mnnllm_android_llm_DiffusionSession_submitDiffusionNative(
        JNIEnv* env, jobject thiz, jlong instance_id, jstring input, jstring joutput_path,
        jint iter_num, jint random_seed, jobject progress_listener) {
    return guarded(env, [&]() -> jobject {
        auto* session = reinterpret_cast<DiffusionSession*>(instance_id);
        if (!session) return errorMap(env, "Diffusion session is not loaded");
        if (!input || !joutput_path) {
            return errorMap(env, "Diffusion request is missing the prompt or the output path", false);
        }
        const auto prompt = toStdString(env, input);
        if (env->ExceptionCheck()) return nullptr;
        const auto output = toStdString(env, joutput_path);
        if (env->ExceptionCheck()) return nullptr;
        ProgressListener listener(env, progress_listener);
        if (env->ExceptionCheck()) return nullptr;
        const auto start = std::chrono::high_resolution_clock::now();
        bool success = false;
        try {
            success = session->Run(prompt, output, iter_num, random_seed, [&](int progress) {
                guarded(env, [&] { listener.onProgress(std::to_string(progress)); });
            });
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& error) {
            MNN_ERROR("Diffusion generation failed: %s", error.what());
        } catch (...) {
            MNN_ERROR("Diffusion generation failed with an unknown exception");
        }
        if (env->ExceptionCheck()) return nullptr;
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        JavaMap result(env);
        if (!success) {
            result.putBool("error", true);
            result.putString("message", "Image generation failed, see logcat for the native error");
            result.putBool("session_invalid", true);
        }
        result.putLong("total_timeus", duration);
        return result.release();
    });
}
