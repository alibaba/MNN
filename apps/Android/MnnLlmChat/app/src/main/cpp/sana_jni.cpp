#include <jni.h>
#include <memory>
#include "sana_session.h"
#include "nlohmann/json.hpp"
#include "mls_log.h"
#include "jni_utils.hpp"

using namespace mls;
using namespace mls::jni;
using namespace nlohmann;

extern "C"
JNIEXPORT jlong JNICALL
Java_com_alibaba_mnnllm_android_llm_SanaSession_initNative(
        JNIEnv* env, jobject thiz, jstring resource_path, jstring config_json) {
    return guarded(env, [&]() -> jlong {
        if (!resource_path) return 0;
        const auto path = toStdString(env, resource_path);
        if (env->ExceptionCheck()) return 0;
        const auto config_text = toStdString(env, config_json);
        if (env->ExceptionCheck()) return 0;
        json config = json::parse(config_text, nullptr, false);
        if (!config.is_object()) config = json::object();
        int memory_mode = 0;
        if (config.contains("diffusion_memory_mode")) {
            const auto& value = config["diffusion_memory_mode"];
            if (value.is_number_integer()) {
                memory_mode = value.get<int>();
            } else if (value.is_string()) {
                memory_mode = std::stoi(value.get<std::string>());
            }
        }
        int backend_type = MNN_FORWARD_OPENCL;
        if (config.contains("backend_type") && config["backend_type"].is_string()) {
            const std::string backend = config["backend_type"];
            if (backend == "cpu") backend_type = MNN_FORWARD_CPU;
            else if (backend == "vulkan") backend_type = MNN_FORWARD_VULKAN;
        }
        int width = 512;
        int height = 512;
        int grid_size = 1;
        if (config.contains("image_width") && config["image_width"].is_number_integer()) width = config["image_width"];
        if (config.contains("image_height") && config["image_height"].is_number_integer()) height = config["image_height"];
        if (config.contains("grid_size") && config["grid_size"].is_number_integer()) grid_size = config["grid_size"];
        auto session = std::make_unique<SanaSession>(path, memory_mode, backend_type, width, height, grid_size);
        session->Load();
        return reinterpret_cast<jlong>(session.release());
    });
}

extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_SanaSession_releaseNative(JNIEnv* env, jobject thiz, jlong instance_id) {
    delete reinterpret_cast<SanaSession*>(instance_id);
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_alibaba_mnnllm_android_llm_SanaSession_generateNative(
        JNIEnv* env, jobject thiz, jlong instance_id, jstring prompt, jstring image_path, jstring output_path,
        jint steps, jint seed, jboolean use_cfg, jfloat cfg_scale, jobject progress_listener) {
    return guarded(env, [&]() -> jobject {
        auto* session = reinterpret_cast<SanaSession*>(instance_id);
        if (!session) return errorMap(env, "Sana session is not loaded");
        if (!prompt || !output_path) return errorMap(env, "Sana request is missing the prompt or the output path", false);
        const auto prompt_text = toStdString(env, prompt);
        if (env->ExceptionCheck()) return nullptr;
        const auto image = toStdString(env, image_path);
        if (env->ExceptionCheck()) return nullptr;
        const auto output = toStdString(env, output_path);
        if (env->ExceptionCheck()) return nullptr;
        ProgressListener listener(env, progress_listener);
        if (env->ExceptionCheck()) return nullptr;
        const auto start = std::chrono::high_resolution_clock::now();
        bool success = false;
        try {
            success = session->Run(prompt_text, image, output, steps, seed, use_cfg, cfg_scale, [&](int progress) {
                guarded(env, [&] { listener.onProgress(std::to_string(progress)); });
            });
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& error) {
            MNN_ERROR("Sana generation failed: %s", error.what());
        } catch (...) {
            MNN_ERROR("Sana generation failed with an unknown exception");
        }
        if (env->ExceptionCheck()) return nullptr;
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        JavaMap result(env);
        result.putLong("total_timeus", duration);
        result.putBool("success", success);
        if (!success) {
            result.putBool("error", true);
            result.putString("message", "Native generation failed. Check logcat for MNN_ERROR messages.");
            result.putBool("session_invalid", true);
        }
        return result.release();
    });
}
