#include <android/log.h>
#include <jni.h>
#include <memory>
#include <limits>
#include "llm_session.h"
#include "mls_log.h"
#include "jni_utils.hpp"

using json = nlohmann::json;
using namespace mls::jni;

namespace {
JavaVM* g_jvm = nullptr;

json parseObjectOrEmpty(const std::string& text, const char* what) {
    json parsed = json::parse(text, nullptr, false);
    if (!parsed.is_object()) {
        MNN_ERROR("initNative: %s is not a JSON object, using defaults\n", what);
        return json::object();
    }
    return parsed;
}

bool requireSession(JNIEnv* env, mls::LlmSession* session) {
    if (env->ExceptionCheck()) return false;
    if (session && session->isModelReady() && !session->lastEngineError()) return true;
    throwException(env, "java/lang/IllegalStateException",
                   session && session->lastEngineError() ? session->lastEngineError() : "LLM is not loaded");
    return false;
}

void putStats(JavaMap& result, const MNN::Transformer::LlmContext* context) {
    result.putLong("prompt_len", context->prompt_len);
    result.putLong("decode_len", context->gen_seq_len);
    result.putLong("vision_time", context->vision_us);
    result.putLong("audio_time", context->audio_us);
    result.putLong("prefill_time", context->prefill_us);
    result.putLong("decode_time", context->decode_us);
}

struct AudioListener {
    JavaVM* vm = nullptr;
    jobject listener = nullptr;
    jmethodID method = nullptr;

    ~AudioListener() {
        if (!vm || !listener) return;
        JNIEnv* env = nullptr;
        bool detach = vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK;
        if (detach && vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
        if (env) env->DeleteGlobalRef(listener);
        if (detach) vm->DetachCurrentThread();
    }
};

jobject benchmarkResult(JNIEnv* env, jobject instance, bool success, const std::string& message) {
    if (env->ExceptionCheck()) return nullptr;
    LocalRef<jclass> cls(env, env->FindClass("com/alibaba/mnnllm/android/benchmark/BenchmarkResult"));
    if (!cls.get() || env->ExceptionCheck()) return nullptr;
    jmethodID ctor = env->GetMethodID(cls.get(), "<init>",
                                    "(Lcom/alibaba/mnnllm/android/benchmark/TestInstance;ZLjava/lang/String;)V");
    if (!ctor || env->ExceptionCheck()) return nullptr;
    LocalRef<jstring> error(env, success ? nullptr : newJString(env, message));
    if (env->ExceptionCheck() || (!success && !error.get())) return nullptr;
    return env->NewObject(cls.get(), ctor, instance, static_cast<jboolean>(success), error.get());
}
}

bool ReportLlmSetConfigToFirebase(const std::string& stage, const std::string& config_json) {
    if (!g_jvm) return true;
    JNIEnv* env = nullptr;
    if (g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_4) != JNI_OK || !env) return true;
    return guarded(env, [&]() -> bool {
        LocalRef<jclass> cls(env, env->FindClass("com/alibaba/mnnllm/android/utils/CrashReportContext"));
        if (!cls.get() || env->ExceptionCheck()) return false;
        jmethodID method = env->GetStaticMethodID(cls.get(), "reportLlmSetConfig",
                                                "(Ljava/lang/String;Ljava/lang/String;)V");
        if (!method || env->ExceptionCheck()) return false;
        LocalRef<jstring> stage_j(env, newJString(env, stage));
        if (!stage_j.get() || env->ExceptionCheck()) return false;
        LocalRef<jstring> config_j(env, newJString(env, config_json));
        if (!config_j.get() || env->ExceptionCheck()) return false;
        env->CallStaticVoidMethod(cls.get(), method, stage_j.get(), config_j.get());
        return !env->ExceptionCheck();
    });
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_jvm = vm;
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    g_jvm = nullptr;
}

JNIEXPORT jlong JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_initNative(
        JNIEnv* env, jobject thiz, jstring modelDir, jobject chat_history,
        jstring mergeConfigStr, jstring configJsonStr) {
    return guarded(env, [&]() -> jlong {
        const std::string model = toStdString(env, modelDir);
        if (env->ExceptionCheck()) return 0;
        const std::string merged = toStdString(env, mergeConfigStr);
        if (env->ExceptionCheck()) return 0;
        const std::string extra = toStdString(env, configJsonStr);
        if (env->ExceptionCheck()) return 0;
        std::vector<std::string> history;
        if (chat_history) {
            LocalRef<jclass> cls(env, env->GetObjectClass(chat_history));
            if (!cls.get() || env->ExceptionCheck()) return 0;
            jmethodID sizeMethod = env->GetMethodID(cls.get(), "size", "()I");
            if (!sizeMethod || env->ExceptionCheck()) return 0;
            jmethodID getMethod = env->GetMethodID(cls.get(), "get", "(I)Ljava/lang/Object;");
            if (!getMethod || env->ExceptionCheck()) return 0;
            const jint size = env->CallIntMethod(chat_history, sizeMethod);
            if (env->ExceptionCheck()) return 0;
            for (jint i = 0; i < size; ++i) {
                LocalRef<jstring> item(env, static_cast<jstring>(env->CallObjectMethod(chat_history, getMethod, i)));
                if (env->ExceptionCheck()) return 0;
                auto text = toStdString(env, item.get());
                if (env->ExceptionCheck()) return 0;
                history.emplace_back(std::move(text));
            }
        }
        auto session = std::make_unique<mls::LlmSession>(model, parseObjectOrEmpty(merged, "merged config"),
                                                       parseObjectOrEmpty(extra, "extra config"), std::move(history));
        if (!session->Load() || !session->isModelReady()) {
            MNN_ERROR("Model load failed: %s", session->getLastLoadError().c_str());
            throwException(env, "java/lang/IllegalStateException", "Model load failed; see logcat for the native error");
            return 0;
        }
        if (env->ExceptionCheck()) return 0;
        return reinterpret_cast<jlong>(session.release());
    });
}

JNIEXPORT jobject JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_submitNative(
        JNIEnv* env, jobject thiz, jlong llmPtr, jstring inputStr, jboolean keepHistory, jobject progressListener) {
    return guarded(env, [&]() -> jobject {
        auto* session = reinterpret_cast<mls::LlmSession*>(llmPtr);
        if (!session || !session->isModelReady()) return errorMap(env, "Failed, Chat is not ready!");
        if (!progressListener) return errorMap(env, "Progress listener is missing", false);
        const std::string input = toStdString(env, inputStr);
        if (env->ExceptionCheck()) return nullptr;
        ProgressListener progress(env, progressListener);
        if (env->ExceptionCheck()) return nullptr;
        const auto* context = session->Response(input, [&](const std::string& text, bool end) {
            const bool stop = guarded(env, [&] { return progress.onProgress(text, end); });
            return env->ExceptionCheck() || stop;
        }, [&] { return env->ExceptionCheck(); });
        if (env->ExceptionCheck()) return nullptr;
        if (!context || session->lastEngineError()) {
            return errorMap(env, session->lastEngineError() ? session->lastEngineError() : "generation_failed");
        }
        const auto stats = session->lastResponseStats();
        JavaMap result(env);
        if (stats.context_too_long) {
            result.putBool("error", true);
            result.putString("message", "context_too_long");
            result.putLong("context_limit", stats.context_limit);
            return result.release();
        }
        putStats(result, context);
        result.putLong("context_tokens", stats.context_tokens);
        result.putLong("context_limit", stats.context_limit);
        result.putBool("context_truncated", stats.truncated);
        result.putLong("peak_anon_kb", stats.peak_anon_kb);
        return result.release();
    });
}

JNIEXPORT jintArray JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_prepareContextNative(
        JNIEnv* env, jobject thiz, jlong llmPtr, jstring promptStr) {
    return guarded(env, [&]() -> jintArray {
        auto* session = reinterpret_cast<mls::LlmSession*>(llmPtr);
        if (!requireSession(env, session)) return nullptr;
        const std::string prompt = toStdString(env, promptStr);
        if (env->ExceptionCheck()) return nullptr;
        const auto check = session->prepareContext(prompt);
        if (!requireSession(env, session)) return nullptr;
        const jint values[4] = {check.tokens, check.budget, check.limit, static_cast<jint>(check.action)};
        LocalRef<jintArray> array(env, env->NewIntArray(4));
        if (!array.get() || env->ExceptionCheck()) return nullptr;
        env->SetIntArrayRegion(array.get(), 0, 4, values);
        return env->ExceptionCheck() ? nullptr : array.release();
    });
}

JNIEXPORT jobject JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_compressContextNative(
        JNIEnv* env, jobject thiz, jlong llmPtr) {
    return guarded(env, [&]() -> jobject {
        auto* session = reinterpret_cast<mls::LlmSession*>(llmPtr);
        if (!requireSession(env, session)) return nullptr;
        const auto compressed = session->compressContext();
        if (!requireSession(env, session)) return nullptr;
        JavaMap result(env);
        result.putBool("success", compressed.success);
        result.putString("summary", compressed.summary);
        result.putLong("kept_user_turns", compressed.kept_user_turns);
        return result.release();
    });
}

JNIEXPORT jint JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_countContextTokensNative(
        JNIEnv* env, jobject thiz, jlong llmPtr) {
    return guarded(env, [&]() -> jint {
        auto* session = reinterpret_cast<mls::LlmSession*>(llmPtr);
        if (!requireSession(env, session)) return 0;
        const int tokens = session->countContextTokens();
        return requireSession(env, session) ? tokens : 0;
    });
}

JNIEXPORT jintArray JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_contextLimitsNative(
        JNIEnv* env, jobject thiz, jlong llmPtr) {
    return guarded(env, [&]() -> jintArray {
        auto* session = reinterpret_cast<mls::LlmSession*>(llmPtr);
        const jint values[2] = {session ? session->contextLimit() : 0, session ? session->modelMaxContext() : 0};
        LocalRef<jintArray> array(env, env->NewIntArray(2));
        if (!array.get() || env->ExceptionCheck()) return nullptr;
        env->SetIntArrayRegion(array.get(), 0, 2, values);
        return env->ExceptionCheck() ? nullptr : array.release();
    });
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_cancelContextWorkNative(
        JNIEnv* env, jobject thiz, jlong llmPtr) {
    auto* session = reinterpret_cast<mls::LlmSession*>(llmPtr);
    if (session) session->cancelContextWork();
}

JNIEXPORT jobject JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_submitFullHistoryNative(
        JNIEnv* env, jobject thiz, jlong llmPtr, jobject historyList, jobject progressListener) {
    return guarded(env, [&]() -> jobject {
        auto* session = reinterpret_cast<mls::LlmSession*>(llmPtr);
        if (!session || !session->isModelReady()) return errorMap(env, "Failed, Chat is not ready!");
        if (!historyList || !progressListener) return errorMap(env, "History or progress listener is missing", false);
        std::vector<mls::PromptItem> history;
        LocalRef<jclass> listClass(env, env->GetObjectClass(historyList));
        if (!listClass.get() || env->ExceptionCheck()) return nullptr;
        jmethodID sizeMethod = env->GetMethodID(listClass.get(), "size", "()I");
        if (!sizeMethod || env->ExceptionCheck()) return nullptr;
        jmethodID getMethod = env->GetMethodID(listClass.get(), "get", "(I)Ljava/lang/Object;");
        if (!getMethod || env->ExceptionCheck()) return nullptr;
        const jint size = env->CallIntMethod(historyList, sizeMethod);
        if (env->ExceptionCheck()) return nullptr;
        LocalRef<jclass> pairClass(env, env->FindClass("android/util/Pair"));
        if (!pairClass.get() || env->ExceptionCheck()) return nullptr;
        jfieldID first = env->GetFieldID(pairClass.get(), "first", "Ljava/lang/Object;");
        if (!first || env->ExceptionCheck()) return nullptr;
        jfieldID second = env->GetFieldID(pairClass.get(), "second", "Ljava/lang/Object;");
        if (!second || env->ExceptionCheck()) return nullptr;
        for (jint i = 0; i < size; ++i) {
            LocalRef<jobject> pair(env, env->CallObjectMethod(historyList, getMethod, i));
            if (env->ExceptionCheck()) return nullptr;
            if (!pair.get()) continue;
            LocalRef<jstring> role(env, static_cast<jstring>(env->GetObjectField(pair.get(), first)));
            if (env->ExceptionCheck()) return nullptr;
            LocalRef<jstring> content(env, static_cast<jstring>(env->GetObjectField(pair.get(), second)));
            if (env->ExceptionCheck()) return nullptr;
            if (!role.get() || !content.get()) continue;
            auto role_text = toStdString(env, role.get());
            if (env->ExceptionCheck()) return nullptr;
            auto content_text = toStdString(env, content.get());
            if (env->ExceptionCheck()) return nullptr;
            history.emplace_back(std::move(role_text), std::move(content_text));
        }
        ProgressListener progress(env, progressListener);
        if (env->ExceptionCheck()) return nullptr;
        const auto* context = session->ResponseWithHistory(history, [&](const std::string& text, bool end) {
            const bool stop = guarded(env, [&] { return progress.onProgress(text, end); });
            return env->ExceptionCheck() || stop;
        }, [&] { return env->ExceptionCheck(); });
        if (env->ExceptionCheck()) return nullptr;
        if (!context || session->lastEngineError()) {
            return errorMap(env, session->lastEngineError() ? session->lastEngineError() : "generation_failed");
        }
        JavaMap result(env);
        putStats(result, context);
        return result.release();
    });
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_resetNative(
        JNIEnv* env, jobject thiz, jlong ptr) {
    guarded(env, [&] {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        if (session) session->Reset();
    });
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_replaceHistoryNative(
        JNIEnv* env, jobject thiz, jlong ptr, jobject historyList, jstring contextSummary) {
    guarded(env, [&] {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        if (!session) return;
        std::vector<std::string> history;
        if (historyList) {
            LocalRef<jclass> cls(env, env->GetObjectClass(historyList));
            if (!cls.get() || env->ExceptionCheck()) return;
            jmethodID sizeMethod = env->GetMethodID(cls.get(), "size", "()I");
            if (!sizeMethod || env->ExceptionCheck()) return;
            jmethodID getMethod = env->GetMethodID(cls.get(), "get", "(I)Ljava/lang/Object;");
            if (!getMethod || env->ExceptionCheck()) return;
            const jint size = env->CallIntMethod(historyList, sizeMethod);
            if (env->ExceptionCheck()) return;
            for (jint i = 0; i < size; ++i) {
                LocalRef<jstring> item(env, static_cast<jstring>(env->CallObjectMethod(historyList, getMethod, i)));
                if (env->ExceptionCheck()) return;
                auto text = toStdString(env, item.get());
                if (env->ExceptionCheck()) return;
                history.emplace_back(std::move(text));
            }
        }
        const std::string summary = contextSummary ? toStdString(env, contextSummary) : std::string();
        if (env->ExceptionCheck()) return;
        session->ReplaceHistory(history, summary);
    });
}

JNIEXPORT jboolean JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_setWavformCallbackNative(
        JNIEnv* env, jobject thiz, jlong ptr, jobject listener) {
    return guarded(env, [&]() -> jboolean {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        if (!listener || !requireSession(env, session)) return JNI_FALSE;
        auto binding = std::make_shared<AudioListener>();
        if (env->GetJavaVM(&binding->vm) != JNI_OK || !binding->vm) return JNI_FALSE;
        LocalRef<jclass> cls(env, env->GetObjectClass(listener));
        if (!cls.get() || env->ExceptionCheck()) return JNI_FALSE;
        binding->method = env->GetMethodID(cls.get(), "onAudioData", "([FZ)Z");
        if (!binding->method || env->ExceptionCheck()) return JNI_FALSE;
        binding->listener = env->NewGlobalRef(listener);
        if (!binding->listener || env->ExceptionCheck()) return JNI_FALSE;
        session->SetWavformCallback([binding, session](const float* data, size_t size, bool end) {
            JNIEnv* callback_env = nullptr;
            const bool detach = binding->vm->GetEnv(reinterpret_cast<void**>(&callback_env), JNI_VERSION_1_6) != JNI_OK;
            if (detach && binding->vm->AttachCurrentThread(&callback_env, nullptr) != JNI_OK) {
                session->failAudioCallback();
                return true;
            }
            if (!callback_env) {
                session->failAudioCallback();
                return true;
            }
            bool stop = guarded(callback_env, [&]() -> bool {
                if (size > static_cast<size_t>(std::numeric_limits<jsize>::max()) || (!data && size)) {
                    throwException(callback_env, "java/lang/IllegalStateException", "Invalid native audio buffer");
                    return true;
                }
                LocalRef<jfloatArray> array(callback_env, callback_env->NewFloatArray(static_cast<jsize>(size)));
                if (!array.get() || callback_env->ExceptionCheck()) return true;
                if (size) callback_env->SetFloatArrayRegion(array.get(), 0, static_cast<jsize>(size), data);
                if (callback_env->ExceptionCheck()) return true;
                const auto result = callback_env->CallBooleanMethod(binding->listener, binding->method,
                                                                    array.get(), static_cast<jboolean>(end));
                return callback_env->ExceptionCheck() || result == JNI_TRUE;
            });
            if (callback_env->ExceptionCheck()) {
                session->failAudioCallback();
                stop = true;
            }
            if (detach) binding->vm->DetachCurrentThread();
            return stop;
        });
        return JNI_TRUE;
    });
}

JNIEXPORT jstring JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_getDebugInfoNative(
        JNIEnv* env, jobject thiz, jlong ptr) {
    return guarded(env, [&] {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        return newJString(env, session ? session->getDebugInfo() : "");
    });
}

JNIEXPORT jstring JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_dumpConfigNative(
        JNIEnv* env, jobject thiz, jlong ptr) {
    return guarded(env, [&] {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        return newJString(env, session ? session->dumpConfig() : "{}");
    });
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_releaseNative(
        JNIEnv* env, jobject thiz, jlong ptr) {
    delete reinterpret_cast<mls::LlmSession*>(ptr);
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_updateMaxNewTokensNative(
        JNIEnv* env, jobject thiz, jlong ptr, jint tokens) {
    auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
    if (session) session->SetMaxNewTokens(tokens);
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_updateSystemPromptNative(
        JNIEnv* env, jobject thiz, jlong ptr, jstring prompt) {
    guarded(env, [&] {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        const auto text = toStdString(env, prompt);
        if (session && !env->ExceptionCheck()) session->setSystemPrompt(text);
    });
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_updateAssistantPromptNative(
        JNIEnv* env, jobject thiz, jlong ptr, jstring prompt) {
    guarded(env, [&] {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        const auto text = toStdString(env, prompt);
        if (session && !env->ExceptionCheck()) session->SetAssistantPrompt(text);
    });
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_updateConfigNative(
        JNIEnv* env, jobject thiz, jlong ptr, jstring config) {
    guarded(env, [&] {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        const auto text = toStdString(env, config);
        if (session && !env->ExceptionCheck()) session->updateConfig(text);
    });
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_updateEnableAudioOutputNative(
        JNIEnv* env, jobject thiz, jlong ptr, jboolean enable) {
    auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
    if (session) session->enableAudioOutput(enable);
}

JNIEXPORT jstring JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_getSystemPromptNative(
        JNIEnv* env, jobject thiz, jlong ptr) {
    return guarded(env, [&]() -> jstring {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        return session ? newJString(env, session->getSystemPrompt()) : nullptr;
    });
}

JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_clearHistoryNative(
        JNIEnv* env, jobject thiz, jlong ptr) {
    guarded(env, [&] {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        if (session) session->clearHistory();
    });
}

JNIEXPORT jobject JNICALL Java_com_alibaba_mnnllm_android_llm_LlmSession_runBenchmarkNative(
        JNIEnv* env, jobject thiz, jlong ptr, jint backend, jint threads, jboolean useMmap, jint power,
        jint precision, jint memory, jint dynamicOption, jint nPrompt, jint nGenerate, jint nRepeat,
        jboolean kvCache, jobject testInstance, jobject callback) {
    return guarded(env, [&]() -> jobject {
        auto* session = reinterpret_cast<mls::LlmSession*>(ptr);
        if (!session || !session->isModelReady() || !testInstance || !callback) {
            return benchmarkResult(env, testInstance, false, "LLM session or benchmark callback is not initialized");
        }
        LocalRef<jclass> callbackClass(env, env->GetObjectClass(callback));
        if (!callbackClass.get() || env->ExceptionCheck()) return nullptr;
        jmethodID onProgress = env->GetMethodID(callbackClass.get(), "onProgress",
                                              "(Lcom/alibaba/mnnllm/android/benchmark/BenchmarkProgress;)V");
        if (!onProgress || env->ExceptionCheck()) return nullptr;
        jmethodID onError = env->GetMethodID(callbackClass.get(), "onBenchmarkError", "(ILjava/lang/String;)V");
        if (!onError || env->ExceptionCheck()) return nullptr;
        LocalRef<jclass> progressClass(env, env->FindClass("com/alibaba/mnnllm/android/benchmark/BenchmarkProgress"));
        if (!progressClass.get() || env->ExceptionCheck()) return nullptr;
        jmethodID progressCtor = env->GetMethodID(progressClass.get(), "<init>", "(ILjava/lang/String;IIIIIFFFFF)V");
        if (!progressCtor || env->ExceptionCheck()) return nullptr;
        mls::LlmSession::BenchmarkCallback callbacks;
        callbacks.shouldAbort = [&] { return env->ExceptionCheck(); };
        callbacks.onProgress = [&](const mls::LlmSession::BenchmarkProgressInfo& info) {
            guarded(env, [&] {
                LocalRef<jstring> text(env, newJString(env, info.statusMessage));
                if (!text.get() || env->ExceptionCheck()) return;
                LocalRef<jobject> progress(env, env->NewObject(progressClass.get(), progressCtor,
                    static_cast<jint>(info.progress), text.get(), static_cast<jint>(info.progressType),
                    static_cast<jint>(info.currentIteration), static_cast<jint>(info.totalIterations),
                    static_cast<jint>(info.nPrompt), static_cast<jint>(info.nGenerate),
                    static_cast<jfloat>(info.runTimeSeconds), static_cast<jfloat>(info.prefillTimeSeconds),
                    static_cast<jfloat>(info.decodeTimeSeconds), static_cast<jfloat>(info.prefillSpeed),
                    static_cast<jfloat>(info.decodeSpeed)));
                if (!progress.get() || env->ExceptionCheck()) return;
                env->CallVoidMethod(callback, onProgress, progress.get());
            });
        };
        callbacks.onError = [&](const std::string& error) {
            guarded(env, [&] {
                LocalRef<jstring> text(env, newJString(env, error));
                if (!text.get() || env->ExceptionCheck()) return;
                env->CallVoidMethod(callback, onError, 0, text.get());
            });
        };
        callbacks.onIterationComplete = [&](const std::string& details) {
            if (env->ExceptionCheck()) return;
            mls::LlmSession::BenchmarkProgressInfo info;
            info.progress = -1;
            info.statusMessage = details;
            callbacks.onProgress(info);
        };
        const auto result = session->runBenchmark(backend, threads, useMmap, power, precision, memory,
                                                  dynamicOption, nPrompt, nGenerate, nRepeat, kvCache, callbacks);
        if (env->ExceptionCheck()) return nullptr;
        if (result.success) {
            LocalRef<jclass> instanceClass(env, env->GetObjectClass(testInstance));
            if (!instanceClass.get() || env->ExceptionCheck()) return nullptr;
            LocalRef<jclass> listClass(env, env->FindClass("java/util/List"));
            if (!listClass.get() || env->ExceptionCheck()) return nullptr;
            jmethodID add = env->GetMethodID(listClass.get(), "add", "(Ljava/lang/Object;)Z");
            if (!add || env->ExceptionCheck()) return nullptr;
            LocalRef<jclass> longClass(env, env->FindClass("java/lang/Long"));
            if (!longClass.get() || env->ExceptionCheck()) return nullptr;
            jmethodID ctor = env->GetMethodID(longClass.get(), "<init>", "(J)V");
            if (!ctor || env->ExceptionCheck()) return nullptr;
            auto append = [&](const char* getter, const std::vector<int64_t>& values) -> bool {
                jmethodID method = env->GetMethodID(instanceClass.get(), getter, "()Ljava/util/List;");
                if (!method || env->ExceptionCheck()) return false;
                LocalRef<jobject> list(env, env->CallObjectMethod(testInstance, method));
                if (env->ExceptionCheck()) return false;
                if (!list.get()) {
                    throwException(env, "java/lang/IllegalStateException", "Benchmark timing list is null");
                    return false;
                }
                for (int64_t value : values) {
                    LocalRef<jobject> boxed(env, env->NewObject(longClass.get(), ctor, static_cast<jlong>(value)));
                    if (!boxed.get() || env->ExceptionCheck()) return false;
                    env->CallBooleanMethod(list.get(), add, boxed.get());
                    if (env->ExceptionCheck()) return false;
                }
                return true;
            };
            if (!append("getPrefillUs", result.prefill_times_us) || !append("getDecodeUs", result.decode_times_us) ||
                !append("getSamplesUs", result.sample_times_us)) return nullptr;
        }
        return benchmarkResult(env, testInstance, result.success, result.error_message);
    });
}

}
