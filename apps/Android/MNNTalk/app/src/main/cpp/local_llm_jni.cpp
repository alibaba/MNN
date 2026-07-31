#include <android/log.h>
#include <jni.h>

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>

#include "llm/llm.hpp"

using MNN::Transformer::ChatMessages;
using MNN::Transformer::Llm;
using MNN::Transformer::LlmContext;
using MNN::Transformer::LlmStatus;

namespace {

constexpr const char* kLogTag = "MNNTalk";

void logError(const std::string& message) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "%s", message.c_str());
}

std::string escapeJsonString(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (char ch : value) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += ch;
                break;
        }
    }
    return escaped;
}

class Utf8StreamProcessor {
public:
    using Callback = std::function<void(const std::string&)>;

    explicit Utf8StreamProcessor(Callback callback) : mCallback(std::move(callback)) {}

    void process(const char* bytes, size_t length) {
        mBuffer.append(bytes, length);
        size_t offset = 0;
        std::string complete;
        while (offset < mBuffer.size()) {
            int charLength = utf8CharLength(static_cast<unsigned char>(mBuffer[offset]));
            if (charLength == 0 || offset + charLength > mBuffer.size()) {
                break;
            }
            complete.append(mBuffer, offset, charLength);
            offset += charLength;
        }
        mBuffer.erase(0, offset);
        if (!complete.empty()) {
            mCallback(complete);
        }
    }

private:
    static int utf8CharLength(unsigned char byte) {
        if ((byte & 0x80) == 0)
            return 1;
        if ((byte & 0xE0) == 0xC0)
            return 2;
        if ((byte & 0xF0) == 0xE0)
            return 3;
        if ((byte & 0xF8) == 0xF0)
            return 4;
        return 0;
    }

    Callback mCallback;
    std::string mBuffer;
};

class CallbackStreamBuffer : public std::streambuf {
public:
    using Callback = std::function<void(const char*, size_t)>;

    explicit CallbackStreamBuffer(Callback callback) : mCallback(std::move(callback)) {}

protected:
    std::streamsize xsputn(const char* bytes, std::streamsize length) override {
        if (mCallback) {
            mCallback(bytes, static_cast<size_t>(length));
        }
        return length;
    }

private:
    Callback mCallback;
};

void restoreRunningStatus(Llm* llm) {
    if (llm == nullptr || llm->getContext() == nullptr) {
        return;
    }
    auto* context = const_cast<LlmContext*>(llm->getContext());
    if (context->status == LlmStatus::MAX_TOKENS_FINISHED || context->status == LlmStatus::NORMAL_FINISHED) {
        context->status = LlmStatus::RUNNING;
    }
}

class LocalLlmSession {
public:
    LocalLlmSession(std::string configPath, std::string cachePath, std::string systemPrompt, int maxNewTokens)
        : mSystemPrompt(std::move(systemPrompt)), mMaxNewTokens(maxNewTokens > 0 ? maxNewTokens : 256) {
        mLlm.reset(Llm::createLLM(configPath));
        if (!mLlm) {
            return;
        }

        const std::string runtimeConfig =
            "{\"backend_type\":\"cpu\",\"thread_num\":4,\"precision\":\"low\","
            "\"memory\":\"low\",\"power\":\"high\",\"async\":false,"
            "\"prompt_cache\":true,\"use_mmap\":true,\"tmp_path\":\"" +
            escapeJsonString(cachePath) + "\",\"jinja\":{\"context\":{\"enable_thinking\":false}}}";
        if (!mLlm->set_config(runtimeConfig)) {
            logError("Failed to apply local voice runtime configuration");
            mLlm.reset();
            return;
        }
        if (!mLlm->load()) {
            logError("Failed to load MNN LLM");
            mLlm.reset();
            return;
        }
        resetMessages();
    }

    bool isReady() const { return mLlm != nullptr; }

    std::array<int64_t, 4> generate(JNIEnv* env, const std::string& prompt, jobject listener) {
        std::lock_guard<std::mutex> lock(mGenerateMutex);
        if (!mLlm) {
            return {0, 0, 0, 0};
        }

        mStopRequested.store(false);
        restoreRunningStatus(mLlm.get());
        mMessages.emplace_back("user", prompt);

        const jclass listenerClass = env->GetObjectClass(listener);
        const jmethodID onToken = env->GetMethodID(listenerClass, "onToken", "(Ljava/lang/String;)Z");
        if (onToken == nullptr) {
            logError("LocalLlm.TokenListener.onToken was not found");
            env->DeleteLocalRef(listenerClass);
            return {0, 0, 0, 0};
        }

        std::stringstream response;
        bool pendingEnd = false;
        Utf8StreamProcessor utf8Processor([&](const std::string& chunk) {
            const size_t endPosition = chunk.find("<eop>");
            const std::string visible = endPosition == std::string::npos ? chunk : chunk.substr(0, endPosition);
            if (!visible.empty()) {
                response << visible;
                jstring javaToken = env->NewStringUTF(visible.c_str());
                const jboolean shouldStop = env->CallBooleanMethod(listener, onToken, javaToken);
                env->DeleteLocalRef(javaToken);
                if (env->ExceptionCheck()) {
                    env->ExceptionDescribe();
                    env->ExceptionClear();
                    mStopRequested.store(true);
                } else if (shouldStop == JNI_TRUE) {
                    mStopRequested.store(true);
                }
            }
            if (endPosition != std::string::npos) {
                pendingEnd = true;
            }
        });
        CallbackStreamBuffer streamBuffer(
            [&](const char* bytes, size_t length) { utf8Processor.process(bytes, length); });
        std::ostream output(&streamBuffer);

        mLlm->response(mMessages, &output, "<eop>", 0);
        int generated = 0;
        resolveIntermediateEnd(pendingEnd, generated);
        while (!mStopRequested.load() && !pendingEnd && generated < mMaxNewTokens) {
            mLlm->generate(1);
            ++generated;
            resolveIntermediateEnd(pendingEnd, generated);
        }

        const std::string assistant = response.str();
        if (!assistant.empty()) {
            mMessages.emplace_back("assistant", assistant);
            mLlm->syncPromptCache(mMessages);
        }

        const LlmContext* context = mLlm->getContext();
        env->DeleteLocalRef(listenerClass);
        if (context == nullptr) {
            return {0, generated, 0, 0};
        }
        return {context->prompt_len, context->gen_seq_len, context->prefill_us, context->decode_us};
    }

    void stop() { mStopRequested.store(true); }

    void reset() {
        stop();
        std::lock_guard<std::mutex> lock(mGenerateMutex);
        if (mLlm) {
            mLlm->reset();
            resetMessages();
        }
    }

    void waitUntilIdle() { std::lock_guard<std::mutex> lock(mGenerateMutex); }

private:
    void resetMessages() {
        mMessages.clear();
        mMessages.emplace_back("system", mSystemPrompt);
    }

    void resolveIntermediateEnd(bool& pendingEnd, int generated) {
        if (!mLlm || !mLlm->getContext()) {
            pendingEnd = true;
            return;
        }
        const auto status = mLlm->getContext()->status;
        if (status == LlmStatus::MAX_TOKENS_FINISHED && !mStopRequested.load() && generated < mMaxNewTokens) {
            restoreRunningStatus(mLlm.get());
            pendingEnd = false;
            return;
        }
        if (status == LlmStatus::NORMAL_FINISHED && !pendingEnd && !mStopRequested.load() &&
            generated < mMaxNewTokens) {
            restoreRunningStatus(mLlm.get());
        }
    }

    std::unique_ptr<Llm> mLlm;
    ChatMessages mMessages;
    std::string mSystemPrompt;
    int mMaxNewTokens;
    std::atomic<bool> mStopRequested{false};
    std::mutex mGenerateMutex;
};

std::string fromJString(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return {};
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    std::string result(chars != nullptr ? chars : "");
    if (chars != nullptr) {
        env->ReleaseStringUTFChars(value, chars);
    }
    return result;
}

jlongArray toJavaMetrics(JNIEnv* env, const std::array<int64_t, 4>& values) {
    jlongArray result = env->NewLongArray(values.size());
    jlong raw[4] = {values[0], values[1], values[2], values[3]};
    env->SetLongArrayRegion(result, 0, values.size(), raw);
    return result;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL Java_com_alibaba_mnntalk_engine_LocalLlm_nativeCreate(
    JNIEnv* env, jclass, jstring configPath, jstring cachePath, jstring systemPrompt, jint maxNewTokens) {
    auto* session = new LocalLlmSession(fromJString(env, configPath), fromJString(env, cachePath),
                                        fromJString(env, systemPrompt), maxNewTokens);
    if (!session->isReady()) {
        delete session;
        return 0;
    }
    return reinterpret_cast<jlong>(session);
}

extern "C" JNIEXPORT jlongArray JNICALL Java_com_alibaba_mnntalk_engine_LocalLlm_nativeGenerate(JNIEnv* env, jobject,
                                                                                                jlong handle,
                                                                                                jstring prompt,
                                                                                                jobject listener) {
    auto* session = reinterpret_cast<LocalLlmSession*>(handle);
    if (session == nullptr || listener == nullptr) {
        return toJavaMetrics(env, {0, 0, 0, 0});
    }
    return toJavaMetrics(env, session->generate(env, fromJString(env, prompt), listener));
}

extern "C" JNIEXPORT void JNICALL Java_com_alibaba_mnntalk_engine_LocalLlm_nativeStop(JNIEnv*, jobject, jlong handle) {
    auto* session = reinterpret_cast<LocalLlmSession*>(handle);
    if (session != nullptr) {
        session->stop();
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_alibaba_mnntalk_engine_LocalLlm_nativeReset(JNIEnv*, jobject, jlong handle) {
    auto* session = reinterpret_cast<LocalLlmSession*>(handle);
    if (session != nullptr) {
        session->reset();
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_alibaba_mnntalk_engine_LocalLlm_nativeRelease(JNIEnv*, jobject,
                                                                                         jlong handle) {
    auto* session = reinterpret_cast<LocalLlmSession*>(handle);
    if (session != nullptr) {
        session->stop();
        session->waitUntilIdle();
        delete session;
    }
}
