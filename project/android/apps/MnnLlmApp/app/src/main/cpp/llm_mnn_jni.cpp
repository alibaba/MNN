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
#include "llm/llm.hpp"

#include <sstream>
#include <mutex>
#include <string>
#include "diffusion_session.h"
#include <chrono>
#include "mls_log.h"
using MNN::Transformer::Llm;
using mls::DiffusionSession;

class MNN_PUBLIC LlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char* str, size_t len)>;;
    explicit LlmStreamBuffer(CallBack callback) : callback_(std::move(callback)) {}

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (callback_) {
            callback_(s, n);
        }
        return n;
    }

private:
    CallBack callback_ = nullptr;
};

using PromptItem = std::pair<std::string, std::string>;
static std::vector<PromptItem> history{};
static bool stop_requested = false;
static bool is_r1 = false;
static std::string prompt_string_for_debug{};
static std::string response_string_for_debug{};

std::string trimLeadingWhitespace(const std::string& str) {
    auto it = std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch); // Find the first non-whitespace character
    });
    return std::string(it, str.end()); // Create a substring from the first non-whitespace character
}
int utf8CharLength(unsigned char byte) {
    if ((byte & 0x80) == 0) return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return 0;
}

class Utf8StreamProcessor {
public:
    explicit Utf8StreamProcessor(std::function<void(const std::string&)> callback)
            : callback(std::move(callback)) {}

    void processStream(const char* str, size_t len) {
        utf8Buffer.append(str, len);

        size_t i = 0;
        std::string completeChars;
        while (i < utf8Buffer.size()) {
            int length = utf8CharLength(static_cast<unsigned char>(utf8Buffer[i]));
            if (length == 0 || i + length > utf8Buffer.size()) {
                break;
            }
            completeChars.append(utf8Buffer, i, length);
            i += length;
        }
        utf8Buffer = utf8Buffer.substr(i);
        if (!completeChars.empty()) {
            callback(completeChars);
        }
    }

private:
    std::string utf8Buffer;
    std::function<void(const std::string&)> callback;
};

//for_history true for insert to history, false for submit prompt
const char* getUserString(const char* user_content, bool for_history) {
    if (is_r1) {
        return ("<|User|>" + std::string(user_content) + "<|Assistant|>" + (for_history ? "" : "<think>\n")).c_str();
    } else {
        return user_content;
    }
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnUnload");
}

JNIEXPORT jlong JNICALL Java_com_alibaba_mnnllm_android_ChatSession_initNative(JNIEnv* env, jobject thiz, jstring modelDir,
                                                                                    jboolean use_tmp_path,
                                                                                    jobject chat_history,
                                                                                    jboolean is_diffusion,
                                                                                    jboolean r1) {
    is_r1 = r1;
    const char* model_dir = env->GetStringUTFChars(modelDir, 0);
    MNN_DEBUG("createLLM BeginLoad %s", model_dir);
    if (is_diffusion) {
        auto diffusion = new DiffusionSession(model_dir);
        return reinterpret_cast<jlong>(diffusion);
    }
    auto llm = Llm::createLLM(model_dir);
    if (use_tmp_path) {
        auto model_dir_str = std::string(model_dir);
        std::string model_dir_parent = model_dir_str.substr(0, model_dir_str.find_last_of('/'));
        std::string temp_dir = model_dir_parent + R"(/tmp")";
        auto extra_config = R"({"tmp_path":")" + temp_dir;
        if (is_r1) {
            extra_config += R"(,"use_template":false)";
        }
        extra_config = extra_config +  R"(,"reuse_kv":true, "backend_type":"cpu", "use_mmap":true})";
        MNN_DEBUG("extra_config: %s", extra_config.c_str());
        llm->set_config(extra_config);
        MNN_DEBUG("dumped config: %s", llm->dump_config().c_str());
    }
    history.clear();
    history.emplace_back("system", is_r1 ? "<|begin_of_sentence|>You are a helpful assistant." : "You are a helpful assistant.");
    if (chat_history != nullptr) {
        jclass listClass = env->GetObjectClass(chat_history);
        jmethodID sizeMethod = env->GetMethodID(listClass, "size", "()I");
        jmethodID getMethod = env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");
        jint listSize = env->CallIntMethod(chat_history, sizeMethod);
        for (jint i = 0; i < listSize; i++) {
            jobject element = env->CallObjectMethod(chat_history, getMethod, i);
            const char *elementCStr = env->GetStringUTFChars((jstring)element, nullptr);
            if (is_r1) {
                if (i % 2 == 0) {
                    history.emplace_back("user",getUserString(elementCStr, true));
                } else {
                    history.emplace_back("assistant",elementCStr);
                }
            } else {
                history.emplace_back(i % 2 == 0 ? "user" : "assistant",elementCStr);
            }
            env->ReleaseStringUTFChars((jstring)element, elementCStr);
            env->DeleteLocalRef(element);
        }
    }
    llm->load();
    MNN_DEBUG("createLLM EndLoad %ld ", reinterpret_cast<jlong>(llm));
    return reinterpret_cast<jlong>(llm);
}



JNIEXPORT jobject JNICALL Java_com_alibaba_mnnllm_android_ChatSession_submitNative(JNIEnv* env, jobject thiz,
                                                                                   jlong llmPtr, jstring inputStr,jboolean keepHistory,
                                                                                   jobject progressListener) {
    Llm* llm = reinterpret_cast<Llm*>(llmPtr);
    if (!llm) {
        return env->NewStringUTF("Failed, Chat is not ready!");
    }
    prompt_string_for_debug.clear();
    response_string_for_debug.clear();
    stop_requested = false;
    if (!keepHistory) {
        history.resize(1);
    }
    const char* input_str = env->GetStringUTFChars(inputStr, nullptr);
    std::stringstream response_buffer;
    jclass progressListenerClass = env->GetObjectClass(progressListener);
    jmethodID onProgressMethod = env->GetMethodID(progressListenerClass, "onProgress", "(Ljava/lang/String;)Z");
    if (!onProgressMethod) {
        MNN_DEBUG("ProgressListener onProgress method not found.");
    }
    Utf8StreamProcessor processor([&response_buffer, env, progressListener, onProgressMethod](const std::string& utf8Char) {
        bool is_eop = utf8Char.find("<eop>") != std::string::npos;
        if (!is_eop) {
            response_buffer << utf8Char;
        } else {
            std::string response_result =  response_buffer.str();
            MNN_DEBUG("submitNative Result %s", response_result.c_str());
            response_string_for_debug = response_result;
            if (is_r1) {
                auto& last_message = history.at(history.size() - 1);
                std::size_t user_think_pos = last_message.second.find("<think>\n");
                if (user_think_pos != std::string::npos) {
                    last_message.second.erase(user_think_pos, std::string("<think>\n").length());
                }
                std::size_t pos = response_result.find("</think>");
                if (pos != std::string::npos) {
                    response_result.erase(0, pos + std::string("</think>").length());
                }
                response_result = trimLeadingWhitespace(response_result) + "<|end_of_sentence|>";
            }
            history.emplace_back("assistant", response_result);
        }
        if (progressListener && onProgressMethod) {
            jstring javaString = is_eop ? nullptr : env->NewStringUTF(utf8Char.c_str());
            jboolean user_stop_requested = env->CallBooleanMethod(progressListener, onProgressMethod,  javaString);
            stop_requested = is_eop || user_stop_requested;
            env->DeleteLocalRef(javaString);
        }
    });
    LlmStreamBuffer stream_buffer{[&processor](const char* str, size_t len){
        processor.processStream(str, len);
    }};
    std::ostream output_ostream(&stream_buffer);
    history.emplace_back("user", getUserString(input_str, false));
    MNN_DEBUG("submitNative history count %zu", history.size());
    for (auto iter= history.begin(); iter != history.end(); ++iter) {
        prompt_string_for_debug += iter->second;
    }
    MNN_DEBUG("submitNative prompt_string_for_debug count %s", prompt_string_for_debug.c_str());
    llm->response(history, &output_ostream, "<eop>", 1);
    while (!stop_requested) {
        llm->generate(1);
    }
    auto& state = llm->getState();
    int64_t prompt_len = 0;
    int64_t decode_len = 0;
    int64_t vision_time = 0;
    int64_t audio_time = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    prompt_len += state.prompt_len_;
    decode_len += state.gen_seq_len_;
    vision_time += state.vision_us_;
    audio_time += state.audio_us_;
    prefill_time += state.prefill_us_;
    decode_time += state.decode_us_;
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


JNIEXPORT void JNICALL Java_com_alibaba_mnnllm_android_ChatSession_resetNative(JNIEnv* env, jobject thiz, jlong llmPtr) {
    history.resize(1);
    Llm* llm = reinterpret_cast<Llm*>(llmPtr);
    if (llm) {
        llm->reset();
    }
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_alibaba_mnnllm_android_ChatSession_getDebugInfoNative(JNIEnv *env, jobject thiz, jlong objecPtr) {
    return env->NewStringUTF(("last_prompt:\n" + prompt_string_for_debug + "\nlast_response:\n" + response_string_for_debug).c_str());
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
        Llm* llm = reinterpret_cast<Llm*>(objecPtr);
        delete llm;
    }
}

JNIEXPORT jobject JNICALL
Java_com_alibaba_mnnllm_android_ChatSession_submitDiffusionNative(JNIEnv *env, jobject thiz,
                                                                       jlong instance_id,
                                                                       jstring input,
                                                                       jstring joutput_path,
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
    diffusion->Run(prompt, output_path, [env, progressListener, onProgressMethod](int progress) {
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