#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>
#include <string>
#include <vector>
#include <sstream>
#include <thread>

#include "llm/llm.hpp"

static std::unique_ptr<MNN::Transformer::Llm> llm(nullptr);
static std::stringstream response_buffer;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnUnload");
}

using PromptItem = std::pair<std::string, std::string>;
static std::vector<PromptItem> history{};

JNIEXPORT jboolean JNICALL Java_com_mnn_llm_Chat_Init(JNIEnv* env, jobject thiz, jstring modelDir) {
    const char* model_dir = env->GetStringUTFChars(modelDir, 0);
    history.clear();
    history.emplace_back("system", "You are a helpful assistant.");
    if (!llm.get()) {
        llm.reset(MNN::Transformer::Llm::createLLM(model_dir));
        llm->load();
    }
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_mnn_llm_Chat_Ready(JNIEnv* env, jobject thiz) {
    if (llm.get()) {
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_mnn_llm_Chat_Submit(JNIEnv* env, jobject thiz, jstring inputStr) {
    if (!llm.get()) {
        return env->NewStringUTF("Failed, Chat is not ready!");
    }
    const char* input_str = env->GetStringUTFChars(inputStr, 0);
    auto chat = [&](std::vector<PromptItem> hist) {
        llm->response(hist, &response_buffer, "<eop>");
    };
    history.emplace_back("user", input_str);
    std::thread chat_thread(chat, history);
    chat_thread.detach();
    jstring result = env->NewStringUTF("Submit success!");
    return result;
}

JNIEXPORT jbyteArray JNICALL Java_com_mnn_llm_Chat_Response(JNIEnv* env, jobject thiz) {
    auto len = response_buffer.str().size();
    jbyteArray res = env->NewByteArray(len);
    env->SetByteArrayRegion(res, 0, len, (const jbyte*)response_buffer.str().c_str());
    return res;
}

JNIEXPORT jfloat JNICALL Java_com_mnn_llm_Chat_Done(JNIEnv* env, jobject thiz) {
    std::string response_result =  response_buffer.str();
    history.emplace_back("assistant", response_result);
    response_buffer.str("");
    auto& state = llm->getState();
    int64_t total_len = 0;
    int64_t total_time = 0;
    total_len += state.prompt_len_;
    total_len += state.gen_seq_len_;
    total_time += state.vision_us_;
    total_time += state.audio_us_;
    total_time += state.prefill_us_;
    total_time += state.decode_us_;
    if(total_time == 0)return 0.0;
    float speed = (float)total_len*1000000/(float)total_time;
    return (jfloat)speed;
}

JNIEXPORT void JNICALL Java_com_mnn_llm_Chat_Reset(JNIEnv* env, jobject thiz) {
    history.resize(1);
    llm->reset();
}

} // extern "C"