#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>
#include <string>
#include <vector>
#include <sstream>
#include <thread>

#include "app/chat/chat.hpp"

using namespace MNN;
using namespace MNN::Transformer;

static std::unique_ptr<Chat> agent(nullptr);
static std::stringstream response_buffer;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnUnload");
}

JNIEXPORT jboolean JNICALL Java_com_mnn_llm_Chat_Init(JNIEnv* env, jobject thiz, jstring modelDir) {
    const char* model_dir = env->GetStringUTFChars(modelDir, 0);
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "model path: %s", model_dir);
    if (!agent.get()) {
        agent.reset(new Chat(model_dir));
        __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "Model Loaded!");
        agent->chat_init();
        __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "Chat Loaded!");
    }
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_mnn_llm_Chat_Ready(JNIEnv* env, jobject thiz) {
    if (agent.get()) {
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_mnn_llm_Chat_Submit(JNIEnv* env, jobject thiz, jstring inputStr) {
    if (!agent.get()) {
        return env->NewStringUTF("Failed, Chat is not ready!");
    }
    const char* input_str = env->GetStringUTFChars(inputStr, 0);
    auto chat = [&](std::string str) {
        agent->getAnswer(str, &response_buffer, "<eop>");
    };
    std::thread chat_thread(chat, input_str);
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

JNIEXPORT void JNICALL Java_com_mnn_llm_Chat_Done(JNIEnv* env, jobject thiz) {
    response_buffer.str("");
}

JNIEXPORT void JNICALL Java_com_mnn_llm_Chat_Reset(JNIEnv* env, jobject thiz) {
    agent->chat_init();
}

} // extern "C"