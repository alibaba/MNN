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
    /*if (!llm.get()) {
        llm.reset(MNN::Transformer::Llm::createLLM(model_dir));
        llm->load();
    }*/
    llm.reset(MNN::Transformer::Llm::createLLM(model_dir));
    llm->load();
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

JNIEXPORT jfloatArray JNICALL Java_com_mnn_llm_Chat_Done(JNIEnv* env, jobject thiz) {
    jfloatArray result= env->NewFloatArray(2);
    jfloat fill[2];
    fill[0] = 0.0f;
    fill[1] = 0.0f;
    std::string response_result =  response_buffer.str();
    history.emplace_back("assistant", response_result);
    response_buffer.str("");
    auto& state = llm->getState();
    auto prefill_s = state.prefill_us_ * 1e-6;
    auto decode_s  = state.decode_us_ * 1e-6;
    if(prefill_s != 0 && decode_s != 0){
        fill[0]  = state.prompt_len_ / prefill_s;
        fill[1] = state.gen_seq_len_ / decode_s;
    }
    env->SetFloatArrayRegion(result, 0, 2, fill);
    return result;
}

JNIEXPORT void JNICALL Java_com_mnn_llm_Chat_Reset(JNIEnv* env, jobject thiz) {
    history.resize(1);
    llm->reset();
}

}