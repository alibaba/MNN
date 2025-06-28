#include <memory>
#include <functional>
#include "tts_service.hpp"
#include <jni.h>

static TaoAvatar::TTSService *gTTSService = nullptr;
std::mutex gTTSMutex;

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_taobao_meta_avatar_tts_TtsService_nativeCreateTTS(JNIEnv *env, jobject thiz, jstring language) {
    auto languageCStr = env->GetStringUTFChars(language, nullptr);
    if (!gTTSService) {
        gTTSService = new TaoAvatar::TTSService(languageCStr);
    }
    env->ReleaseStringUTFChars(language, languageCStr);
    return reinterpret_cast<jlong>(gTTSService);
}

JNIEXPORT void JNICALL
Java_com_taobao_meta_avatar_tts_TtsService_nativeDestroy(JNIEnv *env, jobject thiz, jlong nativePtr) {
    auto ttsService = reinterpret_cast<TaoAvatar::TTSService *>(nativePtr);
    delete ttsService;
    gTTSService = nullptr;
}

JNIEXPORT jboolean JNICALL
Java_com_taobao_meta_avatar_tts_TtsService_nativeLoadResourcesFromFile(JNIEnv *env,
                                                                       jobject thiz,
                                                                       jlong nativePtr,
                                                                       jstring resourceDir,
                                                                       jstring modelName,
                                                                       jstring cacheDir) {
    std::unique_lock<std::mutex> lock(gTTSMutex);
    auto ttsService = reinterpret_cast<TaoAvatar::TTSService *>(nativePtr);
    const char *resourceDirCStr = env->GetStringUTFChars(resourceDir, nullptr);
    const char *modelNameCStr = env->GetStringUTFChars(modelName, nullptr);
    const char *cacheDirCStr = env->GetStringUTFChars(cacheDir, nullptr);
    bool result = false;
    if (ttsService) {
        result = ttsService->LoadTtsResources(resourceDirCStr, modelNameCStr, cacheDirCStr);
    }
    env->ReleaseStringUTFChars(modelName, modelNameCStr);
    env->ReleaseStringUTFChars(resourceDir, resourceDirCStr);
    env->ReleaseStringUTFChars(cacheDir, cacheDirCStr);
    return result ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jshortArray JNICALL
Java_com_taobao_meta_avatar_tts_TtsService_nativeProcess(JNIEnv *env, jobject thiz, jlong nativePtr,
                                                         jstring text, jint id) {
    std::unique_lock<std::mutex> lock(gTTSMutex);
    auto ttsService = reinterpret_cast<TaoAvatar::TTSService *>(nativePtr);
    const char *textCStr = env->GetStringUTFChars(text, nullptr);
    std::vector<int16_t> samples = ttsService->Process(textCStr, id);
    jshortArray samplesArray = env->NewShortArray(samples.size());
    if (samplesArray != nullptr) {
        env->SetShortArrayRegion(samplesArray, 0, samples.size(),samples.data());
    }
    env->ReleaseStringUTFChars(text, textCStr);
    return samplesArray;
}

JNIEXPORT void JNICALL
Java_com_taobao_meta_avatar_tts_TtsService_nativeSetCurrentIndex(JNIEnv *env, jobject thiz,
                                                                 jlong tts_service_native,
                                                                 jint index) {
    auto tts_service = reinterpret_cast<TaoAvatar::TTSService *>(tts_service_native);
    tts_service->SetIndex(index);
}

}