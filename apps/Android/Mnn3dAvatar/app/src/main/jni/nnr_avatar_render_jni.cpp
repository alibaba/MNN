#include <android/native_window_jni.h>
#include "nnr_avatar_render.hpp"
#include "../cpp/common/Camera.hpp"
#include <string>
#include "mh_log.hpp"
#include "jni_utils.h"

static TaoAvatar::NnrAvatarRender *gRuntime = nullptr;
std::mutex gNNRMutex;

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_taobao_meta_avatar_nnr_NnrAvatarRender_nativeCreateNNR(JNIEnv *env, jclass clazz) {
    if (!gRuntime) {
        gRuntime = new TaoAvatar::NnrAvatarRender();
    }
    return reinterpret_cast<jlong>(gRuntime);
}

JNIEXPORT void JNICALL
Java_com_taobao_meta_avatar_nnr_NnrAvatarRender_nativeInitNNR(JNIEnv *env,
                                                              jclass clazz,
                                                              jlong nativePtr,
                                                              jobject surface,
                                                              jstring nnrRootPath,
                                                              jstring cacheDirPath) {
    auto *runtime = reinterpret_cast<TaoAvatar::NnrAvatarRender *>(nativePtr);
    const char* nnrRootPathCstr = env->GetStringUTFChars(nnrRootPath, nullptr);
    const char* cacheDirPathCstr = env->GetStringUTFChars(cacheDirPath, nullptr);
    ANativeWindow *nativeWindow = ANativeWindow_fromSurface(env, surface);
    if (nativeWindow == nullptr) {
        MH_ERROR("Failed to get ANativeWindow from Surface.");
        return;
    }
    if (runtime) {
        runtime->InitNNR(nativeWindow, nnrRootPathCstr, cacheDirPathCstr);
    }
    env->ReleaseStringUTFChars(nnrRootPath, nnrRootPathCstr);
    env->ReleaseStringUTFChars(cacheDirPath, cacheDirPathCstr);
}

JNIEXPORT void JNICALL
Java_com_taobao_meta_avatar_nnr_NnrAvatarRender_nativeUpdateNNRScene(JNIEnv *env,
                                                                     jclass clazz,
                                                                     jlong nativePtr,
                                                                     jobject
                                                                       jCameraControlData,
                                                                     jboolean isPlaying,
                                                                     jlong currentPlayTime,
                                                                     jlong totalTime,
                                                                     jboolean isBuffering,
                                                                     jfloat smoothToIdlePercent,
                                                                     jfloat smoothToTalkPercent,
                                                                     jlong forceFrameIndex) {
    auto *runtime = reinterpret_cast<TaoAvatar::NnrAvatarRender *>(nativePtr);

    TaoAvatar::CameraControlData cameraControlData;
    CopyToNativeCameraControlData(env, jCameraControlData, &cameraControlData);

    runtime->UpdateNNRScene(cameraControlData,
                            isPlaying,
                            (uint64_t) currentPlayTime,
                            (uint64_t) totalTime,
                            isBuffering,
                            smoothToIdlePercent,
                            smoothToTalkPercent,
                            forceFrameIndex);
}

JNIEXPORT jboolean JNICALL
Java_com_taobao_meta_avatar_nnr_NnrAvatarRender_nativeLoadNnrResources(JNIEnv* env, jclass clazz, jlong nativePtr,
                                                                       jstring nnrRootPath,
                                                                       jstring computeSceneFileName,
                                                                       jstring renderSceneFileName,
                                                                       jstring skyboxSceneFileName,
                                                                       jstring deformParamFileName,
                                                                       jstring chatStatusFileName) {
    std::unique_lock<std::mutex> lock(gNNRMutex);
    auto *runtime = reinterpret_cast<TaoAvatar::NnrAvatarRender *>(nativePtr);
    const char *nnrRootPathCStr = env->GetStringUTFChars(nnrRootPath, nullptr);
    const char *computeSceneCStr = env->GetStringUTFChars(computeSceneFileName, nullptr);
    const char *renderSceneCStr = env->GetStringUTFChars(renderSceneFileName, nullptr);
    const char *skyboxSceneCStr = env->GetStringUTFChars(skyboxSceneFileName, nullptr);
    const char *deformParamCStr = env->GetStringUTFChars(deformParamFileName, nullptr);
    const char *chatStatusFileNameCStr = env->GetStringUTFChars(chatStatusFileName, nullptr);

    bool result = false;
    if (runtime) {
        result = runtime->LoadNnrResourcesFromFile(
                 nnrRootPathCStr,
                 computeSceneCStr, renderSceneCStr, skyboxSceneCStr,
                                                   deformParamCStr,chatStatusFileNameCStr);
    }

    env->ReleaseStringUTFChars(computeSceneFileName, computeSceneCStr);
    env->ReleaseStringUTFChars(renderSceneFileName, renderSceneCStr);
    env->ReleaseStringUTFChars(skyboxSceneFileName, skyboxSceneCStr);
    env->ReleaseStringUTFChars(deformParamFileName, deformParamCStr);
    env->ReleaseStringUTFChars(chatStatusFileName, chatStatusFileNameCStr);
    env->ReleaseStringUTFChars(nnrRootPath, nnrRootPathCStr);

    return result ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_taobao_meta_avatar_nnr_NnrAvatarRender_nativeDestroy(JNIEnv *env, jclass clazz, jlong nativePtr) {
    auto *runtime = reinterpret_cast<TaoAvatar::NnrAvatarRender *>(nativePtr);
    if (runtime) {
        runtime->DestroyNNR();
        delete runtime;
        gRuntime = nullptr;
    }
}

JNIEXPORT void JNICALL
Java_com_taobao_meta_avatar_nnr_NnrAvatarRender_nativeRender(JNIEnv *env, jclass clazz, jlong nativePtr) {
    auto *runtime = reinterpret_cast<TaoAvatar::NnrAvatarRender *>(nativePtr);
    if (runtime) {
        runtime->Render();
    }
}

JNIEXPORT jboolean JNICALL
Java_com_taobao_meta_avatar_nnr_NnrAvatarRender_nativeIsNNRReady(JNIEnv *env, jclass clazz, jlong nativePtr) {
    auto *runtime = reinterpret_cast<TaoAvatar::NnrAvatarRender *>(nativePtr);
    if (runtime) {
        return runtime->IsNNRReady() ? JNI_TRUE : JNI_FALSE;
    }
    return JNI_FALSE;
}

}
extern "C"
JNIEXPORT void JNICALL
Java_com_taobao_meta_avatar_nnr_NnrAvatarRender_nativeReset(JNIEnv *env, jclass clazz,
                                                            jlong native_ptr) {
    auto *runtime = reinterpret_cast<TaoAvatar::NnrAvatarRender *>(native_ptr);
    if (runtime) {
         runtime->Reset();
    }
}