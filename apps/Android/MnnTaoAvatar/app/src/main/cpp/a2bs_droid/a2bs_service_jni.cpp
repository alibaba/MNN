#include <string>
#include <vector>
#include <memory>
#include "a2bs_service.hpp"
#include "a2bs_service_jni.h"
#include "jni_utils.h"
#include "common/file_utils.hpp"

static TaoAvatar::A2BSService *gA2BSService = nullptr;
std::mutex gA2BSMutex;
extern "C" {

JNIEXPORT jlong JNICALL
Java_com_taobao_meta_avatar_a2bs_A2BSService_nativeCreateA2BS(JNIEnv *env, jobject thiz) {
    if (!gA2BSService) {
        gA2BSService = new TaoAvatar::A2BSService();
    }
    return reinterpret_cast<jlong>(gA2BSService);
}

JNIEXPORT void JNICALL
Java_com_taobao_meta_avatar_a2bs_A2BSService_nativeDestroy(JNIEnv *env, jobject thiz, jlong nativePtr) {
    auto a2bsService = reinterpret_cast<TaoAvatar::A2BSService *>(nativePtr);
    delete a2bsService;
    gA2BSService = nullptr;
}

JNIEXPORT jboolean JNICALL
Java_com_taobao_meta_avatar_a2bs_A2BSService_nativeLoadA2bsResources(
        JNIEnv *env, jobject thiz, jlong nativePtr, jstring resourceDir, jstring tmp_path_j) {
    std::unique_lock<std::mutex> lock(gA2BSMutex);
    auto a2bsService = reinterpret_cast<TaoAvatar::A2BSService *>(nativePtr);
    const char *resourceDirCStr = env->GetStringUTFChars(resourceDir, nullptr);
    const char *tmp_path_c = env->GetStringUTFChars(tmp_path_j, nullptr);
    bool result = false;
    if (a2bsService) {
        result = a2bsService->LoadA2bsResources(resourceDirCStr, tmp_path_c);
    }
    env->ReleaseStringUTFChars(resourceDir, resourceDirCStr);
    env->ReleaseStringUTFChars(tmp_path_j, tmp_path_c);
    return result ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jobject JNICALL
Java_com_taobao_meta_avatar_a2bs_A2BSService_nativeProcessBuffer(JNIEnv *env, jobject thiz,
                                                                        jlong nativePtr,
                                                                        jint index,
                                                                        jshortArray jAudioData, jint sampleRate) {
    std::unique_lock<std::mutex> lock(gA2BSMutex);
    auto a2bsService = reinterpret_cast<TaoAvatar::A2BSService *>(nativePtr);

    jsize length = env->GetArrayLength(jAudioData);
    jshort* data = env->GetShortArrayElements(jAudioData, nullptr);

    TaoAvatar::AudioToBlendShapeData a2bsData = a2bsService->Process(index, (int16_t *)data, length, sampleRate);

    env->ReleaseShortArrayElements(jAudioData, data, JNI_ABORT);

    jclass a2bsDataClass = env->FindClass("com/taobao/meta/avatar/a2bs/AudioToBlendShapeData");
    jobject a2bsDataObj = env->NewObject(a2bsDataClass, env->GetMethodID(a2bsDataClass, "<init>", "()V"));

    jfieldID exprFieldID = env->GetFieldID(a2bsDataClass, "expr", "Ljava/util/List;");
    jfieldID poseZFieldID = env->GetFieldID(a2bsDataClass, "pose_z", "Ljava/util/List;");
    jfieldID poseFieldID = env->GetFieldID(a2bsDataClass, "pose", "Ljava/util/List;");
    jfieldID appPoseZFieldID = env->GetFieldID(a2bsDataClass, "app_pose_z", "Ljava/util/List;");
    jfieldID jawPoseFieldID = env->GetFieldID(a2bsDataClass, "jaw_pose", "Ljava/util/List;");
    jfieldID jointsTransformFieldID = env->GetFieldID(a2bsDataClass, "joints_transform","Ljava/util/List;");
    jfieldID frameNumFieldID = env->GetFieldID(a2bsDataClass, "frame_num", "I");

    jobject exprList = ConvertVector2DToJavaList(env, a2bsData.expr);
    jobject poseZList = ConvertVector2DToJavaList(env, a2bsData.pose_z);
    jobject poseList = ConvertVector2DToJavaList(env, a2bsData.pose);
    jobject appPoseZList = ConvertVector2DToJavaList(env, a2bsData.app_pose_z);
    jobject jawPoseList = ConvertVector2DToJavaList(env, a2bsData.jaw_pose);
    jobject jointsTransformList = ConvertVector2DToJavaList(env, a2bsData.joints_transform);

    env->SetObjectField(a2bsDataObj, exprFieldID, exprList);
    env->SetObjectField(a2bsDataObj, poseZFieldID, poseZList);
    env->SetObjectField(a2bsDataObj, poseFieldID, poseList);
    env->SetObjectField(a2bsDataObj, appPoseZFieldID, appPoseZList);
    env->SetObjectField(a2bsDataObj, jointsTransformFieldID, jointsTransformList);
    env->SetObjectField(a2bsDataObj, jawPoseFieldID, jawPoseList);
    env->SetIntField(a2bsDataObj, frameNumFieldID, static_cast<jint>(a2bsData.frame_num));
    return a2bsDataObj;
}

}