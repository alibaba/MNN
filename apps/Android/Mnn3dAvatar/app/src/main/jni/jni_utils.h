#pragma once
#include <jni.h>
#include <vector>
#include "common/file_utils.hpp"
#include "common/Common.hpp"

extern "C" {
    jobject ConvertVector2DToJavaList(JNIEnv *env, const std::vector<std::vector<float>>& inputVector);
    void CopyToNativeCameraControlData(JNIEnv *env, jobject jCameraControlData, TaoAvatar::CameraControlData *cameraControlData);
}