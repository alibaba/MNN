#include "jni_utils.h"
#include "mh_log.hpp"
extern "C" {

jobject ConvertVector2DToJavaList(JNIEnv *env, const std::vector<std::vector<float>>& inputVector) {
    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListInitMethodID = env->GetMethodID(arrayListClass, "<init>", "()V");
    jmethodID arrayListAddMethodID = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");

    jobject outputList = env->NewObject(arrayListClass, arrayListInitMethodID);
    for (const auto& innerVector : inputVector) {
        jfloatArray jInnerArray = env->NewFloatArray(innerVector.size());
        env->SetFloatArrayRegion(jInnerArray, 0, innerVector.size(), innerVector.data());
        env->CallBooleanMethod(outputList, arrayListAddMethodID, jInnerArray);
        env->DeleteLocalRef(jInnerArray);
    }

    env->DeleteLocalRef(arrayListClass);
    return outputList;
}

void CopyToNativeCameraControlData(JNIEnv *env, jobject jCameraControlData, TaoAvatar::CameraControlData *cameraControlData) {
    if (env == nullptr || cameraControlData == nullptr) {
        MH_ERROR("Failed to copy camera data.");
        return;
    }

    jclass cameraControlDataClass = env->GetObjectClass(jCameraControlData);

    jfieldID curScaleFieldID = env->GetFieldID(cameraControlDataClass, "curScale", "F");
    jfieldID lastScaleFieldID = env->GetFieldID(cameraControlDataClass, "lastScale", "F");
    jfieldID rotateXFieldID = env->GetFieldID(cameraControlDataClass, "rotateX", "[F");
    jfieldID rotateYFieldID = env->GetFieldID(cameraControlDataClass, "rotateY", "[F");
    jfieldID scaleMatrixFieldID = env->GetFieldID(cameraControlDataClass, "scaleMatrix", "[F");
    jfieldID modelMatrixFieldID = env->GetFieldID(cameraControlDataClass, "modelMatrix", "[F");
    jfieldID rotateSpeedFieldID = env->GetFieldID(cameraControlDataClass, "rotateSpeed", "F");
    jfieldID isFirstFrameFieldID = env->GetFieldID(cameraControlDataClass, "isFirstFrame", "Z");
    jfieldID cameraFieldID = env->GetFieldID(cameraControlDataClass, "camera", "Lcom/taobao/meta/avatar/camera/Camera;");
    jfieldID distanceOnScreenFieldID = env->GetFieldID(cameraControlDataClass, "distanceOnScreen", "F");

    cameraControlData->curScale = env->GetFloatField(jCameraControlData, curScaleFieldID);
    cameraControlData->lastScale = env->GetFloatField(jCameraControlData, lastScaleFieldID);

    jfloatArray rotateXArray = (jfloatArray) env->GetObjectField(jCameraControlData, rotateXFieldID);
    jsize rotateXSize = env->GetArrayLength(rotateXArray);
    assert(rotateXSize == 16);
    jfloat *rotateXPtr = env->GetFloatArrayElements(rotateXArray, nullptr);
    cameraControlData->rotateX = glm::make_mat4(rotateXPtr);

    jfloatArray rotateYArray = (jfloatArray) env->GetObjectField(jCameraControlData, rotateYFieldID);
    jsize rotateYSize = env->GetArrayLength(rotateXArray);
    assert(rotateYSize == 16);
    jfloat *rotateYPtr = env->GetFloatArrayElements(rotateYArray, nullptr);
    cameraControlData->rotateY = glm::make_mat4(rotateYPtr);

    jfloatArray scaleMatrixArray = (jfloatArray) env->GetObjectField(jCameraControlData, scaleMatrixFieldID);
    jsize scaleMatrixSize = env->GetArrayLength(scaleMatrixArray);
    assert(scaleMatrixSize == 16);
    jfloat *scaleMatrixPtr = env->GetFloatArrayElements(scaleMatrixArray, nullptr);
    cameraControlData->scaleMatrix = glm::make_mat4(scaleMatrixPtr);

    jfloatArray modelMatrixArray = (jfloatArray) env->GetObjectField(jCameraControlData, modelMatrixFieldID);
    jsize modelMatrixSize = env->GetArrayLength(modelMatrixArray);
    assert(modelMatrixSize == 16);
    jfloat *modelMatrixPtr = env->GetFloatArrayElements(modelMatrixArray, nullptr);
    cameraControlData->modelMatrix = glm::make_mat4(modelMatrixPtr);

    cameraControlData->rotateSpeed = env->GetFloatField(jCameraControlData, rotateSpeedFieldID);
    cameraControlData->isFirstFrame = env->GetBooleanField(jCameraControlData, isFirstFrameFieldID);
    cameraControlData->distanceOnScreen = env->GetFloatField(jCameraControlData, distanceOnScreenFieldID);

    jobject cameraObject = env->GetObjectField(jCameraControlData, cameraFieldID);
    jclass cameraClass = env->GetObjectClass(cameraObject);
    jfieldID positionFieldID = env->GetFieldID(cameraClass, "position", "[F");
    jfieldID frontFieldID = env->GetFieldID(cameraClass, "front", "[F");
    jfieldID upFieldID = env->GetFieldID(cameraClass, "up", "[F");
    jfieldID rightFieldID = env->GetFieldID(cameraClass, "right", "[F");
    jfieldID worldUpFieldID = env->GetFieldID(cameraClass, "worldUp", "[F");
    jfieldID yawFieldID = env->GetFieldID(cameraClass, "yaw", "F");
    jfieldID pitchFieldID = env->GetFieldID(cameraClass, "pitch", "F");
    jfieldID movementSpeedFieldID = env->GetFieldID(cameraClass, "movementSpeed", "F");
    jfieldID mouseSensitivityFieldID = env->GetFieldID(cameraClass, "mouseSensitivity", "F");
    jfieldID zoomFieldID = env->GetFieldID(cameraClass, "zoom", "F");
    jfieldID nearPlaneFieldID = env->GetFieldID(cameraClass, "nearPlane", "F");
    jfieldID farPlaneFieldID = env->GetFieldID(cameraClass, "farPlane", "F");

    jfloatArray positionArray = (jfloatArray) env->GetObjectField(cameraObject, positionFieldID);
    jsize positionSize = env->GetArrayLength(positionArray);
    assert(positionSize == 3);
    jfloat *positionPtr = env->GetFloatArrayElements(positionArray, nullptr);
    cameraControlData->camera.Position = glm::make_vec3(positionPtr);

    jfloatArray frontArray = (jfloatArray) env->GetObjectField(cameraObject, frontFieldID);
    jsize frontSize = env->GetArrayLength(frontArray);
    assert(frontSize == 3);
    jfloat *frontPtr = env->GetFloatArrayElements(frontArray, nullptr);
    cameraControlData->camera.Front = glm::make_vec3(frontPtr);

    jfloatArray rightArray = (jfloatArray) env->GetObjectField(cameraObject, rightFieldID);
    jsize rightSize = env->GetArrayLength(rightArray);
    assert(rightSize == 3);
    jfloat *rightPtr = env->GetFloatArrayElements(rightArray, nullptr);
    cameraControlData->camera.Right = glm::make_vec3(rightPtr);

    jfloatArray worldUpArray = (jfloatArray) env->GetObjectField(cameraObject, worldUpFieldID);
    jsize worldUpSize = env->GetArrayLength(worldUpArray);
    assert(worldUpSize == 3);
    jfloat *worldUpPtr = env->GetFloatArrayElements(worldUpArray, nullptr);
    cameraControlData->camera.WorldUp = glm::make_vec3(worldUpPtr);

    cameraControlData->camera.Yaw = env->GetFloatField(cameraObject, yawFieldID);
    cameraControlData->camera.Pitch = env->GetFloatField(cameraObject, pitchFieldID);
    cameraControlData->camera.MovementSpeed = env->GetFloatField(cameraObject, movementSpeedFieldID);
    cameraControlData->camera.MouseSensitivity = env->GetFloatField(cameraObject, mouseSensitivityFieldID);
    cameraControlData->camera.Zoom = env->GetFloatField(cameraObject, zoomFieldID);
    cameraControlData->camera.NearPlane = env->GetFloatField(cameraObject, nearPlaneFieldID);
    cameraControlData->camera.FarPlane = env->GetFloatField(cameraObject, farPlaneFieldID);
}

}