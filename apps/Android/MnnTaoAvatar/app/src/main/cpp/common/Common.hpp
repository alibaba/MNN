#pragma once
#include <cassert>
#include <cstring>
#include <vector>
#include <sstream>
#include <memory>
// glm
#include <glm/glm/glm.hpp>
#include <glm/glm/mat4x4.hpp>
#include <glm/glm/vec3.hpp>
#include <glm/glm/vec2.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/vector_relational.hpp>
// json
#include <nlohmann/json.hpp>
#include <android/native_window.h>
#include "common/Camera.hpp"

namespace TaoAvatar {

enum TouchEventAction : int32_t {
    kTouch_Down = 0,
    kTouch_Up,
    kTouch_Moved
};

struct TouchEvent {
    TouchEventAction touch_action;
    int32_t touch_id;
    int32_t touch_x;
    int32_t touch_y;
};

struct CameraControlData {
    float lastScale;
    float curScale;
    glm::mat4 rotateX;
    glm::mat4 rotateY;
    glm::mat4 scaleMatrix;
    glm::mat4 modelMatrix;
    float rotateSpeed;
    bool isFirstFrame;
    Camera camera;
    float distanceOnScreen;
    std::vector<TouchEvent> touchEvents;

public:
    CameraControlData() {
        curScale = 1.0f;
        lastScale = 1.0f;
        rotateX = glm::mat4(1);
        rotateY = glm::mat4(1);
        scaleMatrix = glm::mat4(1);
        modelMatrix = glm::mat4(1);
        rotateSpeed = 0.01f;
        isFirstFrame = true;
        camera = Camera(glm::vec3(0.0f, 1.0f, 2.2f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0, 0.0);
        distanceOnScreen = 0.0f;
        touchEvents.resize(2);
    }
};

struct AudioToBlendShapeData {
    // per frame's data
    std::vector<std::vector<float>> expr;
    std::vector<std::vector<float>> pose;
    std::vector<std::vector<float>> pose_z;
    std::vector<std::vector<float>> app_pose_z;
    std::vector<std::vector<float>> jaw_pose;
    std::vector<std::vector<float>> joints_transform;
    size_t frame_num = 0;
};

//struct AudioToBlendShapeData {
//    // per frame's data
//    std::vector<float> expr;
//    std::vector<float> pose;
//    std::vector<std::vector<float>> pose_z;
//    std::vector<std::vector<float>> app_pose_z;
//    std::vector<std::vector<float>> jaw_pose;
//    std::vector<std::vector<float>> joints_transform;
//    size_t frame_num = 0;
//};

struct NNRRuntimeData {
    CameraControlData camera;
};

struct BSIdleParamsData {
    std::string pos;
    std::vector<float> data;
};

struct BSIdleParams {
    std::map<std::string, BSIdleParamsData> data;
};

struct BlendShapeParams {
    std::map<std::string, std::vector<float>> data;
};

}