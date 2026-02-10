#include "nnr_avatar_render.hpp"
#include <iostream>
#include <fstream>
#include <set>
#include "common/mh_log.hpp"
#include "a2bs_service.hpp"
#include "a2bs/include/a2bs/gs_body_converter.hpp"
#include "mh_log.hpp"
#include "mh_config.hpp"

namespace TaoAvatar {
void GetWindowSize(ANativeWindow *window, int *width, int *height) {
    if (window != nullptr) {
        *width = ANativeWindow_getWidth(window);
        *height = ANativeWindow_getHeight(window);
    }
}

bool NnrAvatarRender::InitNNR(ANativeWindow *window, std::string root_dir, std::string cache_dir) {
    if (target_initialized_) {
        return true;
    }
    if (!window) {
        MH_ERROR("Invalid window.");
        return false;
    }
    root_dir_ = std::move(root_dir);
    cache_dir_ = std::move(cache_dir);
    window_ = window;
    int screenWidth = 0;
    int screenHeight = 0;
    GetWindowSize(window, &screenWidth, &screenHeight);
    width_ = screenHeight;
    height_ = screenHeight;
    context_ = NNRContextCreate(NNRRenderType_VULKAN);
#if SEPRATE_NNR_RESOURCES
    NNRContextSetResourceDir(context_, root_dir_.c_str());
#endif
    NNRContextInit(context_, nullptr);
    NNRTargetInfo targetInfo;
    NNRTargetInfoInit(&targetInfo, width_, height_);
    // Set viewport info
    targetInfo.viewport[0] = (screenWidth - screenHeight)/2.0f;
    targetInfo.viewport[1] = 0.0f;
    targetInfo.viewport[2] = width_;
    targetInfo.viewport[3] = height_;
    targetInfo.window = window_;
    NNRContextInitTarget(context_, &targetInfo);
    NNRContextSetComputeMode(context_, NNRComputeMode_ASYNC);
    render_target_ = NNRTargetCreate(NNRRenderType_VULKAN);
    target_initialized_ = true;
    return true;
}

bool NnrAvatarRender::LoadNnrResourcesFromFile(
                                          const char *nnrRootPath,
                                          const char *computeSceneFileName, const char *renderSceneFileName,
                                          const char *skyboxSceneFileName, const char *deformParamFileName,
                                          const char *chatStatusFileName) {
    auto start = std::chrono::steady_clock::now();
    auto scene = LoadNNRSceneFromFile(computeSceneFileName);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
    );
    MH_DEBUG("NNR Loaded scene in %lld ms", duration.count());
    if (scene == nullptr)
        return false;
    scene_ = scene;
    start = std::chrono::steady_clock::now();
    bool result = ReplaceNNRSceneFromFile(renderSceneFileName, scene_, "Render",
                                          nullptr, 0);
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
    );
    MH_DEBUG("NNR Replace Render in %lld ms", duration.count());
    if (!result) {
        return false;
    }
    start = std::chrono::steady_clock::now();
    result = ReplaceNNRSceneFromFile(skyboxSceneFileName, scene_, "skybox", nullptr, 0);
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
    );
    MH_DEBUG("NNR Replace skybox in %lld ms", duration.count());
    if (!result) {
        return false;
    }
    start = std::chrono::steady_clock::now();
    body_converter_ = new GSBodyConverter(
            nnrRootPath,
            cache_dir_
    );
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
    );
    MH_DEBUG("NNR Loaded body converter in %lld ms", duration.count());
    UpdateChatStatusJson(chatStatusFileName);
    model_initialized_ = true;
    MH_DEBUG("Loaded NNR resources chat_status size:  %zu bs_idle_params_ size: %zu"
             , chat_status.size(), bs_idle_params_.size());
    return true;
}

int NnrAvatarRender::UpdateChatStatusJson(const std::string& jsonPath) {
    std::ifstream file(jsonPath);
    if (!file.is_open()) {
        return 1;
    }
    std::string json_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    auto root = nlohmann::json::parse(json_content);
    for (const auto& item : root) {
        ChatStatus status;
        if (item.contains("status") && item["status"].is_string()) {
            status.status = item["status"];
        }
        if (item.contains("start") && item["start"].is_number()) {
            status.start = item["start"];
        }
        if (item.contains("end") && item["end"].is_number()) {
            status.end = item["end"];
        }
        status.slice_id = static_cast<int>(chat_status.size());
        chat_status.push_back(status);
    }
    if (!chat_status.empty()) {
        SetTargetCycle(0);
    }
    return 0;
}

void NnrAvatarRender::SetDeformParameter() {
    static const std::vector<std::string> validDeforms = {
            "expr",
            "pose",
            "pose_z",
            "app_pose_z",
            "joints_transform"
    };
    for (const auto& name: validDeforms) {
        if (!active_blend_shape_params_.data.empty()) {
            auto iter = active_blend_shape_params_.data.find(name);
            if (iter == active_blend_shape_params_.data.end()) {
                continue;
            }
            auto loc = NNRSceneGetComponentPosition(scene_, name.c_str());
            if (loc != 0 && !iter->second.empty()) {
                NNRSceneSetComponentData(scene_, loc, iter->second.data(),
                                         iter->second.size() * sizeof(float));
            }
        }
    }
}

void NnrAvatarRender::SetSceneParameter(CameraControlData &cameraControl) {
    glm::mat4 matrixProject, matrixView;
    matrixView = cameraControl.camera.GetViewMatrix() * cameraControl.modelMatrix;
    matrixProject = cameraControl.camera.GetProjectMatrix((float) width_ / (float) height_);

    // Set viewport number (can be 2 for VR mode)
    {
        auto viewportNumLoc = NNRSceneGetComponentPosition(scene_, "instance");
        if (viewportNumLoc != 0) {
            int viewportNum = 1;
            NNRSceneSetComponentData(scene_, viewportNumLoc, &viewportNum, 0);
        }
    }
    // Set focal uniform param
    {
        float fovy = cameraControl.camera.Zoom / 180.0 * glm::pi<float>();
        float aspectRatio = (float) (width_) / (float) (height_);
        float tan_fovy = std::tan(fovy / 2.0f);
        float tan_fovx = tan_fovy * aspectRatio;

        std::vector<float> focal = {(float) (width_) / 2.0f / tan_fovx,
                                    (float) (height_) / 2.0f / tan_fovy};
        auto focalLoc = NNRSceneGetComponentPosition(scene_, "focal");
        if (focalLoc != 0) {
            NNRSceneSetComponentData(scene_, focalLoc, focal.data(),
                                     focal.size() * sizeof(float));
        }
    }
    // Set model and view matrix
    {
        auto modelMatrixLoc = NNRSceneGetComponentPosition(scene_, "M");
        if (modelMatrixLoc != 0) {
            void *ptr = NNRSceneMapComponentData(scene_, modelMatrixLoc, NNRMapFlag_WRITE);
            if (cameraControl.isFirstFrame) {
                // Get the original model matrix from nnr
                glm::mat4 model = glm::make_mat4((float *) ptr);
                cameraControl.modelMatrix = model;
                cameraControl.isFirstFrame = false;
            }

            // ::memcpy(ptr, glm::value_ptr(cameraControl.modelMatrix), 16 * sizeof(float));
            NNRSceneUnmapComponentData(scene_, modelMatrixLoc, ptr, NNRMapFlag_WRITE);
        }
        auto viewMatrixLoc = NNRSceneGetComponentPosition(scene_, "V");
        if (viewMatrixLoc != 0) {
            if (modelMatrixLoc != 0) {
                NNRSceneSetComponentData(scene_, viewMatrixLoc, glm::value_ptr(matrixView),
                                         16 * sizeof(float));
            } else {
                // glm::mat4 matrixV = matrixView * cameraControl.modelMatrix * cameraControl.scaleMatrix;
                glm::mat4 matrixV = matrixView;
                NNRSceneSetComponentData(scene_, viewMatrixLoc, glm::value_ptr(matrixV),
                                         16 * sizeof(float));
            }
        }
    }
    // Set image size uniform param (viewport)
    {
        auto imgSizeLoc = NNRSceneGetComponentPosition(scene_, "img_size");
        if (imgSizeLoc != 0) {
            std::vector<float> imgSize = {(float) width_, (float) height_};
            NNRSceneSetComponentData(scene_, imgSizeLoc, imgSize.data(), imgSize.size() * sizeof(float));
        }
    }
    // Set project matrix
    {
        auto projMatrixLoc = NNRSceneGetComponentPosition(scene_, "P");
        if (projMatrixLoc != 0) {
            NNRSceneSetComponentData(scene_, projMatrixLoc, glm::value_ptr(matrixProject), 16 * sizeof(float));
        }
    }
    // Set ibl luminance uniform param
    {
        auto iblLuminanceLoc = NNRSceneGetComponentPosition(scene_, "u_iblLuminance");
        if (iblLuminanceLoc != 0) {
            float luminance = 1.0f;
            NNRSceneSetComponentData(scene_, iblLuminanceLoc, &luminance, sizeof(float));
        }
    }
    // Set sort interval uniform param
    {
        auto sortLoc = NNRSceneGetComponentPosition(scene_, "sort_interval");
        if (sortLoc != 0) {
            float time = 1.0f;
            NNRSceneSetComponentData(scene_, sortLoc, &time, sizeof(float));
        }
    }
    // Set gamma uniform param
    {
        auto gammaLoc = NNRSceneGetComponentPosition(scene_, "gamma");
        float gamma = 1.0f;
        if (gammaLoc != 0) {
            NNRSceneSetComponentData(scene_, gammaLoc, &gamma, 0);
        }
    }
    // Set sort switch
    {
        auto sortOnOffLoc = NNRSceneGetComponentPosition(scene_, "sort_on_off");
        if (sortOnOffLoc != 0) {
            if (!sort_cache_inited_) {
                sort_cache_inited_ = true;
                view_matrix_cache_ = matrixView;
            } else {

                auto src = glm::value_ptr(matrixView);
                auto dst = glm::value_ptr(view_matrix_cache_);
                static const int indexes[] = {
                        2, 6, 10, 14,
                        3, 7, 11, 15
                };
                float maxDiff = 0.0f;
                for (int index : indexes) {
                    auto diff = src[index] - dst[index];
                    diff = diff * diff;
                    if (diff > maxDiff) {
                        maxDiff = diff;
                    }
                }
                if (maxDiff > 0.1f) {
                    view_matrix_cache_ = matrixView;
                }
            }
            // NNRSceneSetComponentData(mScene, sortOnOffLoc, &sort_on_off, 0);
        }
    }
    // Set skybox be visible or not
    {
        float skyboxVisible = 1.0f; // skybox is visible
        auto skyboxLoc = NNRSceneGetComponentPosition(scene_, "skybox_visible");
        if (skyboxLoc != 0) {
            NNRSceneSetComponentData(scene_, skyboxLoc, &skyboxVisible, sizeof(float));
        }
    }
    {
        std::vector<float> background= {.78f, .8f, .8f, 0.0f};
        auto backgroundPosition = NNRSceneGetComponentPosition(scene_, "background");
        if (backgroundPosition != 0) {
            NNRSceneSetComponentData(scene_, backgroundPosition, background.data(), sizeof(float) * 4);
        }
    }
}

int NnrAvatarRender::GetCurrentStatusCycle(float position_time) {
    for (int i = 0; i < chat_status.size(); ++i) {
        if (chat_status[i].start * 1000 <= position_time && chat_status[i].end * 1000 >= position_time) {
            return i;
        }
    }
    return 0;
}

void NnrAvatarRender::UpdatePlayPosition(bool is_playing,
                                         uint64_t playing_time,
                                         uint64_t total_time,
                                         bool is_buffering) {
    auto now = std::chrono::steady_clock::now();
    if (!render_started_) {
        render_started_ = true;
        render_start_time_ = now;
        last_update_time_ = now;
    }
    float unplayed_time = total_time - playing_time;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_time_).count();
    int current_status_cycle_index = GetCurrentStatusCycle(last_position_time_);
    auto& current_cycle = chat_status[current_status_cycle_index];
    MH_LOGV("NnrAvatarRender NNRUpdate Init current status: %s-%d, target_status: %s-%d", current_cycle.status.c_str(), current_cycle.slice_id, target_cycle.status.c_str(), target_cycle.slice_id);
    if (is_playing && unplayed_time > 500 && !(target_cycle.status == "talk") && !DEBUG_ALWAYS_IDLE) {
        if (current_cycle.status == "talk") {
            SetTargetCycle(current_cycle.slice_id);
        } else if (current_cycle.slice_id + 1 >= chat_status.size()) {
            SetTargetCycle(current_cycle.slice_id - 1);
            move_direction_ = 0;
            MH_LOGV("NnrAvatarRender hasNewData Move to play  target_slice_id %d move_direction_: %d  ", target_cycle_slice_id_, move_direction_);
        } else if (target_cycle.slice_id - 1 < 0) {
            SetTargetCycle(target_cycle.slice_id + 1);
            move_direction_ = 1;
            MH_LOGV("NnrAvatarRender hasNewData Move to play  target_slice_id %d move_direction_: %d ", target_cycle_slice_id_, move_direction_);
        } else {
            ChatStatus& next_chat = chat_status[current_cycle.slice_id + 1];
            ChatStatus& pre_chat = chat_status[current_cycle.slice_id - 1];
            float next_dis = next_chat.start - last_position_time_;
            float pre_dis = last_position_time_ - pre_chat.end;
            auto target_slice_id = 0;
            if (next_dis < pre_dis) {
                target_slice_id = target_cycle.slice_id + 1;
                move_direction_ = 1;
            } else {
                target_slice_id = target_cycle.slice_id - 1;
                move_direction_ = 0;
            }
            SetTargetCycle(target_slice_id);
            MH_LOGV("NnrAvatarRender hasNewData Move to play  target_slice_id %d  move_direction_: %d  ", target_cycle_slice_id_, move_direction_);
        }
    }
    last_is_playing_=  is_playing;
    MH_LOGV("NnrAvatarRender NNRUpdate current status: %s-%d, target_status: %s-%d", current_cycle.status.c_str(), current_cycle.slice_id, target_cycle.status.c_str(), target_cycle.slice_id);
    if (target_cycle.status == "idle") {
        if (move_direction_ == 1) {
            current_position_time_ = last_position_time_ += elapsed_ms;
            if (current_position_time_ > target_cycle_end_mills) {
                move_direction_ = 0;
                MH_LOGV("NnrAvatarRender idle pingpong Move direction %d current_position_time_: %f target_cycle_end_mills:%f ", move_direction_, current_position_time_, target_cycle_end_mills);
                current_position_time_ = target_cycle_end_mills - (current_position_time_ - target_cycle_end_mills);
            }
        } else if (move_direction_ == 0) {
            current_position_time_ = last_position_time_ -= elapsed_ms;
            if (current_position_time_ < target_cycle_start_mills) {
                move_direction_ = 1;
                MH_LOGV("NnrAvatarRender idle pingpong Move direction %d current_position_time_: %f target_cycle_start_mills:%f ", move_direction_, current_position_time_, target_cycle_start_mills);
                current_position_time_ = target_cycle_start_mills + (target_cycle_start_mills - current_position_time_);
            }
        }
    } else {
        if (current_cycle.status != "idle") {
            ChatStatus next_idle = chat_status[target_cycle.slice_id + 1];
            ChatStatus pre_idle = chat_status[target_cycle.slice_id - 1];
            ChatStatus target_idle;
            float next_dis = next_idle.start * 1000.0f - last_position_time_;
            float pre_dis = last_position_time_ - pre_idle.end * 1000.0f;
            float min_dis = std::min(pre_dis, next_dis);
            MH_LOGV("NnrAvatarRender NNRUpdate unplayed_time: %f next_dis: %f pre_dis: %f next_start :%f pre end: %f"
            , unplayed_time, next_dis, pre_dis,
                     next_idle.start * 1000.0f, pre_idle.end * 1000.0f);
            if (min_dis >= unplayed_time) {
                int target_cycle_slice_id;
                target_cycle_slice_id = (next_dis < pre_dis) ? target_cycle.slice_id + 1 : target_cycle.slice_id - 1;
                move_direction_ = (next_dis < pre_dis) ? 1 : 0;
                MH_LOGV("NnrAvatarRender unplayed_time Move direction %d unplayed_time: %f min_dis:%f ", move_direction_, unplayed_time, min_dis);
                SetTargetCycle(target_cycle_slice_id);
            }
        }
        if (move_direction_ == 1) {
            current_position_time_ = last_position_time_ += elapsed_ms;
            if (current_position_time_ > target_cycle_end_mills) {
                move_direction_ = 0;
                MH_LOGV("NnrAvatarRender chat pingpong Move direction %d current_position_time_: %f target_cycle_end_mills:%f ", move_direction_, current_position_time_, target_cycle_end_mills);
                current_position_time_ = target_cycle_end_mills - (current_position_time_ - target_cycle_end_mills);
            }
        } else if (move_direction_ == 0) {
            current_position_time_ = last_position_time_ -= elapsed_ms;
            if (current_position_time_ < target_cycle_start_mills) {
                move_direction_ = 1;
                MH_LOGV("NnrAvatarRender chat pingpong Move direction %d current_position_time_: %f target_cycle_start_mills:%f ", move_direction_, current_position_time_, target_cycle_start_mills);
                current_position_time_ = target_cycle_start_mills + (target_cycle_start_mills - current_position_time_);
            }
        }
    }
    last_position_time_ = current_position_time_;
    last_update_time_ = now;
}

void NnrAvatarRender::UpdateNNRScene(CameraControlData &cameraControl,
                                     bool is_playing,
                                     uint64_t playing_time,
                                     uint64_t total_time,
                                     bool is_buffering,
                                     float smooth_to_idle_percent,
                                     float smooth_to_talk_percent,
                                     long force_frame_index) {
    is_playing = is_playing && playing_time > 0;
    if (last_audio_total_time != total_time) {
        MH_LOGV("NnrAvatarRender TotalAudioTimeUpdate is_playing: %d playing_time: %lu total_time: %lu ", is_playing, playing_time, total_time);
        last_audio_total_time = total_time;
    }
    SetSceneParameter(cameraControl);
    UpdatePlayPosition(is_playing, playing_time, total_time, is_buffering);
    auto idle_frame_index = (size_t)std::floor(current_position_time_ / 1000.0f * mocap_fps_);
#if DEBUG_KEEP_BODY_STATIC
    idle_frame_index = 0;
#endif
    bs_driver_params_index_ =  is_playing ?  (size_t)std::floor(playing_time / 1000.0f * mocap_fps_) : 0;
    if (force_frame_index >= 0) {
        bs_driver_params_index_ = (size_t)force_frame_index;
    }
    auto& input_frames = body_converter_->GetBodyParamsInput();
    auto& idle_jaw_pose = input_frames[idle_frame_index].jaw_pose;
    auto& idle_expression = input_frames[idle_frame_index].expression;
    MH_LOGV("NnrAvatarRender NNRUpdate current_position_time_: %f idle_frame_index :%zu total_count: %lu target_begin: %f target_end: %f "
             "current audio time:%lu ,total audio time %lu bs_driver_params_index_: %zu total_frame_count: %zu is_buffering:%d ",
             current_position_time_, idle_frame_index, input_frames.size(), target_cycle_start_mills, target_cycle_end_mills,
             playing_time, total_time, bs_driver_params_index_, A2BSService::GetActiveInstance()->GetTotalFrameNum(), (int)is_buffering);
    if (smooth_to_idle_percent >= 0.0f || smooth_to_talk_percent >= 0.0f) {
        MH_LOGV("NnrAvatarRender NNRUpdate smooth_to_idle_percent: %f smooth_to_talk_percent: %f", smooth_to_idle_percent, smooth_to_talk_percent);
    }
    int segment_index = 0;
    int sub_index = 0;
    const FLAMEOuput active_frame = A2BSService::GetActiveInstance()->GetActiveFrame(
            bs_driver_params_index_, segment_index, sub_index);
    bool use_active_frame = force_frame_index >= 0 || (is_playing && !is_buffering && !active_frame.IsEmpty());
    if (use_active_frame) {
        last_active_frame_ = active_frame;
    }
    auto expression_param = use_active_frame  ? active_frame.expr : idle_expression;
    auto jaw_pose_param = use_active_frame ? active_frame.jaw_pose : idle_jaw_pose;
    if (smooth_to_talk_percent >= 0.0f) {
        if (smooth_begin_frame_.IsEmpty() || smooth_end_frame_.IsEmpty() || !smooth_to_talk_begun_) {
            if (last_render_jaw_pose_.empty() || last_render_expr_.empty()) {
                last_render_jaw_pose_ = idle_jaw_pose;
                last_render_expr_ = idle_expression;
            }
            smooth_begin_frame_ = {-1, last_render_expr_, last_render_jaw_pose_};
            int segment_index_next = 0;
            int sub_index_next = 0;
            const FLAMEOuput next_active_frame = A2BSService::GetActiveInstance()->GetActiveFrame(
                    bs_driver_params_index_ + 1, segment_index_next, sub_index_next);
            if (next_active_frame.IsEmpty()) {
                MH_ERROR("NnrAvatarRender NNRUpdate next_active_frame empty driver index: %zu", bs_driver_params_index_ + 1);
            }
            smooth_end_frame_ = next_active_frame;
        }
        const FLAMEOuput smooth_result = Smooth(smooth_begin_frame_, smooth_end_frame_,
                                                 smooth_to_idle_percent);
        expression_param = smooth_result.expr;
        jaw_pose_param = smooth_result.jaw_pose;
        smooth_to_talk_begun_ = true;
        smooth_to_idle_begun_ = false;
    } else if (smooth_to_idle_percent >= 0.0f) {
        if (smooth_begin_frame_.IsEmpty() || smooth_end_frame_.IsEmpty() || !smooth_to_idle_begun_) {

            smooth_begin_frame_ = last_active_frame_;
            smooth_end_frame_ = {-1, idle_expression, idle_jaw_pose};
        }
        const FLAMEOuput smooth_result = Smooth(smooth_begin_frame_, smooth_end_frame_,
                                                 smooth_to_idle_percent);
        expression_param = smooth_result.expr;
        jaw_pose_param = smooth_result.jaw_pose;
        smooth_to_talk_begun_ = false;
        smooth_to_idle_begun_ = true;
    } else {
        smooth_to_idle_begun_ = false;
        smooth_to_talk_begun_ = false;
        smooth_begin_frame_.Reset();
        smooth_end_frame_.Reset();
    }
    last_render_jaw_pose_ = jaw_pose_param;
    last_render_expr_ = expression_param;

    auto output = body_converter_->Process(
            expression_param,
            jaw_pose_param,
            input_frames,
             idle_frame_index);

#if DEBUG_TALK_PARAMS
    if (use_active_frame) {
        static std::set<size_t> written_indices;
        if (written_indices.find(bs_driver_params_index_) == written_indices.end()) {
            written_indices.insert(bs_driver_params_index_);

            nlohmann::json json_data;
            json_data["bs_driver_params_index"] = bs_driver_params_index_;
            json_data["expr"] = active_frame.expr;
            json_data["jaw_pose"] = active_frame.jaw_pose;
            json_data["segment_index"] = segment_index;
            json_data["sub_index"] = sub_index;

            auto output = body_converter_->Process(
                    expression_param,
                    jaw_pose_param,
                    input_frames,
                    DEBUG_KEEP_BODY_STATIC ? 0 : idle_frame_index);

            nlohmann::json output_json;
            for (const auto &result : output) {
                nlohmann::json result_json;
                result_json["expr"] = result.expr;
                result_json["pose_z"] = result.pose_z;
                result_json["pose"] = result.pose;
                result_json["app_pose_z"] = result.app_pose_z;
                result_json["joints_transform"] = result.joints_transform;
                output_json.push_back(result_json);
            }
            json_data["output"] = output_json;

            std::string filename = this->root_dir_ + "/nnr_frame_data.json";
            std::ofstream file(filename, std::ios::app);
            if (file.is_open()) {
                file << json_data.dump(4);
                file.close();
                MH_DEBUG("NnrAvatarRender NNRUpdate wrote frame data to %s", filename.c_str());
            } else {
                MH_ERROR("NnrAvatarRender NNRUpdate failed to open file %s for writing", filename.c_str());
            }
        }
    }
#endif
    if (output.empty()) {
        MH_ERROR("NnrAvatarRender NNRUpdate output empty");
        return;
    }
    active_blend_shape_data_ = {};
    AudioToBlendShapeData result_data{};
    result_data.frame_num = output.size();
    for (const auto &result : output) {
        result_data.expr.push_back(result.expr);
        result_data.pose_z.push_back(result.pose_z);
        result_data.pose.push_back(result.pose);
        result_data.app_pose_z.push_back(result.app_pose_z);
        std::vector<float> transformed_joints_transform;
        size_t num_transforms =
                result.joints_transform.size() / 16;// Calculate the number of transforms
        const float *transform = result.joints_transform.data();
        for (size_t k = 0; k < num_transforms; ++k) {
            const float *trans_n = transform + k * 16; // Get the k-th transform
            glm::mat4 temp = glm::make_mat4(trans_n); // Create a mat4 from the transform
            glm::mat4 cv2gl = glm::rotate(temp, glm::radians(180.0f),
                                          glm::vec3(1.0f, 0.0f, 0.0f)); // Apply rotation
            float *t = glm::value_ptr(cv2gl); // Flatten the matrix to a float array
            for (int m = 0; m < 16; ++m) {
                transformed_joints_transform.emplace_back(t[m]);
            }
        }
        result_data.joints_transform.push_back(transformed_joints_transform);
    }
    if (result_data.frame_num > 0) {
        active_blend_shape_params_ = {};
        active_blend_shape_params_.data["expr"] = result_data.expr[0];
        active_blend_shape_params_.data["pose_z"] = result_data.pose_z[0];
        active_blend_shape_params_.data["pose"] = result_data.pose[0];
        active_blend_shape_params_.data["app_pose_z"] = result_data.app_pose_z[0];
        active_blend_shape_params_.data["joints_transform"] = result_data.joints_transform[0];
    }
    active_blend_shape_data_ = result_data;
    last_bs_driver_params_index_ = bs_driver_params_index_;
    SetDeformParameter();
}

void NnrAvatarRender::Render() {
    auto start = std::chrono::steady_clock::now();
    MH_LOGV("NnrAvatarRender Render Begin");
    if (scene_ != nullptr && render_target_ != nullptr) {
        NNRSceneRender(scene_, render_target_);
    } else {
        MH_ERROR("Failed to Render scene.");
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start
    );
    MH_LOGV("NnrAvatarRender Render time: %lld ms", duration.count());
}

NNRScene *NnrAvatarRender::LoadNNRSceneFromFile(const char *fileName) {
    if (fileName == nullptr) {
        MH_ERROR("Failed to load scene: File name is empty!");
        return nullptr;
    }
    auto sceneBuffer = file_utils::LoadFileToBuffer(fileName);
    return LoadNNRSceneFromBuffer((void *) sceneBuffer.data(), sceneBuffer.size());
}

NNRScene *NnrAvatarRender::LoadNNRSceneFromBuffer(void* buffer, size_t length) {
    if (buffer == nullptr || length == 0) {
        MH_ERROR("Failed to load scene: File buffer is empty!");
        return nullptr;
    }

    NNRScene *scene = NNRSceneCreateFromBuffer(buffer, length, context_);
    if (scene != nullptr) {
        return scene;
    } else {
        MH_ERROR("Failed to load scene.");
        return nullptr;
    }
}

bool NnrAvatarRender::ReplaceNNRSceneFromFile(const char *fileName, struct NNRScene *scene, const char *key,
                                              const char *parent, size_t flag) {
    if (fileName == nullptr) {
        MH_ERROR("Failed to replace scene: File name is empty!");
        return false;
    }

    if (scene == nullptr) {
        MH_ERROR("Scene pointer is nullptr!");
        return false;
    }
    auto sceneBuffer = file_utils::LoadFileToBuffer(fileName);
    return ReplaceNNRSceneFromBuffer((void *) sceneBuffer.data(), sceneBuffer.size(), scene, key,
                                     parent, flag);
}

bool NnrAvatarRender::ReplaceNNRSceneFromBuffer(void* buffer, size_t length, struct NNRScene *scene, const char *key,
                                                const char *parent, size_t flag) {
    if (buffer == nullptr || length == 0) {
        MH_ERROR("Failed to replace scene: File buffer is empty!");
        return false;
    }

    if (scene == nullptr) {
        MH_ERROR("Scene pointer is nullptr!");
        return false;
    }

    int ret = NNRSceneReplaceFromBuffer(buffer, length, scene, key,
                                        parent, flag);
    if (ret < 0) {
        return false;
    }
    return true;
}

    void NnrAvatarRender::DestroyNNR() {
    if (scene_) {
        NNRSceneDestroy(scene_);
        scene_ = nullptr;
    }
    if (context_) {
        NNRContextDestroy(context_);
        context_ = nullptr;
    }
    if (render_target_) {
        NNRTargetDestroy(render_target_);
        render_target_ = nullptr;
    }
    target_initialized_ = false;
    model_initialized_ = false;
    sort_cache_inited_ = false;
}

void NnrAvatarRender::SetTargetCycle(int target_cycle_slice_id) {
    if (target_cycle_slice_id >= 0 && target_cycle_slice_id < chat_status.size()) {
        target_cycle = chat_status[target_cycle_slice_id];
        target_cycle_start_mills = target_cycle.start * 1000.0f;
        target_cycle_end_mills = target_cycle.end * 1000.0f;
        target_cycle_slice_id_ = target_cycle_slice_id;
        MH_LOGV("NnrAvatarRender target_cycle_slice_id: %d", target_cycle_slice_id);
    }
}

NnrAvatarRender::~NnrAvatarRender() {
    if (body_converter_ != nullptr) {
        delete body_converter_;
        body_converter_ = nullptr;
    }
}

FLAMEOuput NnrAvatarRender::Smooth(FLAMEOuput& start, FLAMEOuput& end, float percent) {
    FLAMEOuput result;
    percent = std::clamp(percent, 0.0f, 1.0f);
    result.expr.resize(start.expr.size());
    result.jaw_pose.resize(start.jaw_pose.size());
    for (size_t i = 0; i < start.expr.size() && i < end.expr.size(); ++i) {
        result.expr[i] = start.expr[i] + (end.expr[i] - start.expr[i]) * percent;
    }
    for (size_t i = 0; i < start.jaw_pose.size() && i < end.jaw_pose.size(); ++i) {
        result.jaw_pose[i] = start.jaw_pose[i] + (end.jaw_pose[i] - start.jaw_pose[i]) * percent;
    }
    return result;
}

void NnrAvatarRender::Reset() {
    render_started_ = false;
    last_audio_total_time = 0;
    current_position_time_ = 0;
    last_position_time_ = 0;
    target_cycle_end_mills = 0;
    target_cycle_start_mills = 0;
    target_cycle_slice_id_ = 0;
    move_direction_ = 1;
    bs_driver_params_index_ = 0;
    last_is_playing_ = false;
    SetTargetCycle(0);
}


} // namespace TaoAvatar