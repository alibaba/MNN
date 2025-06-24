#pragma once

#include "NNRScene.h"
#include "common/file_utils.hpp"
#include <MNN/AutoTime.hpp>
#include "a2bs/gs_body_converter.hpp"
namespace TaoAvatar {

struct ChatStatus {
    std::string status;
    float start;
    float end;
    int slice_id;
};

class NnrAvatarRender {
public:
    ~NnrAvatarRender();
    bool InitNNR(ANativeWindow *window, std::string root_dir, std::string cache_dir);

    bool LoadNnrResourcesFromFile(const char *nnrRootPath, const char *computeSceneFileName, const char *renderSceneFileName,
                                  const char *skyboxSceneFileName, const char *deformParamFileName,
                                  const char* chatStatusFileName);

    NNRScene* LoadNNRSceneFromFile(const char *fileName);

    NNRScene* LoadNNRSceneFromBuffer(void* buffer, size_t length);

    static bool ReplaceNNRSceneFromFile(const char *fileName, struct NNRScene *scene, const char *key,
                                        const char *parent, size_t flag);

    static bool ReplaceNNRSceneFromBuffer(void* buffer, size_t length, struct NNRScene *scene, const char *key,
                                          const char *parent, size_t flag);

    void UpdateNNRScene(CameraControlData &cameraControl,
                        bool is_playing,
                        uint64_t playing_time,
                        uint64_t total_time,
                        bool is_buffering,
                        float smooth_to_idle_percent,
                        float smooth_to_talk_percent,
                        long forceFrameIndex);

    void SetDeformParameter();

    void SetSceneParameter(CameraControlData &cameraControl);

    void Render();

    void DestroyNNR();

    bool IsNNRReady() const { return target_initialized_ && model_initialized_; }

    int UpdateChatStatusJson(const std::string& jsonPath);

    void UpdatePlayPosition(bool is_playing,
                            uint64_t playing_time,
                            uint64_t total_time,
                            bool is_buffering);

    void Reset();

private:
    int GetCurrentStatusCycle(float position_time);
    static FLAMEOuput Smooth(FLAMEOuput& start, FLAMEOuput& end, float percent);
    NNRScene *scene_ = nullptr;
    NNRContext *context_ = nullptr;
    NNRTarget *render_target_ = nullptr;
    ANativeWindow *window_ = nullptr;
    std::string root_dir_;
    std::string cache_dir_;
    std::vector<ChatStatus> chat_status;
    std::vector<BSIdleParams> bs_idle_params_{};
    AudioToBlendShapeData bs_driver_params_;
    glm::mat4 view_matrix_cache_;
    bool sort_cache_inited_ = false;
    int width_ = 0;
    int height_ = 0;
    int mocap_fps_ = 20;
    bool target_initialized_ = false;
    bool model_initialized_ = false;
    void SetTargetCycle(int target_cycle_slice_id);
    ChatStatus target_cycle;
    bool render_started_ = false;
    std::chrono::steady_clock::time_point render_start_time_;
    GSBodyConverter* body_converter_{nullptr};
    AudioToBlendShapeData active_blend_shape_data_{};
    BlendShapeParams active_blend_shape_params_{};
    int move_direction_{1};
    size_t last_bs_driver_params_index_ = 0;
    size_t bs_driver_params_index_ = 0;
    int target_cycle_slice_id_{0};
    float target_cycle_start_mills{0};
    float target_cycle_end_mills{0};
    float last_position_time_{0};
    float current_position_time_{0};
    uint64_t last_audio_total_time{0};
    std::chrono::steady_clock::time_point last_update_time_;
    bool DEBUG_ALWAYS_IDLE = false;
    bool last_is_playing_ = false;
    FLAMEOuput smooth_begin_frame_;
    FLAMEOuput smooth_end_frame_;
    FLAMEOuput last_active_frame_;
    std::vector<float> last_render_expr_;
    std::vector<float> last_render_jaw_pose_;
    float smooth_to_idle_begun_{false};
    float smooth_to_talk_begun_{false};
};
} // namespace TaoAvatar