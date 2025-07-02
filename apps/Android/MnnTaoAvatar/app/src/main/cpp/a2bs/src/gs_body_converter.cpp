#include "gs_body_converter.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include "common/mh_log.hpp"
#include <array>
#include <cmath>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp> // For glm::pi()
#include <glm/gtc/quaternion.hpp> // For quaternions and angleAxis
#include <glm/gtx/quaternion.hpp> // For glm::toMat3, glm::angle, glm::axis
#include <glm/gtc/matrix_transform.hpp> // For glm::rotate (alternative approach, not used here directly for matrix combination)
#include <glm/gtc/type_ptr.hpp> // For printing vectors (optional)

glm::vec3 rotateBodyPoseGlm(const glm::vec3& original_pose_axis_angle,
                            float additional_angle_degrees, // Use float for standard GLM types
                               const glm::vec3& axis) {
    // Normalize the axis for the additional rotation
    float axis_norm = glm::length(axis);
    // Use a small epsilon for floating point comparison
    if (axis_norm < 1e-6f) {
        std::cerr << "Warning: Axis for additional rotation has near-zero length. Cannot normalize. Returning original pose." << std::endl;
        // Or consider throwing an exception
        return original_pose_axis_angle;
    }
    glm::vec3 axis_normalized = glm::normalize(axis);

    float additional_angle_radians = glm::radians(additional_angle_degrees);
    glm::quat additional_rotation_quat = glm::angleAxis(additional_angle_radians, axis_normalized);
    float original_angle_radians = glm::length(original_pose_axis_angle);
    glm::quat original_rotation_quat;
    if (original_angle_radians < 1e-6f) {
        original_rotation_quat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity quaternion
    } else {
        glm::vec3 original_axis = glm::normalize(original_pose_axis_angle); // Safe normalization
        original_rotation_quat = glm::angleAxis(original_angle_radians, original_axis);
    }
    glm::mat3 R_orig = glm::mat3_cast(original_rotation_quat);
    glm::mat3 R_add = glm::mat3_cast(additional_rotation_quat);
    glm::mat3 R_final = R_add * R_orig;
    glm::quat final_rotation_quat = glm::quat_cast(R_final);

    float final_angle_radians = glm::angle(final_rotation_quat);
    glm::vec3 final_axis = glm::axis(final_rotation_quat);
    if (final_angle_radians < 1e-6f || glm::any(glm::isnan(final_axis))) {
        return glm::vec3(0.0f, 0.0f, 0.0f); // Return zero vector for zero rotation
    }
    glm::vec3 final_pose_axis_angle = final_axis * final_angle_radians;

    return final_pose_axis_angle;
}



GSBodyConverter::GSBodyConverter(const std::string &local_resource_root,
                                 const std::string& cache_dir,
                                 const int num_exp): _num_exp(num_exp) {
    local_resource_root_ = local_resource_root;

    MNN::BackendConfig backend_config;
    executor_ = Executor::newExecutor(MNN_FORWARD_CPU, backend_config, 1);
    ExecutorScope scope(executor_);

    MNN::ScheduleConfig sConfig;
    BackendConfig cpuBackendConfig;
    cpuBackendConfig.precision = BackendConfig::Precision_High;
    cpuBackendConfig.memory = BackendConfig::Memory_Low;
    sConfig.numThread = 1;
    sConfig.backendConfig = &cpuBackendConfig;
    sConfig.type = MNN_FORWARD_CPU;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(sConfig),
                                                    Executor::RuntimeManager::destroy);
    if (rtmgr == nullptr) {
        MNN_ERROR("Empty RuntimeManger\n");
    }
    rtmgr->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
    rtmgr_ = rtmgr;

    Module::Config mdconfig;
    auto mnn_model_path = local_resource_root_ + "body_converter.mnn";
    float mem_0, mem_1;
    rtmgr->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_0);
    module.reset(Module::load(input_names, output_names, mnn_model_path.c_str(), rtmgr, &mdconfig));
    rtmgr->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_1);
    std::string error_message;
    std::string body_params_bin_path = local_resource_root_ + "/body_params.bin";
    ref_params_pool_ = ReadFramesFromBinary(body_params_bin_path, error_message);
}

std::vector<FullTypeOutput> GSBodyConverter::Process(const std::vector<float> &expression,
                                                     const std::vector<float> &jaw_pose,
                                                     const std::vector<BodyParamsInput>
                                                             &ref_params_pool, int ref_idx) {
    ExecutorScope scope(executor_);
    std::vector<VARP> inputs(10);
    int num_frames = expression.size() / _num_exp;
    int ref_num_frames = ref_params_pool.size();

    // 初始化输入tensor
    inputs[0] = MNN::Express::_Input({num_frames, _num_exp}, NCHW, halide_type_of<float>()); // expression
    inputs[1] = MNN::Express::_Input({num_frames,  3}, NCHW, halide_type_of<float>());  // jaw_pose
    inputs[2] = MNN::Express::_Input({num_frames, 63}, NCHW, halide_type_of<float>()); // body_pose
    inputs[3] = MNN::Express::_Input({num_frames, 3}, NCHW, halide_type_of<float>());  // leye_pose
    inputs[4] = MNN::Express::_Input({num_frames, 3}, NCHW, halide_type_of<float>());  // reye_pose
    inputs[5] = MNN::Express::_Input({num_frames, 45}, NCHW, halide_type_of<float>()); // left_hand_pose
    inputs[6] = MNN::Express::_Input({num_frames, 45}, NCHW, halide_type_of<float>()); // right_hand_pose
    inputs[7] = MNN::Express::_Input({num_frames, 3}, NCHW, halide_type_of<float>());  // Rh
    inputs[8] = MNN::Express::_Input({num_frames, 3}, NCHW, halide_type_of<float>());  // Th
    inputs[9] = MNN::Express::_Input({num_frames, 100}, NCHW, halide_type_of<float>());  // pose
    std::vector<float *> input_pointer = {
            inputs[0]->writeMap<float>(),   // expression
            inputs[1]->writeMap<float>(),   // jaw_pose
            inputs[2]->writeMap<float>(),   // body_pose
            inputs[3]->writeMap<float>(),   // leye_pose
            inputs[4]->writeMap<float>(),   // reye_pose
            inputs[5]->writeMap<float>(),   // left_hand_pose
            inputs[6]->writeMap<float>(),   // right_hand_pose
            inputs[7]->writeMap<float>(),   // Rh
            inputs[8]->writeMap<float>(),   // Th
            inputs[9]->writeMap<float>(),   // in_pose
    };

    // 设置输入数据
    for (int i = 0; i < num_frames * _num_exp; ++i)
    {
        input_pointer[0][i] = expression[i];
    }
    for (int i = 0; i < num_frames * 3; ++i)
    {
        input_pointer[1][i] = jaw_pose[i];
    }

    for (int i = 0; i < num_frames; ++i)
    {
        glm::vec3 original_pose =             {ref_params_pool[ref_idx].body_pose[33],
                                              ref_params_pool[ref_idx].body_pose[34],
                                              ref_params_pool[ref_idx].body_pose[35]};
        float additional_angle_degrees = 5.0f;
        glm::vec3 rotation_axis_3(1.0f, 0.0f, 0.0f);
        glm::vec3 transformed_pose;
        auto new_pose = rotateBodyPoseGlm(original_pose, additional_angle_degrees,
                                          rotation_axis_3);
        for (int j = 0; j < 63; ++j) {
            if (j >= 33 && j <= 35) {
                input_pointer[2][63*i + j] = new_pose[j - 33];
            } else {
                input_pointer[2][63*i + j] = ref_params_pool[ref_idx].body_pose[j];
            }
//            input_pointer[2][63*i + j] = ref_params_pool[ref_idx].body_pose[j];
        }
        for (int j = 0; j < 3; ++j)
        {
            input_pointer[3][3*i + j] = ref_params_pool[0].leye_pose[j];
            input_pointer[4][3*i + j] = ref_params_pool[0].reye_pose[j];
            input_pointer[7][3*i + j] = ref_params_pool[ref_idx].Rh[j];
            input_pointer[8][3*i + j] = ref_params_pool[ref_idx].Th[j];
        }

        for (int j = 0; j < 45; ++j) {
            input_pointer[5][45*i + j] = ref_params_pool[ref_idx].left_hand_pose[j];
            input_pointer[6][45*i + j] = ref_params_pool[ref_idx].right_hand_pose[j];
        }
        for (int j = 0; j < 100; j++) {
            input_pointer[9][100 * i + j] = ref_params_pool[ref_idx].pose[j];
        }
    }
    float mem_0, mem_1;
    rtmgr_->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_0);
    std::vector<MNN::Express::VARP> outputs = module->onForward(inputs);
    rtmgr_->getInfo(Interpreter::SessionInfoCode::MEMORY, &mem_1);
    MNN_PRINT("### gs_body_converter forward memory increase : %f \n", mem_1 - mem_0);

    // 获取输出
    std::vector<float> expr;
    std::vector<float> pose_z;
    std::vector<float> jaw_transform;

    std::vector<FullTypeOutput> res;
    if (outputs.empty()) {
        MH_ERROR("GSBodyConverter Empty outputs");
        return res;
    }

//    MH_DEBUG("===> frames num: %d\n", num_frames);
    // expr, Nx(_num_exp+1)
    auto output0_ptr = outputs[0]->readMap<float>();
    int output0_size = outputs[0]->getInfo()->size;
//    MH_DEBUG("===> expr size: %d\n", output0_size);
    // joints_transform, Nx55x4x4
    auto output1_ptr = outputs[1]->readMap<float>();
    int output1_size = outputs[1]->getInfo()->size;
//    MH_DEBUG("===> joints_transform size: %d\n", output1_size);
    // local_joints_transform, Nx55x4x4
    auto output2_ptr = outputs[2]->readMap<float>();
    int output2_size = outputs[2]->getInfo()->size;
//    MH_DEBUG("===> local_joints_transform size: %d\n", output2_size);
    // pose_z, Nx8
    auto output3_ptr = outputs[3]->readMap<float>();
    int output3_size = outputs[3]->getInfo()->size;

    // app_pose_z, Nx28
    auto output4_ptr = outputs[4]->readMap<float>();
    int output4_size = outputs[4]->getInfo()->size;
    auto output5_ptr = outputs[5]->readMap<float>();
    int output5_size = outputs[5]->getInfo()->size;


    for(int i = 0; i < num_frames; i++) {
        FullTypeOutput tmp;
        tmp.frame_id = i;
        tmp.expr.reserve(_num_exp + 1);
        tmp.joints_transform.reserve(880);
        tmp.local_joints_transform.reserve(880);
        tmp.pose_z.reserve(8);
        tmp.app_pose_z.reserve(28);
        tmp.pose.reserve(101);
        for(int j = 0; j < _num_exp + 1; j ++) {
            tmp.expr.push_back(output0_ptr[i * (_num_exp+1) + j]);
        }
        for(int j = 0; j < 880; j++) {
            tmp.joints_transform.push_back(output1_ptr[i*880 + j]);
            tmp.local_joints_transform.push_back(output2_ptr[i*880 + j]);
        }
        for(int j = 0; j < 8; j++){
            tmp.pose_z.push_back(output3_ptr[i*8 + j]);
        }
        for(int j = 0; j < 28; j++){
            tmp.app_pose_z.push_back(output4_ptr[i*28 + j]);
        }
        for (int j = 0; j < 101; j ++){
            tmp.pose.push_back(output5_ptr[i * 101 + j]);
        }
        res.push_back(tmp);
    }
    return res;
}
