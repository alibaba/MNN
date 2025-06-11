#pragma once

#include "MNN/AutoTime.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/ExecutorScope.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/expr/Module.hpp"
#include "a2bs_utils.hpp"
#include "common/mh_log.hpp"
#include "common/mh_config.hpp"

using namespace MNN;
using namespace MNN::Express;

class GSBodyConverter {
public:
    explicit GSBodyConverter(const std::string &local_resource_root, const std::string& cache_dir, int num_exp = 50);

    std::vector<FullTypeOutput> Process(const std::vector<float> &expression,
                                        const std::vector<float> &jaw_pose,
                                        const std::vector<BodyParamsInput> &ref_params,
                                        int ref_idx);
    std::vector<BodyParamsInput>& GetBodyParamsInput() {
        return ref_params_pool_;
    };

private:
    std::string local_resource_root_;
    std::vector<BodyParamsInput> ref_params_pool_;
    std::unique_ptr<Module> module;
    std::shared_ptr<Executor> executor_;

    const std::vector<std::string> input_names{"expression", "jaw_pose", "body_pose", "leye_pose",
                                               "reye_pose", "left_hand_pose", "right_hand_pose",
                                               "Rh", "Th", "in_pose"};
    const std::vector<std::string> output_names{"expr", "joints_transform",
                                                "local_joints_transform", "pose_z", "app_pose_z", "out_pose"};
    std::shared_ptr<Executor::RuntimeManager> rtmgr_;
    int _num_exp;
};
