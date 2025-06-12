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
#include "mnn_audio2bs_logger.hpp"

using namespace MNN;
using namespace MNN::Express;

class AudioToFlameBlendShape
{
public:
    AudioToFlameBlendShape(const std::string &local_resource_root, const std::string &mnn_mmap_dir, const bool do_verts2flame, const int ori_fps, const int out_fps, const int num_exp=50);
    std::vector<std::vector<float>> Process(const std::vector<float> &audio,
                                            int sample_rate);

private:
    std::string local_resource_root_;

    std::unique_ptr<Module> audio2verts_module; // module
    std::shared_ptr<Executor> executor_;

    const std::vector<std::string> audio2verts_input_names{"audio"};
    const std::vector<std::string> audio2verts_output_names{"verts"};

    std::unique_ptr<Module> verts2flame_module; // module
    const std::vector<std::string> verts2flame_input_names{"verts"};
    const std::vector<std::string> verts2flame_output_names{"coeff_out"};

    std::shared_ptr<Executor::RuntimeManager> rtmgr_;
    std::vector<int> eye_verts_index_;
    const bool do_verts2flame_;
    const int ori_fps_, out_fps_;
    int _num_exp;
};