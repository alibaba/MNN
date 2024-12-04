//
//  MatMulSpeed.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Optimizer.hpp>
#include "MNN_generated.h"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

#include "llm/llm.hpp"
#include "evaluation/dataset.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <initializer_list>

using namespace MNN::Express;
using namespace MNN::Transformer;

static void trace_prepare(Llm* llm) {
    MNN_PRINT("Prepare for resize opt Begin\n");
    llm->trace(true);
    std::ostringstream cacheOs;
    llm->generate(std::initializer_list<int>{200, 200}, &cacheOs, "");
    MNN_PRINT("Prepare for resize opt End\n");
    llm->trace(false);
    llm->reset();
}

static void fillFloat(float* dst, int h, int w, float offset = 0.0f) {
    for (int y = 0; y < h; ++y) {
        auto dstY = dst + w * y;
        for (int x = 0; x < w; ++x) {
            dstY[x] = ((float)x * 0.1f + (float)y + offset) / 10000.0f;
        }
    }
}

static MNNForwardType backend_type_convert(const std::string& type_str) {
    if (type_str == "cpu") return MNN_FORWARD_CPU;
    if (type_str == "metal") return MNN_FORWARD_METAL;
    if (type_str == "cuda") return MNN_FORWARD_CUDA;
    if (type_str == "opencl") return MNN_FORWARD_OPENCL;
    if (type_str == "opengl") return MNN_FORWARD_OPENGL;
    if (type_str == "vulkan") return MNN_FORWARD_VULKAN;
    if (type_str == "npu") return MNN_FORWARD_NN;
    return MNN_FORWARD_AUTO;
}

float profileMatMul(int e, int h, int l) {
    // Test MatMul
    // prepare MatMul config
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    op->type                = MNN::OpType_MatMul;
    op->main.type           = MNN::OpParameter_MatMul;
    op->main.value          = new MNN::MatMulT;
    auto matmulParam        = op->main.AsMatMul();
    matmulParam->transposeA = false;
    matmulParam->transposeB = false;

    // prepare input and output
    auto x0 = _Input({}, NHWC, halide_type_of<float>());
    auto x1 = _Input({}, NHWC, halide_type_of<float>());
    x0->resize({e, l});
    x1->resize({l, h});
    auto y = Variable::create(Expr::create(op.get(), {x0, x1}));
    Variable::prepareCompute({y});
    fillFloat(x0->writeMap<float>(), e, l);
    fillFloat(x1->writeMap<float>(), l, h);

    // Test for 5 times
    const auto time = 5;
    MNN::Timer _t;
    for (int t = 0; t < time; ++t) {
        x0->writeMap<float>();
        x1->writeMap<float>();
        y->readMap<float>();
    }
    float timeCost = _t.durationInUs() / 1000.0f / (float)time;
    float flops = (float)e * (float)l * (float)h / timeCost / 1000.0f / 1000.0f;
    MNN_PRINT("[%d, %d, %d], Avg time: %f ms , flops: %f G\n", e, l, h, timeCost, flops);
    return timeCost;
}

float profileLLM(Llm* llm, int prefill_len) {
    llm->trace(true);
    std::vector<int> test_prompt(prefill_len, 200);
    MNN::Timer _t;
    llm->forward(test_prompt, true);
    float timeCost = _t.durationInUs() / 1000.0f;
    llm->trace(false);
    llm->reset();   
    MNN_PRINT("[%d], LLM Prefill time: %f ms\n", prefill_len, timeCost);
    return timeCost;
}

std::shared_ptr<Executor> init_tune(MNNForwardType type, int thread) {
    int precision = (int)MNN::BackendConfig::Precision_Low;
    int memory = (int)MNN::BackendConfig::Memory_Low;
    const char* flag = "";
    MNN::BackendConfig config;
    config.precision = (MNN::BackendConfig::PrecisionMode)precision;
    config.memory = (MNN::BackendConfig::MemoryMode)memory;
    auto exe = MNN::Express::Executor::newExecutor(type, config, thread);
    if (exe == nullptr) {
        MNN_PRINT("Can't create executor with type:%d, exit!\n", type);
    }
    return exe;
}

void tune(Llm* llm) {
// void tune() {
    // problems for direct tuning: 1. low frequency, 2. don't know if tempcache need to be stored for opencl.
    // test: setCache for OpenCL
    float inf = 10000.f;
    int hidden_size = 1536; 
    int intermediate_size = hidden_size*3;
    
    std::vector<int> prefill_len_list = {100, 200, 400, 600, 800, 1000, 1500};
    // std::vector<int> prefill_len_list = {100};
    
    float qkv_sim_cpu = inf;
    float up_proj_sim_cpu = inf;
    float down_proj_sim_cpu = inf; 
    float prefill_llm_sim_cpu = inf; 
    auto cpu_exe = init_tune(backend_type_convert("cpu"), 4);
    for (auto prefill_len : prefill_len_list){
        MNN::Express::ExecutorScope scope(cpu_exe);
        // qkv_sim_cpu = profileMatMul(prefill_len, hidden_size, hidden_size);
        // up_proj_sim_cpu = profileMatMul(prefill_len, hidden_size, intermediate_size);
        // down_proj_sim_cpu = profileMatMul(prefill_len, intermediate_size, hidden_size);
        // prefill_llm_sim_cpu = profileLLM(llm, prefill_len);
    }

    float qkv_sim_opencl = inf;
    float up_proj_sim_opencl = inf;
    float down_proj_sim_opencl = inf;
    float prefill_llm_sim_opencl = inf;
    auto opencl_exe = init_tune(backend_type_convert("opencl"), 68);
    if (opencl_exe != nullptr) {
        for (auto prefill_len : prefill_len_list) {
            MNN::Express::ExecutorScope scope(opencl_exe);
            // qkv_sim_opencl = profileMatMul(prefill_len, hidden_size, hidden_size);
            // up_proj_sim_opencl = profileMatMul(prefill_len, hidden_size, intermediate_size);
            // down_proj_sim_opencl = profileMatMul(prefill_len, intermediate_size, hidden_size);
            prefill_llm_sim_opencl = profileLLM(llm, prefill_len);
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json [performance.txt]" << std::endl;
        return 0;
    }
    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    {
        AUTOTIME;
        llm->load();
    }
    {
        AUTOTIME;
        trace_prepare(llm.get());
    }
    tune(llm.get());
    // tune();
    // std::string prompt_file = argv[2];
    // std::unique_ptr<std::ofstream> perfOS(nullptr);
    // if (argc == 4) { perfOS.reset(new std::ofstream(argv[3])); }
    return 0;
}