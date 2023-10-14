//
//  nluDemo.cpp
//  MNN
//
//  Created by MNN on b'2021/06/01'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>

using namespace MNN::Express;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./nluDemo.out DIR XXX.mnn [numberThread] [seqLength]\n");
        MNN_ERROR("Need: segment_ids.txt, input_ids.txt, input_mask.txt\n");
        return 0;
    }
    std::string dirPrefix            = argv[1];
    const std::string model_filename = argv[2];
    const std::vector<std::string> input_names{"input_ids", "input_mask", "segment_ids"};
    const std::vector<std::string> output_names{"loss/pred_prob"};
    MNN_PRINT("Inputs:\n");
    for (auto s : input_names) {
        MNN_PRINT("%s, ", s.c_str());
    }
    MNN_PRINT("\nOutputs:\n");
    for (auto s : output_names) {
        MNN_PRINT("%s, ", s.c_str());
    }
    MNN_PRINT("\n");
    int nthreads = (argc > 3) ? std::stoi(argv[3]) : 1;
    int dim      = (argc > 4) ? std::stoi(argv[4]) : 128;

    MNN::BackendConfig config;
    std::shared_ptr<MNN::Express::Executor> executor(Executor::newExecutor(MNN_FORWARD_CPU, config, nthreads));
    MNN::Express::ExecutorScope scope(executor);

    std::cout << "Dim=" << dim << std::endl;
    std::cout << "Begin" << std::endl;

    std::unique_ptr<Module> module;
    Module::Config mdconfig;
    mdconfig.rearrange = true; // Reduce net buffer memory
    {
        AUTOTIME;
        module.reset(Module::load(input_names, output_names, model_filename.c_str(), &mdconfig));
    }
    std::cout << "Loaded" << std::endl;

    // Load inputs
    std::vector<VARP> inputs(3);
    inputs[0] = _Input({1, dim}, NHWC, halide_type_of<int>());
    inputs[1] = _Input({1, dim}, NHWC, halide_type_of<int>());
    inputs[2] = _Input({1, dim}, NHWC, halide_type_of<int>());
    // inputs[3] = _Input({1, dim}, NHWC, halide_type_of<int>());

    std::ifstream f_input_ids(dirPrefix + "/input_ids.txt");
    for (int i=0; i<dim; ++i) {
        f_input_ids >> inputs[0]->writeMap<int>()[i];
    }
    f_input_ids.close();
    std::ifstream f_input_mask(dirPrefix + "/input_mask.txt");
    for (int i=0; i<dim; ++i) {
        f_input_mask >> inputs[1]->writeMap<int>()[i];
    }
    f_input_mask.close();

    std::cout << "read input_mask done " << inputs[1]->getInfo()->size << std::endl;
    std::ifstream f_segment_ids(dirPrefix + "/segment_ids.txt");
    for (int i=0; i<dim; ++i) {
        f_segment_ids >> inputs[2]->writeMap<int>()[i];
    }
    f_segment_ids.close();
    std::cout << "read segment_ids done " << inputs[2]->getInfo()->size << std::endl;

    // Check output by run twice
    std::vector<VARP> outputs  = module->onForward(inputs);
    std::vector<VARP> outputs2 = module->onForward(inputs);
    std::ofstream ofile(dirPrefix + "/mnn.out");
    for (int i = 0; i < outputs[0]->getInfo()->size; ++i) {
        ofile << outputs[0]->readMap<float>()[i] << std::endl;
    }
    ofile.close();
    std::ofstream ofile2(dirPrefix + "/mnn.out2");
    for (int i = 0; i < outputs2[0]->getInfo()->size; ++i) {
        ofile2 << outputs2[0]->readMap<float>()[i] << std::endl;
    }
    ofile2.close();

    // Benchmark
    int benchTime = 20;
    if (true) {
        auto globalExecutor = MNN::Express::ExecutorScope::Current();
        for (int i = 0; i < 3; ++i) {
            outputs = module->onForward(inputs);
        }
        outputs = module->onForward(inputs);
        {
            MNN::Timer autoTime;
            for (int i = 0; i < benchTime; ++i) {
                MNN::AutoTime _t(0, "Once time");
                // std::cout << i << std::endl;
                outputs = module->onForward(inputs);
            }
            float cost_time = ((float)autoTime.durationInUs() / (float)1000 / 20.f);
            printf("cost_time = %f\n", cost_time);
            printf("done!\n");
        }
    }
    return 0;
}
