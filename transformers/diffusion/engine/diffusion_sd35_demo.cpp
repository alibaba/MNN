//
//  diffusion_sd35_demo.cpp
//
//  Created by zlaa on 2025/12/18.
//

#include <iostream>
#include "diffusion/diffusion_sd35.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
using namespace MNN::DIFFUSION;

int main(int argc, const char* argv[]) {
    if (argc < 9) {
        MNN_PRINT("=====================================================================================================================\n");
        MNN_PRINT("Usage: ./diffusion_sd35_demo <resource_path> <model_type> <memory_mode> <backend_type> <iteration_num> <random_seed> <output_image_name> <prompt_text>\n");
        MNN_PRINT("=====================================================================================================================\n");
        return 0;
    }

    auto resource_path = argv[1];
    // auto model_type = argv[2];
    auto memory_mode = atoi(argv[3]);
    auto backend_type = (MNNForwardType)atoi(argv[4]);
    auto iteration_num = atoi(argv[5]);
    auto random_seed = atoi(argv[6]);
    auto img_name = argv[7];

    std::string input_text;
    for (int i = 8; i < argc; ++i) {
        input_text += argv[i];
        if (i < argc - 1) {
            input_text += " ";
        }
    }
    
    MNN_PRINT("Model resource path: %s\n", resource_path);
    MNN_PRINT("Model type is stable diffusion 3.5\n");

    if(memory_mode == 1) {
        MNN_PRINT("(Memory Enough) All Diffusion models will be initialized when application enter. with fast initialization\n");
    } else {
        MNN_PRINT("(Memory Lack) Each diffusion model will be initialized when using, freed after using. with slow initialization\n");
    }
    MNN_PRINT("Backend type: %d\n", (int)backend_type);
    MNN_PRINT("Output image name: %s\n", img_name);
    MNN_PRINT("Prompt text: %s\n", input_text.c_str());

    // Create Diffusion_sd35 instance
    // We pass STABLE_DIFFUSION_1_5 as placeholder for modelType since DiffusionSD35 is dedicated to SD3.5
    std::shared_ptr<DiffusionSD35> diffusion(DiffusionSD35::createDiffusionSD35(resource_path, STABLE_DIFFUSION_1_5, backend_type, memory_mode));
    
    if (!diffusion->load()) {
        MNN_ERROR("Failed to load diffusion models\n");
        return -1;
    }
    
    diffusion->run(input_text, img_name, iteration_num, random_seed, [](int progress){
        printf("Progress: %d%%\r", progress);
        fflush(stdout);
    });
    
    printf("\nDone.\n");
    return 0;
}
