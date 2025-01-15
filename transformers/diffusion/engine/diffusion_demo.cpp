#include <iostream>
#include "diffusion/diffusion.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
using namespace MNN::DIFFUSION;

int main(int argc, const char* argv[]) {
    if (argc < 8) {
        MNN_PRINT("=====================================================================================================================\n");
        MNN_PRINT("Usage: ./diffusion_demo <resource_path> <model_type> <memory_mode> <backend_type> <iteration_num> <output_image_name> <prompt_text>\n");
        MNN_PRINT("=====================================================================================================================\n");
        return 0;
    }

    auto resource_path = argv[1];
    auto model_type = (DiffusionModelType)atoi(argv[2]);
    auto memory_mode = atoi(argv[3]);
    auto backend_type = (MNNForwardType)atoi(argv[4]);
    auto iteration_num = atoi(argv[5]);
    auto img_name = argv[6];

    std::string input_text;
    for (int i = 7; i < argc; ++i) {
        input_text += argv[i];
        if (i < argc - 1) {
            input_text += " ";
        }
    }
    
    MNN_PRINT("Model resource path: %s\n", resource_path);
    if(model_type == STABLE_DIFFUSION_1_5) {
        MNN_PRINT("Model type is stable diffusion 1.5\n");
    } else if (model_type == STABLE_DIFFUSION_TAIYI_CHINESE) {
        MNN_PRINT("Model type is stable diffusion taiyi chinese version\n");
    } else {
        MNN_PRINT("Error: Model type %d not supported, please check\n", (int)model_type);
    }

    if(memory_mode == 0) {
	    MNN_PRINT("(Memory Lack) Each diffusion model will be initialized when using, freed after using. with slow initialization\n");
    } else {
	    MNN_PRINT("(Memory Enough) All Diffusion models will be initialized when application enter. with fast initialization\n");
    }
    MNN_PRINT("Backend type: %d\n", (int)backend_type);
    MNN_PRINT("Output image name: %s\n", img_name);
    MNN_PRINT("Prompt text: %s\n", input_text.c_str());

    
    std::unique_ptr<Diffusion> diffusion(Diffusion::createDiffusion(resource_path, model_type, backend_type, memory_mode, iteration_num));

    diffusion->load();
    
    // callback to show progress
    auto progressDisplay = [](int progress) {
        std::cout << "Progress: " << progress << "%" << std::endl;
    };
    diffusion->run(input_text, img_name, progressDisplay);
    
    /*
     when need multi text-generation-image:
     if you choose memory lack mode, need diffusion load with each diffusion run.
     if you choose memory enough mode,  just start another diffusion run, only need diffusion load in first time.
     */
    while(0) {
        if(memory_mode == 0) {
            diffusion->load();
        }
        
        diffusion->run("a big horse", "demo_2.jpg", progressDisplay);
    }
    return 0;
}
