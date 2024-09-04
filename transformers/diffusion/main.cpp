#include <iostream>
#include "pipeline.hpp"

int main(int argc, const char* argv[]) {
    if (argc < 7) {
        MNN_PRINT("=====================================================================================================================\n");
        MNN_PRINT("Usage: ./diffusion_demo <resource_path> <model_type> <output_image_name> <memory_mode> <backend_type> <input_text>\n");
        MNN_PRINT("=====================================================================================================================\n");
        return 0;
    }

    auto resource_path = argv[1];
    auto model_type = (diffusion::DiffusionModelType)atoi(argv[2]);
    auto img_name = argv[3];
    auto memory_mode = atoi(argv[4]);
    auto backend_type = (MNNForwardType)atoi(argv[5]);
    std::string input_text;
    for (int i = 6; i < argc; ++i) {
        input_text += argv[i];
        if (i < argc - 1) {
            input_text += " ";
        }
    }
    
    MNN_PRINT("Model resource path: %s\n", resource_path);
    if(model_type == diffusion::STABLE_DIFFUSION_1_5) {
        MNN_PRINT("Model type is stable diffusion 1.5\n");
    } else if (model_type == diffusion::STABLE_DIFFUSION_TAIYI_CHINESE) {
        MNN_PRINT("Model type is stable diffusion taiyi chinese version\n");
    } else {
        MNN_PRINT("Error: Model type %d not supported, please check\n", (int)model_type);
    }

    if(memory_mode == 0) {
	    MNN_PRINT("(Memory Lack) Each diffusion model will be initilized when using, freed after using.\n");
    } else {
	    MNN_PRINT("(Memory Enough) All Diffusion models will be initilized when application enter.\n");
    }
    MNN_PRINT("Backend type: %d\n", (int)backend_type);
    MNN_PRINT("Output image name: %s\n", img_name);
    MNN_PRINT("Prompt text: %s\n", input_text.c_str());

    
    diffusion::Pipeline pipeline(resource_path, model_type, backend_type, memory_mode);
    pipeline.run(input_text, img_name);
    return 0;
}
