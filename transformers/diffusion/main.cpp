#include <iostream>
#include "pipeline.hpp"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./diffusion_demo <resource_path> <model_type> <output_image_name> <input_text>\n");
        return 0;
    }

    auto resource_path = argv[1];
    auto model_type = (diffusion::DiffusionModelType)atoi(argv[2]);
    auto img_name = argv[3];
    
    std::string input_text;
    for (int i = 4; i < argc; ++i) {
        input_text += argv[i];
        if (i < argc - 1) {
            input_text += " ";
        }
    }
    
    MNN_PRINT("model resource path: %s\n", resource_path);
    if(model_type == diffusion::STABLE_DIFFUSION_1_5) {
        MNN_PRINT("model type is stable diffusion 1.5\n");
    } else if (model_type == diffusion::STABLE_DIFFUSION_TAIYI_CHINESE) {
        MNN_PRINT("model type is stable diffusion taiyi chinese version\n");
    } else {
        MNN_PRINT("model type: %d not supported, please check\n", (int)model_type);
    }
    MNN_PRINT("output img_name: %s\n", img_name);
    MNN_PRINT("input texts: %s\n", input_text.c_str());

    
    diffusion::Pipeline pipeline(resource_path, model_type);
    pipeline.run(input_text, img_name);
    return 0;
}
