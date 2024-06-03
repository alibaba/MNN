#include <iostream>
#include "pipeline.hpp"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("Usage: ./diffusion_demo <resource_path> <output_image_name> <input_text>");
        return 0;
    }

    auto resource_path = argv[1];
    auto img_name = argv[2];
    
    std::string input_text;
    for (int i = 3; i < argc; ++i) {
        input_text += argv[i];
        if (i < argc - 1) {
            input_text += " ";
        }
    }
    
    printf("model resource path: %s\n", resource_path);
    printf("output img_name: %s\n", img_name);
    printf("input texts: %s\n", input_text.c_str());
    
    diffusion::Pipeline pipeline(resource_path);
    pipeline.run(input_text, img_name);
    return 0;
}
