#include <iostream>
#include "pipeline.hpp"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("Usage: ./diffusion_demo <senetence> <img_name> <resource_path>");
        return 1;
    }
    auto sentence = argv[1];
    auto img_name = argv[2];
    auto resource_path = argv[3];

    printf("input setnetce: %s\n", sentence);
    printf("output img_name: %s\n", img_name);
    printf("model resource path: %s\n", resource_path);
    diffusion::Pipeline pipeline(resource_path);
    pipeline.run(sentence, img_name);
    return 0;
}
