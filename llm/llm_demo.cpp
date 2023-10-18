//
//  llm_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm.hpp"
#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./llm_demo.out <model_path>" << std::endl;
        return 0;
    }
    std::string model_dir = argv[1];
    std::cout << "model path is " << model_dir << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(model_dir));
    llm->load(model_dir);
    llm->response("你好");
    return 0;
}
