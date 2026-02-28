//
//  tokenizer_demo.cpp
//
//  Created by MNN on 2025/09/01.
//  ZhaodeWang
//

#include "../src/tokenizer.hpp"
#include "../src/tokenizer.cpp"

using namespace MNN::Transformer;

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " tokenizer.txt" << std::endl;
        return 0;
    }
    std::string tokenizer_path = argv[1];
    std::unique_ptr<Tokenizer> tokenizer(Tokenizer::createTokenizer(tokenizer_path));
    auto ids = tokenizer->encode("介绍一下北京的首都");
    std::string str;
    for (auto id : ids) {
        printf("%d, ", id);
        str += tokenizer->decode(id);
    }
    printf("\n");
    printf("%s\n", str.c_str());
    return 0;
}
