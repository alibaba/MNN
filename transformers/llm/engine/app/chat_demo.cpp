#include "app/chat/chat.hpp"

using namespace MNN::Transformer;

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json" << std::endl;
        return 0;
    }
    std::string config_path = argv[1];
    Chat chat(config_path);
    chat.chat();
    return 0;
}