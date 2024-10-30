#include <fstream>
#include "app/chat/chat.hpp"

using namespace MNN::Transformer;

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json [prompt.txt]" << std::endl;
        return 0;
    }
    std::string config_path = argv[1];
    Chat chat(config_path);
    if (argc == 3) {
        // prompt from file
        std::string prompt_file = argv[2];
        std::cout << "prompt file is " << prompt_file << std::endl;
        std::ifstream prompt_fs(prompt_file); 
        chat.chat(false, true, &prompt_fs);
    } else {
        // chat in terminal
        chat.chat();
    }
    return 0;
}