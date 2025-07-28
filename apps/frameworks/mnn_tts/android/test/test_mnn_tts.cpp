#include <iostream>
extern "C" void mnn_tts_dummy(); // Assume there is such an exported function

int main() {
    std::cout << "mnn_tts test start" << std::endl;
    // Call a function in the library, please replace with real API in actual project
    mnn_tts_dummy();
    std::cout << "mnn_tts test success" << std::endl;
    return 0;
} 