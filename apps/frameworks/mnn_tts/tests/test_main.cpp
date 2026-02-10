#include "mnn_tts/common.h"
#include "mnn_tts_sdk.hpp"
#include "supertonic/mnn_supertonic_tts_impl.hpp"
#include <iostream>

int main(int argc, char** argv) {
    mnn_tts::sharedFunction();
    mnn_tts::platformFunction();
    
    // Test supertonic TTS compilation
    // Note: This is a compilation test only, actual model files are required for runtime testing
    std::cout << "Testing Supertonic TTS compilation..." << std::endl;
    
    // Verify that MNNSupertonicTTSImpl class is accessible
    // We can't instantiate it without model files, but we can verify the header compiles
    std::cout << "Supertonic TTS header compiled successfully" << std::endl;
    
    return 0;
}
