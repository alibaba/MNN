#ifndef OptimizeCommandBuffer_hpp
#define OptimizeCommandBuffer_hpp
#include "ConvertTflite.hpp"
namespace MNN {
class OptimizeCommandBuffer {
public:
    OptimizeCommandBuffer(ConvertTflite* root);
    ~ OptimizeCommandBuffer();
    ConvertTflite::CommandBuffer reduce(ConvertTflite::CommandBuffer&& cmdBuffer);
private:
    ConvertTflite* mRoot;
};
};
#endif
