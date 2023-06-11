#ifndef VulkanLoop_hpp
#define VulkanLoop_hpp
#include "VulkanBasicExecution.hpp"
namespace MNN {
struct VulkanBatchMatMulInfo {
    ivec4 size;
    ivec4 stride_o;
    ivec4 stride_a;
    ivec4 stride_b;
    ivec4 stride_c;
    ivec4 step;
    ivec4 iter;
};
class VulkanLoop {
public:
    static VulkanBasicExecution* create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op, Backend* bn);
};
};

#endif
