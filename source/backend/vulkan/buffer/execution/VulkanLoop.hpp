#ifndef VulkanRaster_hpp
#define VulkanRaster_hpp
#include "VulkanBasicExecution.hpp"
namespace MNN {
class VulkanLoop {
public:
    static VulkanBasicExecution* create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op, Backend* bn);
};
};

#endif
