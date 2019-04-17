//
//  VulkanBasicExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanBasicExecution_hpp
#define VulkanBasicExecution_hpp

#include "Execution.hpp"
#include "VulkanBackend.hpp"

namespace MNN {
class VulkanBasicExecution : public Execution {
public:
    VulkanBasicExecution(Backend *bn);
    virtual ~VulkanBasicExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) = 0;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<VulkanCommandPool::Buffer> mCmdBuffer;
};
typedef int ivec2[2];
typedef int ivec3[3];
typedef int ivec4[4];

typedef float vec2[2];
typedef float vec3[3];
typedef float vec4[4];

} // namespace MNN
#endif /* VulkanBasicExecution_hpp */
