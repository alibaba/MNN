//
//  VulkanBasicExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanBasicExecution_hpp
#define VulkanBasicExecution_hpp

#include "core/Execution.hpp"
#include "VulkanBackend.hpp"

namespace MNN {
class VulkanBasicExecution {
public:
    VulkanBasicExecution(Backend *bn) : mBackend(bn) {
        //Do nothing
    }
    virtual ~VulkanBasicExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) = 0;

    Backend* backend() {
        return mBackend;
    }
private:
    Backend* mBackend;
};

class VulkanBasicExecutionDirect : public Execution {
public:
    VulkanBasicExecutionDirect(std::shared_ptr<VulkanBasicExecution> encoder);
    virtual ~ VulkanBasicExecutionDirect() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<VulkanBasicExecution> mEncoder;
    std::shared_ptr<VulkanCommandPool::Buffer> mCmdBuffer;
};
class VulkanBasicExecutionInDirect : public Execution {
public:
    VulkanBasicExecutionInDirect(std::shared_ptr<VulkanBasicExecution> encoder);
    virtual ~ VulkanBasicExecutionInDirect() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_ERROR;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
private:
    std::shared_ptr<VulkanBasicExecution> mEncoder;
};
typedef int ivec2[2];
typedef int ivec3[3];
typedef int ivec4[4];

typedef float vec2[2];
typedef float vec3[3];
typedef float vec4[4];

} // namespace MNN
#endif /* VulkanBasicExecution_hpp */
