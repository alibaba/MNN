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
#ifdef ENABLE_VULKAN_TIME_PROFILE
static constexpr const char* kVulkanTimeProfileDefaultExecutionName = "General_Execution";
#endif

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
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
        return false;
    }
#ifdef ENABLE_VULKAN_TIME_PROFILE
    void setName(const char * name) {
        mName = name;
    }
    std::string getName() {
        return mName;
    }
protected:
    std::string mName = kVulkanTimeProfileDefaultExecutionName;
#endif
private:
    Backend* mBackend;
};

class VulkanBasicExecutionDirect : public Execution {
public:
    VulkanBasicExecutionDirect(std::shared_ptr<VulkanBasicExecution> encoder);
    virtual ~ VulkanBasicExecutionDirect() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return mEncoder->onClone(bn, op, nullptr);
        }
        VulkanBasicExecution* dstExe = nullptr;
        mEncoder->onClone(bn, op, &dstExe);
        if (nullptr == dstExe) {
            return false;
        }
#ifdef ENABLE_VULKAN_TIME_PROFILE
        if (dstExe->getName() == kVulkanTimeProfileDefaultExecutionName && nullptr != op) {
            dstExe->setName(EnumNameOpType(op->type()));
        }
#endif
        std::shared_ptr<VulkanBasicExecution> dstExePtr(dstExe);
        *dst = new VulkanBasicExecutionDirect(dstExePtr);
        return true;
    }

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
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return mEncoder->onClone(bn, op, nullptr);
        }
        VulkanBasicExecution* dstExe = nullptr;
        mEncoder->onClone(bn, op, &dstExe);
        if (nullptr == dstExe) {
            return false;
        }
#ifdef ENABLE_VULKAN_TIME_PROFILE
        if (dstExe->getName() == kVulkanTimeProfileDefaultExecutionName && nullptr != op) {
            dstExe->setName(EnumNameOpType(op->type()));
        }
#endif
        std::shared_ptr<VulkanBasicExecution> dstExePtr(dstExe);
        *dst = new VulkanBasicExecutionInDirect(dstExePtr);
        return true;
    }
private:
    std::shared_ptr<VulkanBasicExecution> mEncoder;
};
typedef int ivec2[2];
typedef int ivec3[3];
typedef int ivec4[4];

typedef float vec2[2];
typedef float vec3[3];
typedef float vec4[4];

typedef int16_t f16vec2[2];
typedef int16_t f16vec3[3];
typedef int16_t f16vec4[4];

} // namespace MNN
#endif /* VulkanBasicExecution_hpp */
