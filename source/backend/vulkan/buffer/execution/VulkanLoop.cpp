#include "VulkanLoop.hpp"
#include "VulkanBinary.hpp"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include "core/OpCommonUtils.hpp"
#include "core/Macro.h"

namespace MNN {

static void _setTensorStack(std::vector<Tensor*>& result, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const LoopParam* loop) {
    if (loop->inputIndexes() != nullptr) {
        for (int i=0; i<loop->inputIndexes()->size(); ++i) {
            result[loop->inputIndexes()->data()[i]] = inputs[i];
        }
    }
    for (int i=0; i<loop->outputIndexes()->size(); ++i) {
        result[loop->outputIndexes()->data()[i]] = outputs[i];
    }
}

class VulkanBatchMatMul : public VulkanBasicExecution {
public:
    VulkanBatchMatMul(const LoopParam* loop, Backend *bn) : VulkanBasicExecution(bn) {
        mLoop = loop;
        auto vkbackend = static_cast<VulkanBackend*>(bn);
        mParam.reset(new VulkanBuffer(vkbackend->getMemoryPool(), false, sizeof(VulkanBatchMatMulInfo), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        
        auto cmd = loop->commands()->GetAs<RegionCommand>(0);
        mHasBias = cmd->indexes()->size() > 3;
        if (!mHasBias) {
            mPipeline = vkbackend->getPipeline("glsl_matmulunit_comp", {
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            });
        } else {
            mPipeline = vkbackend->getPipeline("glsl_matmulunit_HAS_BIAS_comp", {
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            });
        }
        mDescribe.reset(mPipeline->createSet());
        mTensors.resize(mLoop->tensorNumber());
    }
    virtual ~VulkanBatchMatMul() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override {
        _setTensorStack(mTensors, inputs, outputs, mLoop);
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto size = cmd->size()->data();
        auto AStride = cmd->view()->GetAs<View>(1)->stride()->data();
        auto BStride = cmd->view()->GetAs<View>(2)->stride()->data();
        auto OStride = cmd->view()->GetAs<View>(0)->stride()->data();
        int totalSize = mLoop->loopNumber() * size[0] * size[1] * size[2];
        auto param = reinterpret_cast<VulkanBatchMatMulInfo*>(mParam->map());
        param->size[3] = mLoop->loopNumber();
        auto vkBn = static_cast<VulkanBackend*>(backend());
        for (int i=0; i<3; ++i) {
            param->size[i] = size[i];
            param->stride_o[i] = OStride[i];
            param->stride_a[i] = AStride[i];
            param->stride_b[i] = BStride[i];
        }
        param->stride_o[3] = cmd->view()->GetAs<View>(0)->offset();
        param->stride_a[3] = cmd->view()->GetAs<View>(1)->offset();
        param->stride_b[3] = cmd->view()->GetAs<View>(2)->offset();
        if (mHasBias) {
            param->stride_c[3] = cmd->view()->GetAs<View>(3)->offset();
        }
        ::memcpy(param->step, cmd->steps()->data(), cmd->steps()->size() * sizeof(int));
        ::memcpy(param->iter, cmd->iterIndexes()->data(), cmd->iterIndexes()->size() * sizeof(int));
        std::vector<VULKAN_TENSOR> strideBuffers(cmd->indexes()->size());
        for (int i=0; i<cmd->indexes()->size(); ++i) {
            std::get<0>(strideBuffers[i]) = 0;
            if (param->iter[i] >= 0) {
                strideBuffers[i] = vkBn->getBuffer(mTensors[param->iter[i]]);
            }
        }
        mParam->unmap();
        for (int i=0; i<cmd->indexes()->size(); ++i) {
            auto tensor = mTensors[cmd->indexes()->data()[i]];
            mDescribe->writeBuffer(vkBn->getBuffer(tensor), i);
        }
        for (int i=0; i<strideBuffers.size(); ++i) {
            if (0 != std::get<0>(strideBuffers[i])) {
                mDescribe->writeBuffer(strideBuffers[i], cmd->indexes()->size() + i);
            } else {
                mDescribe->writeBuffer(vkBn->getBuffer(inputs[0]), cmd->indexes()->size() + i);
            }
        }
        mDescribe->writeBuffer(mParam->buffer(), cmd->indexes()->size() * 2, mParam->size());
        mPipeline->bind(cmdBuffer->get(), mDescribe->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize,256), 1, 1);

        return NO_ERROR;
    }
private:
    const LoopParam* mLoop;
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanBuffer> mParam;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescribe;
    std::vector<Tensor*> mTensors;
    bool mHasBias = false;
};


struct BinaryBroadCastInfo {
    ivec4 srcview0;
    ivec4 srcview1;
    ivec4 dstview;
    ivec4 size;
};

class VulkanBinaryBroadCast : public VulkanBasicExecution {
public:
    VulkanBinaryBroadCast(const LoopParam* loop, Backend *bn, bool isInt) : VulkanBasicExecution(bn) {
        mLoop = loop;
        auto vkbackend = static_cast<VulkanBackend*>(bn);
        mParam.reset(new VulkanBuffer(vkbackend->getMemoryPool(), false, sizeof(BinaryBroadCastInfo), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        std::string shaderName;
        if (isInt) {
            shaderName = "glsl_binary_blit_int_" + VulkanBinary::getMidName( mLoop->commands()->GetAs<RegionCommand>(0)->op()) + "_comp";
        } else {
            shaderName = "glsl_binary_blit_" + VulkanBinary::getMidName( mLoop->commands()->GetAs<RegionCommand>(0)->op()) + "_comp";
        }

        mPipeline = vkbackend->getPipeline(shaderName, {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        });
        mDescribe.reset(mPipeline->createSet());
        mTensors.resize(mLoop->tensorNumber());
    }
    virtual ~VulkanBinaryBroadCast() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override {
        _setTensorStack(mTensors, inputs, outputs, mLoop);
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto size = cmd->size()->data();
        auto vkBn = static_cast<VulkanBackend*>(backend());
        auto srcStride0 = cmd->view()->GetAs<View>(1)->stride()->data();
        auto srcStride1 = cmd->view()->GetAs<View>(2)->stride()->data();
        auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
        int totalSize = size[0] * size[1] * size[2];
        auto param = reinterpret_cast<BinaryBroadCastInfo*>(mParam->map());
        for (int i=0; i<3; ++i) {
            param->size[i] = size[i];
            param->srcview0[i] = srcStride0[i];
            param->srcview1[i] = srcStride1[i];
            param->dstview[i] = dstStride[i];
        }
        param->srcview0[3] = cmd->view()->GetAs<View>(1)->offset();
        param->srcview1[3] = cmd->view()->GetAs<View>(2)->offset();
        param->dstview[3] = cmd->view()->GetAs<View>(0)->offset();
        param->size[3] = size[0] * size[1] * size[2];
        mParam->unmap();
        auto dstTensor = mTensors[cmd->indexes()->data()[0]];
        auto srcTensor = mTensors[cmd->indexes()->data()[1]];
        auto srcTensor1 = mTensors[cmd->indexes()->data()[2]];
        mDescribe->writeBuffer(vkBn->getBuffer(dstTensor), 0);
        mDescribe->writeBuffer(vkBn->getBuffer(srcTensor), 1);
        mDescribe->writeBuffer(vkBn->getBuffer(srcTensor1), 2);
        mDescribe->writeBuffer(mParam->buffer(), 3, mParam->size());
        mPipeline->bind(cmdBuffer->get(), mDescribe->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize,256), 1, 1);

        return NO_ERROR;
    }
private:
    const LoopParam* mLoop;
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanBuffer> mParam;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescribe;
    std::vector<Tensor*> mTensors;
};
struct GatherInfo {
    ivec4 stride;
    ivec4 size;
    ivec4 extent;
    ivec4 step;
    ivec4 iter;
};

class VulkanGather : public VulkanBasicExecution {
public:
    VulkanGather(const LoopParam* loop, Backend *bn) : VulkanBasicExecution(bn) {
        mLoop = loop;
        auto vkbackend = static_cast<VulkanBackend*>(bn);
        mParam.reset(new VulkanBuffer(vkbackend->getMemoryPool(), false, sizeof(GatherInfo), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        
        mPipeline = vkbackend->getPipeline("glsl_blitregion_comp", {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        });
        mDescribe.reset(mPipeline->createSet());
        mTensors.resize(mLoop->tensorNumber());
    }
    virtual ~VulkanGather() = default;
    virtual ErrorCode onEncode(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override {
        _setTensorStack(mTensors, inputs, outputs, mLoop);
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto size = cmd->size()->data();
        auto vkBn = static_cast<VulkanBackend*>(backend());
        auto srcStride = cmd->view()->GetAs<View>(1)->stride()->data();
        auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
        int totalSize = mLoop->loopNumber() * size[0] * size[1] * size[2];
        VULKAN_TENSOR srcOffsetBuffer;
        std::get<0>(srcOffsetBuffer) = 0;
        VULKAN_TENSOR dstOffsetBuffer;
        std::get<0>(dstOffsetBuffer) = 0;
        auto param = reinterpret_cast<GatherInfo*>(mParam->map());
        for (int i=0; i<3; ++i) {
            param->size[i] = size[i];
            param->stride[i] = srcStride[i];
            param->extent[i] = dstStride[i];
        }
        param->stride[3] = cmd->view()->GetAs<View>(1)->offset();
        param->extent[3] = cmd->view()->GetAs<View>(0)->offset();
        param->size[3] = size[0] * size[1] * size[2];
        param->step[3] = totalSize;
        param->step[0] = cmd->steps()->data()[0];
        param->step[1] = cmd->steps()->data()[1];
        param->iter[0] = 0;
        param->iter[1] = 0;
        auto iterIndex = cmd->iterIndexes()->data();
        if (iterIndex[0] >= 0) {
            dstOffsetBuffer = vkBn->getBuffer(mTensors[iterIndex[0]]);
            param->iter[0] = 1;
        }
        if (iterIndex[1] >= 0) {
            srcOffsetBuffer = vkBn->getBuffer(mTensors[iterIndex[1]]);
            param->iter[1] = 1;
        }
        mParam->unmap();
        auto dstTensor = mTensors[cmd->indexes()->data()[0]];
        auto srcTensor = mTensors[cmd->indexes()->data()[1]];
        mDescribe->writeBuffer(vkBn->getBuffer(dstTensor), 0);
        mDescribe->writeBuffer(vkBn->getBuffer(srcTensor), 1);
        if (std::get<0>(srcOffsetBuffer) != 0) {
            mDescribe->writeBuffer(srcOffsetBuffer, 2);
        } else {
            // Use Invalide buffer
            mDescribe->writeBuffer(vkBn->getBuffer(srcTensor), 2);
        }
        if (std::get<0>(dstOffsetBuffer) != 0) {
            mDescribe->writeBuffer(dstOffsetBuffer, 3);
        } else {
            // Use Invalide buffer
            mDescribe->writeBuffer(vkBn->getBuffer(srcTensor), 3);
        }
        mDescribe->writeBuffer(mParam->buffer(), 4, mParam->size());
        mPipeline->bind(cmdBuffer->get(), mDescribe->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize,256), 1, 1);

        return NO_ERROR;
    }
private:
    const LoopParam* mLoop;
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanBuffer> mParam;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescribe;
    std::vector<Tensor*> mTensors;
};

VulkanBasicExecution* VulkanLoop::create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op, Backend* bn) {
    auto loop = op->main_as_LoopParam();
    if (nullptr == loop || loop->commands() == nullptr) {
        return nullptr;
    }
    if (nullptr != loop->initCommand()) {
        return nullptr;
    }
    // Make Tensor Stack
    if (1 == loop->commands()->size()) {
        auto cmd = loop->commands()->GetAs<RegionCommand>(0);
        auto subop = cmd->op();
        if (OpType_UnaryOp == subop->type() && nullptr == subop->main() && cmd->fuse() < 0) {
            return new VulkanGather(loop, bn);
        }
        if (OpType_MatMul == subop->type() && loop->parallel()) {
            return new VulkanBatchMatMul(loop, bn);
        }
        if (OpType_BinaryOp == subop->type() && cmd->fuse() < 0 && 1 == loop->loopNumber()) {
            bool isInt = inputs[1]->getType().code == halide_type_int;
            return new VulkanBinaryBroadCast(loop, bn, isInt);
        }
    }
    return nullptr;
}

class VulkanLoopCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return VulkanLoop::create(inputs, outputs, op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_While, new VulkanLoopCreator);
    return true;
}();

};
