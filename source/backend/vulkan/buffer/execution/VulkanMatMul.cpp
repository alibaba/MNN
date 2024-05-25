//
//  VulkanMatMul.cpp
//  MNN
//
//  Created by MNN on 2020/03/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanMatMul.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {
VulkanMatMul::VulkanMatMul(bool transposeA, bool transposeB, Backend* bn, bool hasBias) : VulkanBasicExecution(bn) {
    mTransposeA = transposeA;
    mTransposeB = transposeB;
    auto vkbackend = static_cast<VulkanBackend*>(bn);
    mParam = vkbackend->allocUniform();
    mHasBias = hasBias;
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
}
VulkanMatMul::~ VulkanMatMul() {
    auto vkbackend = static_cast<VulkanBackend*>(backend());
    vkbackend->recycleUniform(mParam);
}

ErrorCode VulkanMatMul::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                           const VulkanCommandPool::Buffer *cmdBuffer) {
    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];
    Tensor* C       = outputs[0];
    auto w0         = inputs[0]->length(1);
    auto h0         = inputs[0]->length(0);
    auto e = A->length(0);
    auto h = B->length(1);
    auto l = A->length(1);
    if (mTransposeA) {
        l = A->length(0);
        e = A->length(1);
    }
    if (mTransposeB) {
        h = B->length(0);
    }
    int totalSize = e * h;
    auto param = reinterpret_cast<VulkanBatchMatMulInfo*>(mParam->map());
    ::memset(param, 0, sizeof(VulkanBatchMatMulInfo));
    param->size[3] = 1;
    param->size[0] = e;
    param->size[1] = l;
    param->size[2] = h;
    param->stride_o[1] = 0;
    param->stride_o[0] = h;
    param->stride_o[2] = 1;
    param->stride_c[0] = 0;
    param->stride_c[1] = 0;
    param->stride_c[2] = 1;
    param->iter[0] = -1;
    param->iter[1] = -1;
    param->iter[2] = -1;

    param->stride_a[2] = 0;
    if (mTransposeA) {
        param->stride_a[0] = 1;
        param->stride_a[1] = e;
    } else {
        param->stride_a[0] = l;
        param->stride_a[1] = 1;
    }
    
    param->stride_b[0] = 0;
    if (mTransposeB) {
        param->stride_b[1] = 1;
        param->stride_b[2] = l;
    } else {
        param->stride_b[1] = h;
        param->stride_b[2] = 1;
    }
    mParam->unmap();
    auto vkBn = static_cast<VulkanBackend*>(backend());
    mDescribe->writeBuffer(vkBn->getBuffer(C), 0);
    mDescribe->writeBuffer(vkBn->getBuffer(A), 1);
    mDescribe->writeBuffer(vkBn->getBuffer(B), 2);
    int offset = 3;
    if (inputs.size() > 2) {
        mDescribe->writeBuffer(vkBn->getBuffer(inputs[2]), 3);
        offset = 4;
    }
    // stride's
    mDescribe->writeBuffer(vkBn->getBuffer(A), offset);
    mDescribe->writeBuffer(vkBn->getBuffer(A), offset + 1);
    mDescribe->writeBuffer(vkBn->getBuffer(A), offset + 2);
    if (inputs.size() > 2) {
        mDescribe->writeBuffer(vkBn->getBuffer(A), offset + 3);
    }
    mDescribe->writeBuffer(mParam->buffer(), offset * 2, mParam->size());
    mPipeline->bind(cmdBuffer->get(), mDescribe->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize,256), 1, 1);

    return NO_ERROR;
}
class VulkanMatMulCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        const auto mat = op->main_as_MatMul();
        return new VulkanMatMul(mat->transposeA(), mat->transposeB(), bn, inputs.size() > 2);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_MatMul, new VulkanMatMulCreator);
    return true;
}();

}
