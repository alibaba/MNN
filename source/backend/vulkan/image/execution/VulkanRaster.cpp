#include "VulkanRaster.hpp"
#include "VulkanMatMul.hpp"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include "core/OpCommonUtils.hpp"
#include "core/Macro.h"
namespace MNN {
static void writeNCHW(int* dims, Tensor* origin) {
    int w     = std::max(origin->width(), 1);
    int h     = std::max(origin->height(), 1);
    int b     = origin->batch();
    dims[0]   = w;
    dims[1]   = h;
    dims[2]   = origin->channel();
    dims[3]   = b;
}
struct SamplerInfo {
    ivec4 stride;//stride[3] + offset
    ivec4 size;//size[3] + totalSize
    ivec4 extent;//dstStride[3]+dstOffset
    ivec4 imageSize;// srcwh and dstwh
    ivec2 depth;//c4 for src and dst
};

static void writeSamplerInfo(SamplerInfo& info, const Tensor::InsideDescribe::Region& sampler) {
    int sizeTotal = 1;
    for (int i=0; i<3; ++i) {
        info.size[i] = sampler.size[i];
        info.stride[i] = sampler.src.stride[i];
        info.extent[i] = sampler.dst.stride[i];
        sizeTotal *= info.size[i];
    }
    info.size[3] = sizeTotal;
    info.stride[3] = sampler.src.offset;
    info.extent[3] = sampler.dst.offset;
}

void VulkanRaster::onEncodeFast(const Tensor* input, const Tensor* output, const VulkanCommandPool::Buffer *cmdBuffer, bool zero) {
    auto des = TensorUtils::getDescribe(input);
    mBlitImages.resize(des->regions.size());
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto dstTensor = reinterpret_cast<VulkanTensor*>(output->deviceId());
    if (zero) {
        auto fillPipeline = vkBn->getPipeline("glsl_fill_image_comp", {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        });
        struct FillImage {
            vec4 value;
            ivec4 imageSize;
        };
        FillImage uniformInfo;
        ::memset(&uniformInfo, 0, sizeof(FillImage));
        auto image = dstTensor->image();
        uniformInfo.imageSize[0] = image->width();
        uniformInfo.imageSize[1] = image->height();
        uniformInfo.imageSize[2] = 0;
        uniformInfo.imageSize[3] = image->width() * image->height();
        std::shared_ptr<VulkanBuffer> uniform(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(FillImage), &uniformInfo, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        mExtraUniform.emplace_back(uniform);
        std::shared_ptr<VulkanLayout::DescriptorSet> des(fillPipeline->createSet());
        des->writeImage(image->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        des->writeBuffer(uniform->buffer(), 1, uniform->size());
        auto totalSize = UP_DIV(uniformInfo.imageSize[3], 256);
        mExtraDescribes.emplace_back(des);
        fillPipeline->bind(cmdBuffer->get(), des->get());
        image->barrierWrite(cmdBuffer->get());
        vkCmdDispatch(cmdBuffer->get(), totalSize, 1, 1);
    }
    
    auto blitPipeline = vkBn->getPipeline("glsl_blit_image_comp", {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    });

    for (int i=0; i< des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        Tensor::InsideDescribe::Region newRegion;
        OpCommonUtils::turnToPackRegion(slice, newRegion, output, 4);
        // TODO: Find better way
        newRegion.dst.offset /= 4;
        newRegion.src.offset /= 4;
        auto& dst = mBlitImages[i];
        SamplerInfo info;
        writeSamplerInfo(info, newRegion);
        auto nhwcSrc = VulkanTensor::tensorShapeFormat(slice.origin);
        auto nhwcDst = VulkanTensor::tensorShapeFormat(output);
        info.imageSize[0] = nhwcSrc[2];
        info.imageSize[1] = nhwcSrc[1];
        info.imageSize[2] = nhwcDst[2];
        info.imageSize[3] = nhwcDst[1];
        info.depth[0] = UP_DIV(nhwcSrc[3], 4);
        info.depth[1] = UP_DIV(nhwcDst[3], 4);
        auto total = info.size[0] * info.size[1] * info.size[2];
        auto group = UP_DIV(total, 256);
        dst.describe.reset(blitPipeline->createSet());
        dst.uniform.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(SamplerInfo), &info, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        auto srcTensor = reinterpret_cast<VulkanTensor*>(slice.origin->deviceId());
        dst.describe->writeImage(srcTensor->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        dst.describe->writeImage(dstTensor->image()->view(), vkBn->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
        dst.describe->writeBuffer(dst.uniform->buffer(), 2, dst.uniform->size());
        srcTensor->image()->barrierRead(cmdBuffer->get());
        dstTensor->image()->barrierWrite(cmdBuffer->get());
        blitPipeline->bind(cmdBuffer->get(), dst.describe->get());
        vkCmdDispatch(cmdBuffer->get(), group, 1, 1);
    }
}


ErrorCode VulkanRaster::onEncode(const std::vector<Tensor *> &___inputs, const std::vector<Tensor *> &outputs,
                           const VulkanCommandPool::Buffer *cmdBuffer) {
    MNN_ASSERT(outputs.size() == 1);
    auto output = outputs[0];
    OpCommonUtils::rasterInputReset(___inputs, outputs[0]);
    auto des = TensorUtils::getDescribe(output);
    auto outputDes = TensorUtils::getDescribe(output);
    bool needZero = !TensorUtils::regionIsFull(output);

    /** Alloc Begin*/
    mInputBuffers.clear();
    mOutputBuffer.buffer = nullptr;
    mBlits.resize(des->regions.size());
    mBlitImages.clear();
    mExtraUniform.clear();
    mExtraDescribes.clear();
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        // TODO: Optimize it
        bool fast = true;
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                fast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output)) {
                fast = false;
                break;
            }
        }
        if (fast) {
            onEncodeFast(output, output, cmdBuffer, needZero);
            return NO_ERROR;
        }
    }
    auto vkBn = static_cast<VulkanBackend*>(backend());
    std::vector<VkDescriptorType> nchwConvertTypes{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    for (auto& slice : des->regions) {
        auto origin = slice.origin;
        if (mInputBuffers.find(origin)!=mInputBuffers.end()) {
            continue;
        }
        MNN_ASSERT(origin->deviceId() != 0);
        int bufferSize = sizeof(float);
        for (int i=0; i<origin->dimensions(); ++i) {
            bufferSize *= origin->length(i);
        }
        ConvertInfo info;
        info.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(),
                                           false, bufferSize, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        info.convert.reset(new VulkanImageConverter(vkBn));
        mInputBuffers.insert(std::make_pair(origin, std::move(info)));
    }
    {
        int bufferSize = sizeof(float);
        for (int i=0; i<output->dimensions(); ++i) {
            bufferSize *= output->length(i);
        }
        mOutputBuffer.convert.reset(new VulkanImageConverter(vkBn));
        mOutputBuffer.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, bufferSize, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    }
    for (auto& iter : mInputBuffers) {
        iter.second.buffer->release();
    }
    if (nullptr != mOutputBuffer.buffer) {
        mOutputBuffer.buffer->release();
    }
    auto blitPipeline = vkBn->getPipeline("glsl_blit_comp", {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    });
    for (int i=0; i<mBlits.size(); ++i) {
        auto& origin = des->regions[i];
        auto& dst = mBlits[i];
        SamplerInfo info;
        writeSamplerInfo(info, origin);
        auto total = info.size[0] * info.size[1] * info.size[2];
        dst.workGroup[2] = 1;
        dst.workGroup[1] = 1;
        dst.workGroup[0] = UP_DIV(total, 256);
        dst.pipeline = blitPipeline;
        dst.describe.reset(blitPipeline->createSet());
        dst.uniform.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, sizeof(info), &info, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        auto srcIter = mInputBuffers.find(origin.origin);
        dst.srcBuffer = srcIter->second.buffer->buffer();
        dst.srcBufferSize = srcIter->second.buffer->size();
        dst.dstBuffer = mOutputBuffer.buffer->buffer();
        dst.dstBufferSize = mOutputBuffer.buffer->size();
    }
    if (needZero) {
        mZero.dstBuffer = mOutputBuffer.buffer->buffer();
        mZero.dstBufferSize = mOutputBuffer.buffer->size();
    }
    /** Alloc End*/

    /** Encode Begin*/
    // Convert NC4HW4 image to buffer
    for (auto& iter : mInputBuffers) {
        auto& info = iter.second;
        info.convert->encodeTensorToBuffer(iter.first, info.buffer->buffer(), info.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(iter.first), cmdBuffer);
        cmdBuffer->barrierSource(info.buffer->buffer(), 0, info.buffer->size());
    }
    //Blit
    if (needZero) {
        vkCmdFillBuffer(cmdBuffer->get(), mZero.dstBuffer, 0, mZero.dstBufferSize, 0);
        cmdBuffer->barrierSource(mZero.dstBuffer, 0, mZero.dstBufferSize, VulkanCommandPool::Buffer::WRITE_WRITE);
    }
    for (auto& info : mBlits) {
        info.describe->writeBuffer(info.dstBuffer, 0, info.dstBufferSize);
        info.describe->writeBuffer(info.srcBuffer, 1, info.srcBufferSize);
        info.describe->writeBuffer(info.uniform->buffer(), 2, info.uniform->size());
        info.pipeline->bind(cmdBuffer->get(), info.describe->get());
        vkCmdDispatch(cmdBuffer->get(), info.workGroup[0], info.workGroup[1], info.workGroup[2]);
    }

    // Convert buffer to NC4HW4 image
    {
        auto& info = mOutputBuffer;
        info.convert->encodeBufferToTensor(info.buffer->buffer(), output, info.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(output), cmdBuffer);
    }
    /** Encode End*/
    return NO_ERROR;
}
class VulkanRasterCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanRaster(bn);
    }
};


class VulkanLoop : public VulkanBasicExecution {
public:
    VulkanLoop(Backend *bn, const LoopParam* loop) : VulkanBasicExecution(bn) {
        mLoop = loop;
    }
    virtual ~VulkanLoop() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) override {
        mExecutions.clear();
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        std::vector<Tensor*> tensors(mLoop->tensorNumber());
        for (int i=0; i<mLoop->inputIndexes()->size(); ++i) {
            tensors[mLoop->inputIndexes()->data()[i]] = inputs[i];
        }
        for (int i=0; i<mLoop->outputIndexes()->size(); ++i) {
            tensors[mLoop->outputIndexes()->data()[i]] = outputs[i];
        }
        auto C = tensors[cmd->indexes()->data()[0]];
        auto A = tensors[cmd->indexes()->data()[1]];
        auto B = tensors[cmd->indexes()->data()[2]];
        for (int i=0; i<mLoop->loopNumber(); ++i) {
            VulkanMatMul::MatMulInfo matInfo;
            matInfo.e = cmd->size()->data()[0];
            matInfo.l = cmd->size()->data()[1];
            matInfo.h = cmd->size()->data()[2];
            matInfo.offsetC = cmd->view()->GetAs<View>(0)->offset() + i * cmd->steps()->data()[0];
            matInfo.offsetA = cmd->view()->GetAs<View>(1)->offset() + i * cmd->steps()->data()[1];
            matInfo.offsetB = cmd->view()->GetAs<View>(2)->offset() + i * cmd->steps()->data()[2];
            ::memcpy(matInfo.aStride, cmd->view()->GetAs<View>(1)->stride()->data(), 3 * sizeof(int));
            ::memcpy(matInfo.bStride, cmd->view()->GetAs<View>(2)->stride()->data(), 3 * sizeof(int));
            ::memcpy(matInfo.cStride, cmd->view()->GetAs<View>(0)->stride()->data(), 3 * sizeof(int));
            Tensor* bias = nullptr;
            if (cmd->indexes()->size() > 3) {
                bias = tensors[cmd->indexes()->data()[3]];
                matInfo.offsetBias = cmd->view()->GetAs<View>(3)->offset() + i * cmd->steps()->data()[3];
            }
            auto matmulOp = cmd->op();
            std::shared_ptr<VulkanBasicExecution> exe(new VulkanMatMul(matmulOp->main_as_MatMul()->transposeA(), matmulOp->main_as_MatMul()->transposeB(), backend()));
            auto matmulExe = static_cast<VulkanMatMul*>(exe.get());
            bool res = true;
            if (bias == nullptr) {
                res = matmulExe->encode({{A, B}}, {C}, cmdBuffer, matInfo);
            } else {
                res = matmulExe->encode({{A, B, bias}}, {C}, cmdBuffer, matInfo);
            }
            if (!res) {
                return NOT_SUPPORT;
            }
            mExecutions.emplace_back(exe);
        }
        return NO_ERROR;
    }
private:
    std::vector<std::shared_ptr<VulkanBasicExecution>> mExecutions;
    const LoopParam* mLoop;
};

class VulkanLoopCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        auto loop = op->main_as_LoopParam();
        if (1 != loop->commands()->size()) {
            return nullptr;
        }
        auto cmd = loop->commands()->GetAs<RegionCommand>(0);
        if (OpType_MatMul != cmd->op()->type()) {
            return nullptr;
        }
        return new VulkanLoop(bn, loop);
    }
};


static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Raster, new VulkanRasterCreator);
//    VulkanBackend::addCreator(OpType_While, new VulkanLoopCreator);
    return true;
}();

};
