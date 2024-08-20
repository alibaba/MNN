#include "VulkanRaster.hpp"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include "core/OpCommonUtils.hpp"
#include "core/Macro.h"

namespace MNN {
struct NCHWInfo {
    ivec4 size; // NCHW
    ivec4 stride; // NCHW
};
static void writeNCHW(NCHWInfo& dims, Tensor* origin) {
    MNN_ASSERT(origin->dimensions() >= 2);
    int w     = 1;
    int h     = 1;
    int b     = origin->length(0);
    int c = origin->length(1);
    if (origin->dimensions() >= 3) {
        h = origin->length(2);
    }
    for (int i=3; i<origin->dimensions(); ++i) {
        w *= origin->length(i);
    }
    dims.size[0]   = b;
    dims.size[1]   = c;
    dims.size[2]   = h;
    dims.size[3]   = w;
    dims.stride[0] = c * h * w;
    dims.stride[1] = h * w;
    dims.stride[2] = w;
    dims.stride[3] = 1;
}
struct SamplerInfo {
    ivec4 stride;//stride[3] + offset
    ivec4 size;//size[3] + totalSize
    ivec4 extent;//dstStride[3]+dstOffset
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
VulkanRaster::~VulkanRaster() {
    _recycle();
}
void VulkanRaster::_recycle() {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    for (auto& uniform : mExtraUniform) {
        ((VulkanRuntime*)(vkBn->getRuntime()))->recycleUniform(uniform);
    }
    mExtraUniform.clear();
    mExtraDescribes.clear();
    mOutputBuffer.first = MemChunk();
    mInputBuffers.clear();
}

void VulkanRaster::onEncodeFast(const Tensor* input, const Tensor* output, const VulkanCommandPool::Buffer *cmdBuffer, bool zero) {
    auto des = TensorUtils::getDescribe(input);
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto dstTensor = vkBn->getTensorBuffer(output);
    auto dstTensorSize = vkBn->getTensorSize(output);
    if (zero) {
        vkCmdFillBuffer(cmdBuffer->get(), dstTensor.first->buffer(), dstTensor.second, dstTensorSize, 0);
        cmdBuffer->barrierSource(dstTensor.first->buffer(), dstTensor.second, dstTensorSize, VulkanCommandPool::Buffer::WRITE_WRITE);
    }
    
    auto blitPipeline = vkBn->getPipeline("glsl_blit_C4_comp", {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    });

    for (int i=0; i< des->regions.size(); ++i) {
        auto& slice = des->regions[i];
        Tensor::InsideDescribe::Region newRegion;
        OpCommonUtils::turnToPackRegion(slice, newRegion, output, 4);
        // TODO: Find better way
        newRegion.dst.offset /= 4;
        newRegion.src.offset /= 4;
        SamplerInfo info;
        writeSamplerInfo(info, newRegion);
        auto total = info.size[0] * info.size[1] * info.size[2];
        auto group = UP_DIV(total, 256);
        std::shared_ptr<VulkanLayout::DescriptorSet> describe(blitPipeline->createSet());
        std::shared_ptr<VulkanBuffer> uniform = vkBn->allocUniform();
        auto srcTensor = vkBn->getTensorBuffer(slice.origin);
        auto srcTensorSize = vkBn->getTensorSize(slice.origin);
        describe->writeBuffer(dstTensor.first->buffer(), 0, dstTensorSize, dstTensor.second);
        describe->writeBuffer(srcTensor.first->buffer(), 1, srcTensorSize, srcTensor.second);
        describe->writeBuffer(uniform->buffer(), 2, uniform->size());
        cmdBuffer->barrierSource(srcTensor.first->buffer(), srcTensor.second, srcTensorSize);
        blitPipeline->bind(cmdBuffer->get(), describe->get());
        vkCmdDispatch(cmdBuffer->get(), group, 1, 1);
        mExtraUniform.emplace_back(uniform);
        mExtraDescribes.emplace_back(describe);
    }
}


ErrorCode VulkanRaster::onEncode(const std::vector<Tensor *> &____inputs, const std::vector<Tensor *> &outputs,
                           const VulkanCommandPool::Buffer *cmdBuffer) {
    MNN_ASSERT(outputs.size() == 1);
    if (____inputs.size() > 0) {
        OpCommonUtils::rasterInputReset(____inputs, outputs[0]);
    }
    auto output = outputs[0];
    auto des = TensorUtils::getDescribe(output);
    bool needZero = !TensorUtils::regionIsFull(output);
    _recycle();
    auto vkBn = static_cast<VulkanBackend*>(backend());
    auto bufferAlloc = vkBn->getDynamicMemoryPool();
    auto vkRt = ((VulkanRuntime*)(vkBn->getRuntime()));
    if (des->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
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
        fast = false;
        if (fast) {
            onEncodeFast(output, output, cmdBuffer, needZero);
            return NO_ERROR;
        }
    }
    // Single Convert Optimize
    std::vector<VkDescriptorType> nchwConvertTypes{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    do {
        if (des->regions.size() != 1) {
            break;
        }
        OpCommonUtils::TensorConvertParameter convertParameter;
        OpCommonUtils::turnRegion2Convert(des->regions[0], output, convertParameter);
        if (convertParameter.type == 0) {
            break;
        }
        const VulkanPipeline* convertPipeline = nullptr;
        int srcIndex;
        int dstIndex;
        if (des->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            convertPipeline = vkBn->getPipeline("glsl_nchwTonc4hw4_comp", nchwConvertTypes);
            srcIndex = 0;
            dstIndex = 1;
        } else {
            convertPipeline = vkBn->getPipeline("glsl_nc4hw4Tonchw_comp", nchwConvertTypes);
            srcIndex = 1;
            dstIndex = 0;
        }
        NCHWInfo dims;
        dims.size[0] = convertParameter.batch;
        dims.size[1] = convertParameter.channel;
        dims.size[2] = 1;
        dims.size[3] = convertParameter.area;
        if (convertParameter.type == 1) {
            dims.stride[0] = convertParameter.channel * convertParameter.area;
            dims.stride[1] = convertParameter.area;
            dims.stride[2] = 0;
            dims.stride[3] = 1;
        } else {
            dims.stride[0] = convertParameter.channel * convertParameter.area;
            dims.stride[1] = 1;
            dims.stride[2] = 0;
            dims.stride[3] = convertParameter.channel;
        }
        std::shared_ptr<VulkanLayout::DescriptorSet> describe(convertPipeline->createSet());
        std::shared_ptr<VulkanBuffer> uniform = vkRt->allocUniform(&dims, sizeof(dims));
        mExtraDescribes.emplace_back(describe);
        mExtraUniform.emplace_back(uniform);
        auto inputBuffer = vkBn->getBuffer(des->regions[0].origin);
        auto outputBuffer = vkBn->getBuffer(output);
        describe->writeBuffer(outputBuffer, dstIndex);
        describe->writeBuffer(inputBuffer, srcIndex);
        describe->writeBuffer(uniform->buffer(), 2, uniform->size());
        
        convertPipeline->bind(cmdBuffer->get(), describe->get());
        auto totalSize = UP_DIV(dims.size[1], 4) * dims.size[0] * dims.size[2] * dims.size[3];
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize, 256), 1, 1);
        return NO_ERROR;
    } while(false);

    // Can't use fast mode, create temp buffer
    if (des->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        int bufferSize = sizeof(float);
        for (int i=0; i<output->dimensions(); ++i) {
            bufferSize *= output->length(i);
        }
        mOutputBuffer = std::make_pair(bufferAlloc->alloc(bufferSize), bufferSize);
        if (mOutputBuffer.first.first == nullptr) {
            return OUT_OF_MEMORY;
        }
    }
    // Input Convert
    for (auto& slice : des->regions) {
        auto origin = slice.origin;
        if (TensorUtils::getDescribe(origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
            continue;
        }
        if (mInputBuffers.find(origin)!=mInputBuffers.end()) {
            continue;
        }
        MNN_ASSERT(origin->deviceId() != 0);
        int bufferSize = sizeof(float);
        for (int i=0; i<origin->dimensions(); ++i) {
            bufferSize *= origin->length(i);
        }
        auto temp = bufferAlloc->alloc(bufferSize);
        if (temp.first == nullptr) {
            return OUT_OF_MEMORY;
        }
        mInputBuffers.insert(std::make_pair(origin, std::make_pair(temp, bufferSize)));
        NCHWInfo dims;
        writeNCHW(dims, origin);
        auto convertPipeline = vkBn->getPipeline("glsl_nc4hw4Tonchw_comp", nchwConvertTypes);
        std::shared_ptr<VulkanLayout::DescriptorSet> describe(convertPipeline->createSet());
        std::shared_ptr<VulkanBuffer> uniform = vkRt->allocUniform(&dims, sizeof(dims));
        mExtraDescribes.emplace_back(describe);
        mExtraUniform.emplace_back(uniform);
        auto originBuffer = vkBn->getTensorBuffer(origin);
        auto originSize = vkBn->getTensorSize(origin);
        describe->writeBuffer(((VulkanBuffer*)(temp.first))->buffer(), 0, bufferSize, temp.second);
        describe->writeBuffer(originBuffer.first->buffer(), 1, originSize, originBuffer.second);
        describe->writeBuffer(uniform->buffer(), 2, uniform->size());
        
        cmdBuffer->barrierSource(originBuffer.first->buffer(), originBuffer.second, originSize);
        convertPipeline->bind(cmdBuffer->get(), describe->get());
        auto totalSize = UP_DIV(dims.size[1], 4) * dims.size[0] * dims.size[2] * dims.size[3];
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize, 256), 1, 1);
        cmdBuffer->barrierSource(((VulkanBuffer*)(temp.first))->buffer(), temp.second, bufferSize);
    }
    // Blit
    auto blitPipeline = vkBn->getPipeline("glsl_blit_comp", {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    });
    std::pair<const VulkanBuffer*, size_t> dstBuffer;
    dstBuffer.first = (VulkanBuffer*)mOutputBuffer.first.first;
    dstBuffer.second = mOutputBuffer.first.second;
    if (nullptr == dstBuffer.first) {
        dstBuffer = vkBn->getTensorBuffer(output);
        mOutputBuffer.second = vkBn->getTensorSize(output);
    }
    if (needZero) {
        vkCmdFillBuffer(cmdBuffer->get(), dstBuffer.first->buffer(), dstBuffer.second, mOutputBuffer.second, 0);
        cmdBuffer->barrierSource(dstBuffer.first->buffer(), dstBuffer.second, mOutputBuffer.second, VulkanCommandPool::Buffer::WRITE_WRITE);
    }
    for (int i=0; i<des->regions.size(); ++i) {
        auto& origin = des->regions[i];
        SamplerInfo info;
        writeSamplerInfo(info, origin);
        auto total = info.size[0] * info.size[1] * info.size[2];
        std::shared_ptr<VulkanLayout::DescriptorSet> describe(blitPipeline->createSet());
        auto src = vkBn->getTensorBuffer(origin.origin);
        auto srcSize = vkBn->getTensorSize(origin.origin);
        if (TensorUtils::getDescribe(origin.origin)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            auto iter = mInputBuffers.find(origin.origin);
            src.first = (VulkanBuffer*)(iter->second.first.first);
            src.second = iter->second.first.second;
            srcSize = iter->second.second;
        }
        std::shared_ptr<VulkanBuffer> uniform = vkRt->allocUniform(&info, sizeof(info));
        mExtraUniform.emplace_back(uniform);
        mExtraDescribes.emplace_back(describe);
        describe->writeBuffer(dstBuffer.first->buffer(), 0, mOutputBuffer.second, dstBuffer.second);
        describe->writeBuffer(src.first->buffer(), 1, srcSize, src.second);
        describe->writeBuffer(uniform->buffer(), 2, uniform->size());

        blitPipeline->bind(cmdBuffer->get(), describe->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(total, 256), 1, 1);
    }

    // Convert buffer to NC4HW4 image
    if (nullptr != mOutputBuffer.first.first) {
        auto& info = mOutputBuffer;
        NCHWInfo dims;
        writeNCHW(dims, output);
        auto convertPipeline = vkBn->getPipeline("glsl_nchwTonc4hw4_comp", nchwConvertTypes);
        std::shared_ptr<VulkanLayout::DescriptorSet> describe(convertPipeline->createSet());
        std::shared_ptr<VulkanBuffer> uniform = vkRt->allocUniform(&dims, sizeof(dims));
        mExtraDescribes.emplace_back(describe);
        mExtraUniform.emplace_back(uniform);
        auto originBuffer = vkBn->getTensorBuffer(output);
        auto originSize = vkBn->getTensorSize(output);
        describe->writeBuffer(originBuffer.first->buffer(), 1, originSize, originBuffer.second);
        describe->writeBuffer(((VulkanBuffer*)(mOutputBuffer.first.first))->buffer(), 0, mOutputBuffer.second, mOutputBuffer.first.second);
        describe->writeBuffer(uniform->buffer(), 2, uniform->size());
        
        cmdBuffer->barrierSource(((VulkanBuffer*)(mOutputBuffer.first.first))->buffer(), mOutputBuffer.first.second, mOutputBuffer.second);
        convertPipeline->bind(cmdBuffer->get(), describe->get());
        auto totalSize = UP_DIV(dims.size[1], 4) * dims.size[0] * dims.size[2] * dims.size[3];
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize, 256), 1, 1);
    }
    /** Encode End*/
    for (auto& iter : mInputBuffers) {
        bufferAlloc->free(iter.second.first);
    }
    if (nullptr != mOutputBuffer.first.first) {
        bufferAlloc->free(mOutputBuffer.first);
    }
    return NO_ERROR;
}
class VulkanRasterCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        if (outputs[0]->getType().bytes() < 4) {
            return nullptr;
        }
        return new VulkanRaster(bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Raster, new VulkanRasterCreator);
    return true;
}();


VulkanRaster::Componet VulkanRaster::create(Tensor* real, Backend* bn) {
    Componet comp;
    comp.real = real;
    comp.exe.reset(new VulkanRaster(bn));
    return comp;
}

};
