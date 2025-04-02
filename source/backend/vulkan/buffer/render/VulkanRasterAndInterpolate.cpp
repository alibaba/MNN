//
//  VulkanRasterAndInterpolate.cpp
//  MNN
//
//  Created by MNN on 2023/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "VulkanBasicExecution.hpp"
#include "VulkanGaussianRender.hpp"
#include "component/VulkanTarget.hpp"
namespace MNN {
struct ConstBuffer {
    ivec4 size;
    ivec4 unit;
};
struct Float2IntBuffer {
    ivec4 size;
    vec4 unit;
};
class VulkanRasterAndInterpolate : public VulkanBasicExecution {
public:
    VulkanRasterAndInterpolate(Backend* bn);
    virtual ~ VulkanRasterAndInterpolate();
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) override;
    ErrorCode _initTarget(int width, int height);
    ErrorCode _initVertexBuffer(int size);
private:
    SharedPtr<VulkanTarget> mTarget;
    SharedPtr<VulkanGraphicPipelineCache> mGraphicCache;
    int mWidth = 0;
    int mHeight = 0;
    const VulkanPipeline* mIndiceCopyPipeline = nullptr;
    SharedPtr<VulkanLayout::DescriptorSet> mIndiceCopySet;
    std::shared_ptr<VulkanBuffer> mUniformForIndiceCopy;

    const VulkanPipeline* mComponentCopyPipeline = nullptr;
    const VulkanPipeline* mVertexCopyPipeline = nullptr;

    std::vector<std::shared_ptr<VulkanBuffer>> mUniformForExtractComponent;
    std::shared_ptr<VulkanBuffer> mIndiceInt;
    std::vector<SharedPtr<VulkanLayout::DescriptorSet>> mComponentCopySet;
    SharedPtr<VulkanPipeline> mGraphicPipeline;
    SharedPtr<VulkanLayout::DescriptorSet> mRenderSet;
    std::vector<std::shared_ptr<VulkanBuffer>> mVertexBuffers;
    int mVertexSize = 0;
    SharedPtr<VulkanLayout> mEmptyLayout;
};

VulkanRasterAndInterpolate::VulkanRasterAndInterpolate(Backend* bn) : VulkanBasicExecution(bn) {
    auto extra = static_cast<VulkanBackend*>(backend());
    {
        std::vector<VkDescriptorType> types {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        };
        mIndiceCopyPipeline = extra->getPipeline("glsl_float2int_comp", types);
        mIndiceCopySet = mIndiceCopyPipeline->createSet();
    }
    {
        std::vector<VkDescriptorType> types {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        };
        mVertexCopyPipeline = extra->getPipeline("glsl_copy_render_unit_comp", types);
    }
    {
        std::vector<VkDescriptorType> types {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        };
        mComponentCopyPipeline = extra->getPipeline("glsl_load_render_unit_comp", types);
    }
    mUniformForIndiceCopy = extra->allocUniform(nullptr, sizeof(ConstBuffer));
    VulkanGraphicPipelineCache::ShaderSource shaders;
    shaders.fragment = extra->getPipelineFactory()->createShader("glsl_render_frag_frag");
    shaders.vertex = extra->getPipelineFactory()->createShader("glsl_render_vert_vert");
    mGraphicCache = VulkanGraphicPipelineCache::create(extra->device(), shaders);
    mGraphicCache->setVertexFormats(std::vector<int>{
        4,
        4,
        4,
        4
    });
    mVertexBuffers.resize(4);
    std::vector<VulkanLayout::LayoutType> bufferTypes;

    mEmptyLayout = VulkanLayout::create(extra->device(), bufferTypes);
    mRenderSet = mEmptyLayout->createSet();
}
VulkanRasterAndInterpolate::~VulkanRasterAndInterpolate() {
    auto extra = static_cast<VulkanBackend*>(backend());
    extra->recycleUniform(mUniformForIndiceCopy);
    for (auto c : mUniformForExtractComponent) {
        extra->recycleUniform(c);
    }
    // Remove renderset before pipeline release
    mRenderSet = nullptr;
    mTarget = nullptr;
    mGraphicPipeline = nullptr;
}
ErrorCode VulkanRasterAndInterpolate::_initVertexBuffer(int size) {
    if (mVertexSize == size) {
        return NO_ERROR;
    }
    mVertexSize = size;
    auto extra = static_cast<VulkanBackend*>(backend());
    for (int i=0; i<mVertexBuffers.size(); ++i) {
        mVertexBuffers[i].reset(new VulkanBuffer(extra->getMemoryPool(), false, size * sizeof(float) * 4, nullptr,
                                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    }
    return NO_ERROR;
}

ErrorCode VulkanRasterAndInterpolate::_initTarget(int width, int height) {
    if (mWidth == width || mHeight == height) {
        return NO_ERROR;
    }
    mTarget = nullptr;
    auto extra = static_cast<VulkanBackend*>(backend());
    SharedPtr<VulkanImage> depth = new VulkanImage(extra->getMemoryPool(), false, std::vector<int>{width, height}, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    std::vector<SharedPtr<VulkanImage>> colors(4);
    for (int i=0; i<colors.size(); ++i) {
        colors[i] = new VulkanImage(extra->getMemoryPool(), false, std::vector<int>{width, height}, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    }
    mTarget = VulkanTarget::create(colors, depth);
#ifdef TEST
    auto& cmdPool = extra->getPool();
    auto buffer = cmdPool.allocBuffer();
    buffer->begin(0);
    mTarget->onEnter(buffer->get());
    mTarget->onExit(buffer->get());
    buffer->end();
    cmdPool.submitAndWait(buffer->get());
    delete buffer;
#endif
    auto& info = mGraphicCache->info();
    mTarget->writePipelineInfo(info);
    mGraphicPipeline = extra->getPipelineFactory()->createGraphicPipeline(mEmptyLayout, mGraphicCache.get());
    mWidth = width;
    mHeight = height;
    return NO_ERROR;
}

ErrorCode VulkanRasterAndInterpolate::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                               const VulkanCommandPool::Buffer *cmdBuffer) {
    auto extra = static_cast<VulkanBackend*>(backend());
    auto width = outputs[0]->length(2);
    auto heigth = outputs[0]->length(1);
    // Load Result from Image to Buffer
    for (auto c : mUniformForExtractComponent) {
        extra->recycleUniform(c);
    }
    mUniformForExtractComponent.clear();
    mComponentCopySet.clear();
    auto code = _initTarget(width, heigth);
    if (code != NO_ERROR) {
        return code;
    }
    auto rasterOutput = outputs[0];
    auto indice = inputs[1];
    auto position = inputs[2];
    int vertexSize = position->length(position->dimensions() - 2);
    code = _initVertexBuffer(vertexSize);
    if (code != NO_ERROR) {
        return code;
    }
    // Copy Indice Buffer from float to int
    auto indiceSize = indice->elementSize();
    {
        // Init indice int buffer for copy
        mIndiceInt.reset(new VulkanBuffer(extra->getMemoryPool(), false, indiceSize * sizeof(int), nullptr,
                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
        auto param = reinterpret_cast<Float2IntBuffer*>(mUniformForIndiceCopy->map());
        param->size[0] = indiceSize;
        param->size[1] = 1;
        param->size[2] = 1;
        param->size[3] = 1;
        param->unit[0] = 1.0f;
        param->unit[1] = 0.0f;
        param->unit[2] = 0.0f;
        param->unit[3] = 0.0f;
        mUniformForIndiceCopy->unmap();
        auto indiceBuffer = extra->getBuffer(indice);
        mIndiceCopySet->writeBuffer(mIndiceInt->buffer(), 0, mIndiceInt->size());
        mIndiceCopySet->writeBuffer(indiceBuffer, 1);
        mIndiceCopySet->writeBuffer(mUniformForIndiceCopy->buffer(), 2, mUniformForIndiceCopy->size());
        mIndiceCopyPipeline->bind(cmdBuffer->get(), mIndiceCopySet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(indiceSize, 256), 1, 1);
        VkBufferMemoryBarrier barrier;
        barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.buffer              = mIndiceInt->buffer();
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.offset              = 0;
        barrier.pNext               = nullptr;
        barrier.size                = indice->size();
        barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer->get(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 1,
                             &barrier, 0, nullptr);
    }
    auto vertexCopyFromInput = [&](int i, Tensor* input) {
        SharedPtr<VulkanLayout::DescriptorSet> copyVertexLay = mVertexCopyPipeline->createSet();
        mComponentCopySet.emplace_back(copyVertexLay);
        auto copyVertexUnifom = extra->allocUniform();
        auto ptr = (ConstBuffer*)copyVertexUnifom->map();
        ptr->size[0] = input->length(input->dimensions()-1);
        ptr->size[1] = input->length(input->dimensions()-1);
        ptr->size[2] = 0;
        ptr->size[3] = vertexSize;
        copyVertexUnifom->unmap();
        mUniformForExtractComponent.emplace_back(copyVertexUnifom);
        copyVertexLay->writeBuffer(mVertexBuffers[i]->buffer(), 0, mVertexBuffers[i]->size());
        copyVertexLay->writeBuffer(extra->getBuffer(input), 1);
        copyVertexLay->writeBuffer(copyVertexUnifom->buffer(), 2, sizeof(ConstBuffer));
        mVertexCopyPipeline->bind(cmdBuffer->get(), copyVertexLay->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(vertexSize, 256), 1, 1);
        VkBufferMemoryBarrier barrier;
        barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.buffer              = mVertexBuffers[i]->buffer();
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.offset              = 0;
        barrier.pNext               = nullptr;
        barrier.size                = mVertexBuffers[i]->size();
        barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer->get(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 1,
                             &barrier, 0, nullptr);
    };
    auto barrierVertex = [&](int i) {
        VkBufferMemoryBarrier barrier;
        barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.buffer              = mVertexBuffers[i]->buffer();
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.offset              = 0;
        barrier.pNext               = nullptr;
        barrier.size                = mVertexBuffers[i]->size();
        barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer->get(), VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 0, nullptr, 1,
                             &barrier, 0, nullptr);
    };
    vertexCopyFromInput(0, position);
    barrierVertex(0);
    
    auto copyImageDest = [&](int index, Tensor* output) {
        SharedPtr<VulkanLayout::DescriptorSet> copyVertexLay = mComponentCopyPipeline->createSet();
        mComponentCopySet.emplace_back(copyVertexLay);
        auto copyVertexUnifom = extra->allocUniform();
        mUniformForExtractComponent.emplace_back(copyVertexUnifom);
        int unitSize = output->length(output->dimensions()-1);
        auto ptr = (ConstBuffer*)copyVertexUnifom->map();
        ptr->size[0] = width;
        ptr->size[1] = heigth;
        ptr->size[2] = 0;
        ptr->size[3] = 0;
        ptr->unit[0] = unitSize;
        ptr->unit[1] = unitSize;
        ptr->unit[2] = 0;
        ptr->unit[3] = unitSize;
        copyVertexUnifom->unmap();
        copyVertexLay->writeBuffer(extra->getBuffer(output), 0);
        auto image = mTarget->content().colors[index];
        copyVertexLay->writeImage(image->view(), extra->getCommonSampler()->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        copyVertexLay->writeBuffer(copyVertexUnifom->buffer(), 2, sizeof(ConstBuffer));
        mComponentCopyPipeline->bind(cmdBuffer->get(), copyVertexLay->get());
        VulkanImage::insertMemoryBarrier(
            cmdBuffer->get(),
            image->get(),
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(width, 16), UP_DIV(heigth, 16), 1);
        VulkanImage::insertMemoryBarrier(
            cmdBuffer->get(),
            image->get(),
            VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
            VkImageSubresourceRange{
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
            });
    };
    int renderTime = UP_DIV((outputs.size() - 1), 3);
    for (int time = 0; time < renderTime; ++time) {
        // Copy Vertex Data
        int sta = time * 3;
        int end = ALIMIN(outputs.size()-1, sta + 3);
        int con = end - sta;
        for (int si=0; si<con; ++si) {
            auto i = si + 1;
            auto input = inputs[sta+i+2];
            vertexCopyFromInput(i, input);
        }
        
        // Render
        mTarget->onEnter(cmdBuffer->get());
        mGraphicPipeline->bind(cmdBuffer->get(), mRenderSet->get());
        for (int i=0; i<mVertexBuffers.size(); ++i) {
            VkDeviceSize offset = 0;
            VkBuffer buffer = mVertexBuffers[i]->buffer();
            vkCmdBindVertexBuffers(cmdBuffer->get(), i, 1, &buffer, &offset);
        }
        vkCmdBindIndexBuffer(cmdBuffer->get(), mIndiceInt->buffer(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmdBuffer->get(), indiceSize, 1, 0, 0, 0);
        mTarget->onExit(cmdBuffer->get());

        for (int i=0; i<con; ++i) {
            barrierVertex(i+1);
            copyImageDest(i+1, outputs[1+i+sta]);
        }
        auto& content = mTarget->content();
        if (renderTime - 1 == time) {
            copyImageDest(0, outputs[0]);
        }
    }
    // Revert Index Barrier
    {
        VkBufferMemoryBarrier barrier;
        barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.buffer              = mIndiceInt->buffer();
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.offset              = 0;
        barrier.pNext               = nullptr;
        barrier.size                = indice->size();
        barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcAccessMask       = VK_ACCESS_INDEX_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer->get(), VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                             , 0, 0, nullptr, 1,
                             &barrier, 0, nullptr);

    }
    return NO_ERROR;
}


class VulkanRasterAndInterpolateCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        bool hasIndice = true;
        int type = 4;
        if (op->main_type() == OpParameter_Extra) {
            auto extra = op->main_as_Extra();
            if (nullptr != extra->attr()) {
                for (int i=0; i<extra->attr()->size(); ++i) {
                    auto attr = extra->attr()->GetAs<Attribute>(i);
                    if (attr->key()->str() == "index") {
                        hasIndice = attr->b();
                        continue;
                    }
                    if (attr->key()->str() == "primitiveType") {
                        type = attr->i();
                        continue;
                    }
                }
            }
        }
        if (6 == type) {
            return new VulkanRasterSort(backend);
        }
        if (2 != type) {
            return nullptr;
        }
        return new VulkanRasterAndInterpolate(backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_RasterAndInterpolate, new VulkanRasterAndInterpolateCreator);
    return true;
}();

} // namespace MNN

