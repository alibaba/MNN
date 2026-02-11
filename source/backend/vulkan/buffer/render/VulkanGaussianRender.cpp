#include <cmath>
#include <limits.h>
#include "VulkanGaussianRender.hpp"
namespace MNN {
struct ImageConstant {
    ivec4 point;
    ivec4 size;
    ivec4 block;
};

struct VulkanRasterSort::Content {
    const VulkanPipeline* cumsum;
    const VulkanPipeline* rastersort_count_valid_number;
    const VulkanPipeline* rastersort_collect_key;
    std::vector<SharedPtr<VulkanLayout::DescriptorSet>> layouts;
    std::vector<std::shared_ptr<VulkanBuffer>> uniforms;
    VulkanBackend* extra;
    void reset() {
        for (auto u : uniforms) {
            extra->recycleUniform(u);
        }
        uniforms.clear();
        layouts.clear();
    }
    Content(VulkanBackend* vkBn) {
        extra = vkBn;
    }
    ~ Content() {
        reset();
    }
};
VulkanRasterSort::VulkanRasterSort(Backend* bn) : VulkanBasicExecution(bn) {
    mContent = new Content(static_cast<VulkanBackend*>(bn));
    auto extra = static_cast<VulkanBackend*>(bn);
    mLocalSize = std::min(mLocalSize, extra->device().getMaxComputeWorkGroupInvocations());
    {
        std::vector<uint32_t> spc = {(uint32_t)256, 1, 1};
        std::vector<VkDescriptorType> types {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        };
        if (extra->device().getLocalMemorySize() > 0) {
            mContent->cumsum = extra->getPipelineFactory()->getPipeline("glsl_cumsum_comp", types, spc);
        } else {
            mContent->cumsum = extra->getPipelineFactory()->getPipeline("glsl_cumsum_single_comp", types, spc);
        }
    }
    mRadixSort.reset(new VulkanRadixSort(bn, 16));
}
VulkanRasterSort:: ~VulkanRasterSort() {
    delete mContent;
}
ErrorCode VulkanRasterSort::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                       const VulkanCommandPool::Buffer *cmdBuffer) {
    auto attr = inputs[0];
    auto numAttr = attr->length(1);
    auto extra = static_cast<VulkanBackend*>(backend());
    {
        std::vector<uint32_t> spc = {(uint32_t)mLocalSize, 1, 1, (uint32_t)mLocalSize};
        std::vector<VkDescriptorType> types {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        };
        if (4 == numAttr) {
            mContent->rastersort_count_valid_number = extra->getPipelineFactory()->getPipeline("glsl_rastersort_count_valid_number_comp", types, spc);
        } else {
            mContent->rastersort_count_valid_number = extra->getPipelineFactory()->getPipeline("glsl_rastersort_count_valid_number_USE_HALF_comp", types, spc);
        }
    }
    {
        std::vector<uint32_t> spc = {(uint32_t)mLocalSize, 1, 1, (uint32_t)mLocalSize};
        std::vector<VkDescriptorType> types {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
        if (4 == numAttr) {
            mContent->rastersort_collect_key = extra->getPipelineFactory()->getPipeline("glsl_rastersort_collect_key_comp", types, spc);
        } else {
            mContent->rastersort_collect_key = extra->getPipelineFactory()->getPipeline("glsl_rastersort_collect_key_USE_HALF_comp", types, spc);
        }
    }
    mContent->reset();
    if(extra->isSupportAutotune()){
        autoTune(inputs, outputs);
    }
    auto memPool = extra->getDynamicMemoryPool();
    auto viewProj = inputs[1];
    auto numberPoint = attr->length(0);
    auto pointOffsetBytes = mLocalSize * mGroupSize * sizeof(uint32_t);
    auto pointOffsets = memPool->alloc(pointOffsetBytes);
    std::shared_ptr<VulkanBuffer> imageConstant = extra->allocUniform(nullptr, sizeof(ImageConstant));
    {
        auto ptr = (ImageConstant*)imageConstant->map();
        ptr->point[0] = numberPoint;
        //ptr->point[1] = attr->length(1) / 4;
        imageConstant->unmap();
    }
    mContent->uniforms.emplace_back(imageConstant);
    // Compute valid pointNumber
    {
        SharedPtr<VulkanLayout::DescriptorSet> des = mContent->rastersort_count_valid_number->createSet();
        des->writeBuffer(((VulkanBuffer*)pointOffsets.first)->buffer(), 0, pointOffsetBytes, pointOffsets.second);
        des->writeBuffer(extra->getBuffer(attr), 1);
        des->writeBuffer(extra->getBuffer(viewProj), 2);
        des->writeBuffer(imageConstant->buffer(), 3, imageConstant->size());
        mContent->layouts.emplace_back(des);
        mContent->rastersort_count_valid_number->bind(cmdBuffer->get(), des->get());
        vkCmdDispatch(cmdBuffer->get(), mGroupSize, 1, 1);
        cmdBuffer->barrierSource(((VulkanBuffer*)pointOffsets.first)->buffer(), pointOffsets.second, pointOffsetBytes);
    }

    // Compute cusum of point offset
    auto pointOffsetSum = memPool->alloc(pointOffsetBytes);
    {
        SharedPtr<VulkanLayout::DescriptorSet> des = mContent->cumsum->createSet();
        des->writeBuffer(((VulkanBuffer*)pointOffsetSum.first)->buffer(), 0, pointOffsetBytes, pointOffsetSum.second);
        des->writeBuffer(((VulkanBuffer*)pointOffsets.first)->buffer(), 1, pointOffsetBytes, pointOffsets.second);
        auto cumSumNumber = extra->allocUniform();
        ((int*)cumSumNumber->map())[0] = pointOffsetBytes / sizeof(uint32_t);
        cumSumNumber->unmap();
        mContent->uniforms.emplace_back(cumSumNumber);

        des->writeBuffer(cumSumNumber->buffer(), 2, cumSumNumber->size());
        mContent->cumsum->bind(cmdBuffer->get(), des->get());
        vkCmdDispatch(cmdBuffer->get(), 1, 1, 1);
        mContent->layouts.emplace_back(des);
        cmdBuffer->barrierSource(((VulkanBuffer*)pointOffsetSum.first)->buffer(), pointOffsetSum.second, pointOffsetBytes);
    }
    memPool->free(pointOffsets);
    auto sortNumber = extra->allocUniform();
    mContent->uniforms.emplace_back(sortNumber);
    {
        // Copy pointOffsetSum's lastnumber to sortNumber
        VkBufferCopy region;
        region.size = sizeof(uint32_t);
        region.dstOffset = 0;
        region.srcOffset = pointOffsetSum.second + (pointOffsetBytes / sizeof(uint32_t) - 1) * sizeof(uint32_t);
        vkCmdCopyBuffer(cmdBuffer->get(), ((VulkanBuffer*)pointOffsetSum.first)->buffer(), sortNumber->buffer(), 1, &region);
        
        auto output = extra->getBuffer(outputs[0]);
        VkBufferCopy region2;
        region2.size = sizeof(uint32_t);
        region2.dstOffset = std::get<2>(output);
        region2.srcOffset = pointOffsetSum.second + (pointOffsetBytes / sizeof(uint32_t) - 1) * sizeof(uint32_t);

        vkCmdCopyBuffer(cmdBuffer->get(), ((VulkanBuffer*)pointOffsetSum.first)->buffer(), std::get<0>(output), 1, &region2);

        cmdBuffer->barrierSource(sortNumber->buffer(), 0, sizeof(uint32_t));
    }

    // Collect pointKeys
    auto keySize = UP_DIV(numberPoint, 2) * 2 * sizeof(uint32_t) * 2;
    auto outputBuffer = extra->getTensorBuffer(outputs[1]);
    {
        SharedPtr<VulkanLayout::DescriptorSet> des = mContent->rastersort_collect_key->createSet();
        des->writeBuffer(((VulkanBuffer*)outputBuffer.first)->buffer(), 0, keySize, outputBuffer.second);
        des->writeBuffer(extra->getBuffer(attr), 1);
        des->writeBuffer(extra->getBuffer(viewProj), 2);
        des->writeBuffer(((VulkanBuffer*)pointOffsetSum.first)->buffer(), 3, pointOffsetBytes, pointOffsetSum.second);
        des->writeBuffer(imageConstant->buffer(), 4, imageConstant->size());
        mContent->rastersort_collect_key->bind(cmdBuffer->get(), des->get());
        vkCmdDispatch(cmdBuffer->get(), mGroupSize, 1, 1);
        mContent->layouts.emplace_back(des);
        cmdBuffer->barrierSource(((VulkanBuffer*)outputBuffer.first)->buffer(), outputBuffer.second, keySize);
    }
    memPool->free(pointOffsetSum);
    // Radix Sort
    auto pointKeysMid = memPool->alloc(keySize);
    std::pair<VulkanBuffer*, VkDeviceSize> srcIndex;
    std::pair<VulkanBuffer*, VkDeviceSize> dstIndex;
    {
        srcIndex.first = const_cast<VulkanBuffer*>(outputBuffer.first);
        srcIndex.second = outputBuffer.second;
        dstIndex.first = static_cast<VulkanBuffer*>(pointKeysMid.first);
        dstIndex.second = pointKeysMid.second;
    }
    mRadixSort->onExcute(srcIndex, dstIndex, cmdBuffer, numberPoint, sortNumber);
    memPool->free(MemChunk(std::make_pair(dstIndex.first, dstIndex.second)));
    return NO_ERROR;
}

void VulkanRasterSort::autoTune(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    // Tune Raster Sort
    auto extra = static_cast<VulkanBackend*>(backend());
    auto memPool = extra->getDynamicMemoryPool();
    auto attr = inputs[0];
    auto viewProj = inputs[1];
    auto numberPoint = attr->length(0);
    std::shared_ptr<VulkanBuffer> imageConstant = extra->allocUniform(nullptr, sizeof(ImageConstant));
    {
        auto ptr = (ImageConstant*)imageConstant->map();
        ptr->point[0] = numberPoint;
        imageConstant->unmap();
    }
    int maxLocalSize = extra->device().getMaxComputeWorkGroupInvocations();
    uint32_t min_cost = UINT_MAX;
    
    size_t maxHistogramSize = maxLocalSize * mGroupSize * sizeof(uint32_t);
    std::shared_ptr<Tensor> histogram;
    std::shared_ptr<Tensor> histogramSum;
    histogram.reset(Tensor::createDevice<uint32_t>({static_cast<int>(maxHistogramSize)}));
    histogramSum.reset(Tensor::createDevice<uint32_t>({static_cast<int>(maxHistogramSize)}));
    auto res = extra->onAcquireBuffer(histogram.get(), Backend::STATIC);
    if (!res) {
        return;
    }
    res = extra->onAcquireBuffer(histogramSum.get(), Backend::STATIC);
    if (!res) {
        return;
    }
    
    for(int g = 8; g <= 1024; g *= 2){
        for(int l = 8; l <= maxLocalSize; l *= 2){
            size_t histogramSize = l * g * sizeof(uint32_t);
            uint32_t time = 0;
            // Compute valid pointNumber
            {
                std::vector<int> gps = {g, 1, 1};
                std::vector<uint32_t> lws = {(uint32_t)l, 1, 1,(uint32_t)l};
                mContent->rastersort_count_valid_number->changePipeline(lws);
                SharedPtr<VulkanLayout::DescriptorSet> des = mContent->rastersort_count_valid_number->createSet();
                des->writeBuffer(extra->getTensorBuffer(histogram.get()).first->buffer(), 0, histogramSize, extra->getTensorBuffer(histogram.get()).second);
                des->writeBuffer(extra->getBuffer(attr), 1);
                des->writeBuffer(extra->getBuffer(viewProj), 2);
                des->writeBuffer(imageConstant->buffer(), 3, imageConstant->size());
                time += (uint32_t)extra->getPipelineTime(mContent->rastersort_count_valid_number, des, gps);
            }
            
            // Compute cusum of point offset
            {
                std::vector<int> gps = {1, 1, 1};
                SharedPtr<VulkanLayout::DescriptorSet> des = mContent->cumsum->createSet();
                des->writeBuffer(extra->getTensorBuffer(histogramSum.get()).first->buffer(), 0, histogramSize, extra->getTensorBuffer(histogramSum.get()).second);
                des->writeBuffer(extra->getTensorBuffer(histogram.get()).first->buffer(), 1, histogramSize, extra->getTensorBuffer(histogram.get()).second);
                auto cumSumNumber = extra->allocUniform();
                ((int*)cumSumNumber->map())[0] = histogramSize / sizeof(uint32_t);
                cumSumNumber->unmap();
                des->writeBuffer(cumSumNumber->buffer(), 2, cumSumNumber->size());
                time += (uint32_t)extra->getPipelineTime(mContent->cumsum, des, gps);
            }
            
            // Collect pointKeys
            auto keySize = UP_DIV(numberPoint, 2) * 2 * sizeof(uint32_t) * 2;
            auto outputBuffer = extra->getTensorBuffer(outputs[1]);
            {
                std::vector<int> gps = {g, 1, 1};
                std::vector<uint32_t> lws = {(uint32_t)l, 1, 1,(uint32_t)l};
                mContent->rastersort_collect_key->changePipeline(lws);
                SharedPtr<VulkanLayout::DescriptorSet> des = mContent->rastersort_collect_key->createSet();
                des->writeBuffer(((VulkanBuffer*)outputBuffer.first)->buffer(), 0, keySize, outputBuffer.second);
                des->writeBuffer(extra->getBuffer(attr), 1);
                des->writeBuffer(extra->getBuffer(viewProj), 2);
                des->writeBuffer(extra->getTensorBuffer(histogramSum.get()).first->buffer(), 3, histogramSize, extra->getTensorBuffer(histogramSum.get()).second);
                des->writeBuffer(imageConstant->buffer(), 4, imageConstant->size());
                time += (uint32_t)extra->getPipelineTime(mContent->rastersort_collect_key, des, gps);
            }
            if(time < min_cost){
                min_cost = time;
                mLocalSize = l;
                mGroupSize = g;
            }
        }
    }
    
    std::vector<int> gps = {mGroupSize, 1, 1};
    std::vector<uint32_t> lws = {(uint32_t)mLocalSize, 1, 1, (uint32_t)mLocalSize};
    mContent->rastersort_count_valid_number->changePipeline(lws);
    mContent->rastersort_collect_key->changePipeline(lws);
}

struct VulkanRadixSort::Content {
    const VulkanPipeline* cumsum;
    const VulkanPipeline* radixsort_histogram;
    const VulkanPipeline* radixsort_reorder;
    std::vector<SharedPtr<VulkanLayout::DescriptorSet>> layouts;
    std::vector<std::shared_ptr<VulkanBuffer>> uniforms;
    VulkanBackend* extra;
    void reset() {
        for (auto u : uniforms) {
            extra->recycleUniform(u);
        }
        uniforms.clear();
        layouts.clear();
    }
    Content(VulkanBackend* vkBn) {
        extra = vkBn;
    }
    ~ Content() {
        reset();
    }
};
VulkanRadixSort::VulkanRadixSort(Backend* bn, int needBit) : mBackend(bn), mNeedBits(needBit) {
    mContent = new Content(static_cast<VulkanBackend*>(bn));
    auto extra = static_cast<VulkanBackend*>(bn);
    {
        std::vector<uint32_t> spc = {(uint32_t)mLocalSize, 1, 1, (uint32_t)128, (uint32_t)mLocalSize};
        std::vector<VkDescriptorType> types {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        };
        if (extra->device().getLocalMemorySize() > 0) {
            mContent->cumsum = extra->getPipelineFactory()->getPipeline("glsl_cumsum_comp", types, spc, {}, true);
        } else {
            mContent->cumsum = extra->getPipelineFactory()->getPipeline("glsl_cumsum_single_comp", types, spc, {}, true);
        }
    }
    {
        std::vector<uint32_t> spc = {(uint32_t)mLocalSize, 1, 1, (uint32_t)(1<<mPerSortBit), (uint32_t)mLocalSize};
        std::vector<VkDescriptorType> types {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        };
        mContent->radixsort_reorder = extra->getPipelineFactory()->getPipeline("glsl_radixsort_reorder_comp", types, spc, {}, true);
    }
    {
        std::vector<uint32_t> spc = {(uint32_t)mLocalSize, 1, 1, (uint32_t)(1<<mPerSortBit), (uint32_t)mLocalSize};
        std::vector<VkDescriptorType> types {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        };
        mContent->radixsort_histogram = extra->getPipelineFactory()->getPipeline("glsl_radixsort_histogram_comp", types, spc, {}, true);
    }
}
VulkanRadixSort:: ~VulkanRadixSort() {
    delete mContent;
}
ErrorCode VulkanRadixSort::onExcute(std::pair<VulkanBuffer*, VkDeviceSize> srcIndex, std::pair<VulkanBuffer*, VkDeviceSize> dstIndex, const VulkanCommandPool::Buffer *cmdBuffer,
                                     int numberPoint, std::shared_ptr<VulkanBuffer> sortNumber) {
    mContent->reset();
    auto extra = static_cast<VulkanBackend*>(mBackend);
    auto memPool = extra->getDynamicMemoryPool();
    auto keySize = UP_DIV(numberPoint, 2) * 2 * sizeof(uint32_t) * 2;
    if(extra->isSupportAutotune()){
        autoTune(srcIndex, dstIndex, numberPoint, sortNumber);
    }
    const int binSize = (1<<mPerSortBit);
    // Radix Sort
    size_t histogramSize = binSize * mLocalSize * mGroupSize * sizeof(uint32_t);
    auto historyCumSumSize = extra->allocUniform();
    {
        auto ptr = (uint32_t*)historyCumSumSize->map();
        ptr[0] = binSize * mLocalSize * mGroupSize;
        historyCumSumSize->unmap();
        mContent->uniforms.emplace_back(historyCumSumSize);
    }
    auto histogram = memPool->alloc(histogramSize);
    auto histogramSum = memPool->alloc(histogramSize);
    int numerPass = UP_DIV(mNeedBits, mPerSortBit);
    for (int i=0; i<numerPass; ++i) {
        auto pass = extra->allocUniform();
        auto ptr = (uint32_t*)pass->map();
        ptr[0] = i * mPerSortBit;
        pass->unmap();
        mContent->uniforms.emplace_back(pass);
        // compute histogram
        {
            SharedPtr<VulkanLayout::DescriptorSet> des = mContent->radixsort_histogram->createSet();
            mContent->layouts.emplace_back(des);
            des->writeBuffer(static_cast<VulkanBuffer*>(histogram.first)->buffer(), 0, histogramSize, histogram.second);
            des->writeBuffer(srcIndex.first->buffer(), 1, keySize,  srcIndex.second);
            des->writeBuffer(sortNumber->buffer(), 2, sortNumber->size());
            des->writeBuffer(pass->buffer(), 3, 4 * sizeof(uint32_t));
            mContent->radixsort_histogram->bind(cmdBuffer->get(), des->get());
            vkCmdDispatch(cmdBuffer->get(), mGroupSize, 1, 1);
            cmdBuffer->barrierSource(static_cast<VulkanBuffer*>(histogram.first)->buffer(), histogram.second, histogramSize);
        }
        // cumsum histogram
        {
            SharedPtr<VulkanLayout::DescriptorSet> des = mContent->cumsum->createSet();
            des->writeBuffer(((VulkanBuffer*)histogramSum.first)->buffer(), 0, histogramSize, histogramSum.second);
            des->writeBuffer(((VulkanBuffer*)histogram.first)->buffer(), 1, histogramSize, histogram.second);
            des->writeBuffer(historyCumSumSize->buffer(), 2, historyCumSumSize->size());
            mContent->cumsum->bind(cmdBuffer->get(), des->get());
            vkCmdDispatch(cmdBuffer->get(), 1, 1, 1);
            mContent->layouts.emplace_back(des);
            cmdBuffer->barrierSource(((VulkanBuffer*)histogramSum.first)->buffer(), histogramSum.second, histogramSize);
        }
        // reorder
        {
            SharedPtr<VulkanLayout::DescriptorSet> des = mContent->radixsort_reorder->createSet();
            mContent->layouts.emplace_back(des);
            des->writeBuffer(dstIndex.first->buffer(), 0, keySize, dstIndex.second);
            des->writeBuffer(srcIndex.first->buffer(), 1, keySize, srcIndex.second);
            des->writeBuffer(((VulkanBuffer*)histogramSum.first)->buffer(), 2, histogramSize, histogramSum.second);
            des->writeBuffer(sortNumber->buffer(), 3, sortNumber->size());
            des->writeBuffer(pass->buffer(), 4, pass->size());
            mContent->radixsort_reorder->bind(cmdBuffer->get(), des->get());
            vkCmdDispatch(cmdBuffer->get(), mGroupSize, 1, 1);
            cmdBuffer->barrierSource(dstIndex.first->buffer(), dstIndex.second, keySize);
            cmdBuffer->barrierSource(((VulkanBuffer*)histogramSum.first)->buffer(), histogramSum.second, histogramSize, VulkanCommandPool::Buffer::WRITE_READ);
        }
        // Swap dst/src
        auto temp = srcIndex;
        srcIndex = dstIndex;
        dstIndex = temp;
    }
    memPool->free(histogram);
    memPool->free(histogramSum);
    cmdBuffer->barrierSource(srcIndex.first->buffer(), srcIndex.second, keySize);

    return NO_ERROR;
}

void VulkanRadixSort::autoTune(std::pair<VulkanBuffer*, VkDeviceSize> srcIndex, std::pair<VulkanBuffer*, VkDeviceSize> dstIndex, int numberPoint, std::shared_ptr<VulkanBuffer> sortNumber){
    // Tune Radix Sort
    auto extra = static_cast<VulkanBackend*>(mBackend);
    auto memPool = extra->getDynamicMemoryPool();
    auto keySize = UP_DIV(numberPoint, 2) * 2 * sizeof(uint32_t) * 2;
    std::shared_ptr<VulkanBuffer> pass = extra->allocUniform();
    auto ptr = (uint32_t*)pass->map();
    int maxLocalSize = extra->device().getMaxComputeWorkGroupInvocations();
    ptr[0] = 0;
    pass->unmap();
    uint32_t min_cost = UINT_MAX;
    
    size_t maxHistogramSize = 1024 * 256 * 16 * sizeof(uint32_t);
    std::shared_ptr<Tensor> histogram;
    std::shared_ptr<Tensor> histogramSum;
    histogram.reset(Tensor::createDevice<uint32_t>({static_cast<int>(maxHistogramSize)}));
    histogramSum.reset(Tensor::createDevice<uint32_t>({static_cast<int>(maxHistogramSize)}));
    auto res = extra->onAcquireBuffer(histogram.get(), Backend::STATIC);
    if (!res) {
        return;
    }
    res = extra->onAcquireBuffer(histogramSum.get(), Backend::STATIC);
    if (!res) {
        return;
    }
    //std::shared_ptr<VulkanBuffer> tmpMem = std::make_shared<VulkanBuffer>(extra->getMemoryPool(), false, sizeof(float) * totalWeightSize, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    if (extra->device().getLocalMemorySize() > 0) {
        size_t histogramSize = 256 * 256 * 8 * sizeof(uint32_t);
        std::shared_ptr<VulkanBuffer> historyCumSumSize = extra->allocUniform();
        {
            auto ptr = (uint32_t*)historyCumSumSize->map();
            ptr[0] = 256 * 256 * 8;
            historyCumSumSize->unmap();
        }
        int unit = 128;
        int cumsum_local_size = 256;
        uint32_t cumsum_min_cost = UINT_MAX;
        for(int l = 64; l <= maxLocalSize; l *= 2){
            for(int un = 64; un <= 256; un *= 2){
                std::vector<int> gps = {1, 1, 1};
                std::vector<uint32_t> spc = {(uint32_t)l, 1, 1, (uint32_t)un, (uint32_t)l};
                mContent->cumsum->changePipeline(spc);
                SharedPtr<VulkanLayout::DescriptorSet> des = mContent->cumsum->createSet();
                des->writeBuffer(extra->getTensorBuffer(histogram.get()).first->buffer(), 0, histogramSize, extra->getTensorBuffer(histogram.get()).second);
                des->writeBuffer(extra->getTensorBuffer(histogramSum.get()).first->buffer(), 1, histogramSize, extra->getTensorBuffer(histogramSum.get()).second);
                des->writeBuffer(historyCumSumSize->buffer(), 2, historyCumSumSize->size());
                auto time = (uint32_t)extra->getPipelineTime(mContent->cumsum, des, gps);
                if(time < cumsum_min_cost){
                    unit = un;
                    cumsum_local_size = l;
                    cumsum_min_cost = time;
                }
            }
        }
        std::vector<uint32_t> spc = {(uint32_t)cumsum_local_size, 1, 1, (uint32_t)unit, (uint32_t)cumsum_local_size};
        mContent->cumsum->changePipeline(spc);
    }
    int binSize = (1<<mPerSortBit);
    int numerPass = UP_DIV(mNeedBits, mPerSortBit);
    for(int g = 8; g <= 256; g *= 2){
        for(int l = 128; l <= maxLocalSize; l *= 2){
            uint32_t time = 0;
            size_t histogramSize = binSize * l * g * sizeof(uint32_t);
            std::shared_ptr<VulkanBuffer> historyCumSumSize = extra->allocUniform();
            {
                auto ptr = (uint32_t*)historyCumSumSize->map();
                ptr[0] = binSize * l * g;
                historyCumSumSize->unmap();
            }
            // compute histogram
            {
                std::vector<int> gps = {g, 1, 1};
                std::vector<uint32_t> lws = {(uint32_t)l, 1, 1, (uint32_t)binSize, (uint32_t)l};
                mContent->radixsort_histogram->changePipeline(lws);
                SharedPtr<VulkanLayout::DescriptorSet> des = mContent->radixsort_histogram->createSet();
                des->writeBuffer(extra->getTensorBuffer(histogram.get()).first->buffer(), 0, histogramSize, extra->getTensorBuffer(histogram.get()).second);
                des->writeBuffer(srcIndex.first->buffer(), 1, keySize,  srcIndex.second);
                des->writeBuffer(sortNumber->buffer(), 2,  sortNumber->size());
                des->writeBuffer(pass->buffer(), 3, 4 * sizeof(uint32_t));
                time += (uint32_t)extra->getPipelineTime(mContent->radixsort_histogram, des, gps);
            }
            // cumsum histogram
            {
                std::vector<int> gps = {1, 1, 1};
                SharedPtr<VulkanLayout::DescriptorSet> des = mContent->cumsum->createSet();
                des->writeBuffer(extra->getTensorBuffer(histogram.get()).first->buffer(), 0, histogramSize, extra->getTensorBuffer(histogram.get()).second);
                des->writeBuffer(extra->getTensorBuffer(histogramSum.get()).first->buffer(), 1, histogramSize, extra->getTensorBuffer(histogramSum.get()).second);
                des->writeBuffer(historyCumSumSize->buffer(), 2, historyCumSumSize->size());
                time += (uint32_t)extra->getPipelineTime(mContent->cumsum, des, gps);
            }
            // reorder
            {
                std::vector<int> gps = {g, 1, 1};
                std::vector<uint32_t> lws = {(uint32_t)l, 1, 1,(uint32_t)binSize, (uint32_t)l};
                mContent->radixsort_reorder->changePipeline(lws);
                SharedPtr<VulkanLayout::DescriptorSet> des = mContent->radixsort_reorder->createSet();
                des->writeBuffer(dstIndex.first->buffer(), 0, keySize, dstIndex.second);
                des->writeBuffer(srcIndex.first->buffer(), 1, keySize, srcIndex.second);
                des->writeBuffer(extra->getTensorBuffer(histogramSum.get()).first->buffer(), 2, histogramSize, extra->getTensorBuffer(histogramSum.get()).second);
                des->writeBuffer(sortNumber->buffer(), 3, sortNumber->size());
                des->writeBuffer(pass->buffer(), 4, pass->size());
                time += (uint32_t)extra->getPipelineTime(mContent->radixsort_reorder, des, gps);
            }
            time *= numerPass;
            if(time < min_cost){
                min_cost = time;
                mLocalSize = l;
                mGroupSize = g;
            }
        }
    }
    
    std::vector<int> gps = {mGroupSize, 1, 1};
    std::vector<uint32_t> lws = {(uint32_t)mLocalSize, 1, 1, (uint32_t)(1<<mPerSortBit), (uint32_t)mLocalSize};
    mContent->radixsort_histogram->changePipeline(lws);
    mContent->radixsort_reorder->changePipeline(lws);
}
};
