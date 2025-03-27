//
//  MetalRasterAndInterpolate.mm
//  MNN
//
//  Created by MNN on b'2023/11/28'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#import "backend/metal/MetalUnary.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalRaster.hpp"
#import "AllRenderShader.hpp"
#if MNN_METAL_ENABLED
namespace MNN {
struct ImageConstant {
    int point[4];
    int size[4];
    int block[4];
};
struct SamplerInfo {
    unsigned int stride[4];//stride[3] + offset
    unsigned int size[4];//size[3] + totalSize
    unsigned int extent[4];//dstStride[3]+dstOffset
    unsigned int imageSize[4];
};

static void _setMemChunk(const MemChunk& mem, id<MTLComputeCommandEncoder> encoder, int index) {
    [encoder setBuffer:((MetalRuntimeAllocator::MetalBufferAlloc *)mem.first)->getBuffer() offset:mem.second atIndex:index];
}
static void _setTensor(MNN::Tensor* tensor, id<MTLComputeCommandEncoder> encoder, int index) {
    [encoder setBuffer:((MetalRuntimeAllocator::MetalBufferAlloc *)tensor->deviceId())->getBuffer() offset:TensorUtils::getDescribe(tensor)->extra.offset atIndex:index];
}

class MetalRadixSort {
private:
    struct Shaders {
        id<MTLComputePipelineState> cumsum;
        id<MTLComputePipelineState> radixsort_histogram;
        id<MTLComputePipelineState> radixsort_reorder;
    };
    Shaders mPipeline;
    
    struct MidBuffers {
        MemChunk histogram;
        MemChunk histogramSum;
    };
    MidBuffers mBuffer;
    
    struct ConstBuffer {
        id<MTLBuffer> historyCumSumSize;
        std::vector<id<MTLBuffer>> pass;
    };
    ConstBuffer mConst;
    int mPerSortBit = 4;
    int mLocalSize = 256;
    int mGroupSize = 32;
    int mNeedBits = 16;
    int mCumsumLocalSize = 256;
    bool mUseAutoTune = true;
    MetalBackend *mtbn;

public:
    friend class MetalRasterSort;
    MetalRadixSort(Backend *backend, const MNN::Op *op, int needBit) {
        mtbn = static_cast<MetalBackend*>(backend);
        mNeedBits = needBit;
        if (mtbn->isIphone()){
            mUseAutoTune = false;
        }
    }
    virtual ~MetalRadixSort() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, MemChunk &srcIndex, MemChunk &dstIndex) {
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        auto attr = inputs[0];
        auto viewProj = inputs[1];
        auto numberPoint = attr->length(0);
        auto memPool = mtbn->getBufferPool();
        size_t maxHistogramSize = 1024 * 256 * 16 * sizeof(uint32_t);
        const int defaultConstantSize = 128;
        auto sortNumber = [context newDeviceBuffer:defaultConstantSize access:CPUWriteOnly];
        {
            auto ptr = (ImageConstant*)[sortNumber contents];
            ptr->point[0] = numberPoint;
        }
        auto pass = [context newDeviceBuffer:defaultConstantSize access:CPUWriteOnly];
        {
            auto ptr = (uint32_t*)[pass contents];
            ptr[0] = 0;
        }
        auto historyCumSumSize = [context newDeviceBuffer:defaultConstantSize access:CPUWriteOnly];
        ((uint32_t*)[historyCumSumSize contents])[0] = 256 * 32 * 16;
        auto histogram = memPool->alloc(maxHistogramSize);
        auto histogramSum = memPool->alloc(maxHistogramSize);
        
        int unit = 16;
        if (mtbn->isIphone()) {
            unit = 8;
        }
        uint32_t cumsum_min_cost = UINT_MAX;
        int loopNumber = 10;
        if(mUseAutoTune){
            for(int l = 8; l <= 256; l *= 2){
                for(int un = 8; un <= 256; un *= 2){
                    MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
                    compileOptions.preprocessorMacros = @{
                        @"UNIT" : @(un).stringValue,
                        @"LOCAL_SIZE" : @(l).stringValue
                    };
                    id<MTLComputePipelineState> cumsum = mtbn->makeComputePipelineWithSourceOption(render_shader_radixsort_cumsum_metal, "main0", compileOptions);
                    NSArray *arr_cumsum = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)histogramSum.first)->getBuffer(),
                                           (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)histogram.first)->getBuffer(),
                                           historyCumSumSize, nil];
                    auto time = [context PipelinetimeUsed:cumsum global:MTLSizeMake(1, 1, 1) local:MTLSizeMake(l, 1, 1) loop:loopNumber buffer:arr_cumsum queue:mtbn->queue()];
                    if(time < cumsum_min_cost){
                        unit = un;
                        mCumsumLocalSize = l;
                        cumsum_min_cost = time;
                    }
                }
            }
        }
        
        {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"UNIT" : @(unit).stringValue,
                @"LOCAL_SIZE" : @(mCumsumLocalSize).stringValue
            };
            mPipeline.cumsum = mtbn->makeComputePipelineWithSourceOption(render_shader_radixsort_cumsum_metal, "main0", compileOptions);
        }
        
        int binSize = (1<<mPerSortBit);
        int numerPass = UP_DIV(mNeedBits, mPerSortBit);
        if(mUseAutoTune){
            uint32_t min_cost = UINT_MAX;
            for(int g = 8; g <= 512; g *= 2){
                for(int l = 32; l <= 512; l *= 2){
                    uint32_t time = 0;
                    ((uint32_t*)[historyCumSumSize contents])[0] = binSize * l * g;
                    // compute histogram
                    {
                        MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
                        compileOptions.preprocessorMacros = @{
                            @"BIN_NUMBER" : @(binSize).stringValue,
                            @"LOCAL_SIZE" : @(l).stringValue
                        };
                        id<MTLComputePipelineState> radixsort_histogram = mtbn->makeComputePipelineWithSourceOption(render_shader_radixsort_histogram_option_metal, "main0", compileOptions);
                        NSArray *arr_histogram = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)histogram.first)->getBuffer(),
                                                  (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)srcIndex.first)->getBuffer(),
                                                  sortNumber, pass, nil];
                        time += [context PipelinetimeUsed:radixsort_histogram global:MTLSizeMake(g, 1, 1) local:MTLSizeMake(l, 1, 1) loop:10 buffer:arr_histogram queue:mtbn->queue()];
                    }
                    // cumsum histogram
                    {
                        NSArray *arr_cumsum = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)histogramSum.first)->getBuffer(),
                                               (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)histogram.first)->getBuffer(),
                                               historyCumSumSize, nil];
                        time += [context PipelinetimeUsed:mPipeline.cumsum global:MTLSizeMake(1, 1, 1) local:MTLSizeMake(mCumsumLocalSize, 1, 1) loop:loopNumber buffer:arr_cumsum queue:mtbn->queue()];
                    }
                    // reorder
                    {
                        MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
                        compileOptions.preprocessorMacros = @{
                            @"BIN_NUMBER" : @(binSize).stringValue,
                            @"LOCAL_SIZE" : @(l).stringValue
                        };
                        id<MTLComputePipelineState> radixsort_reorder = mtbn->makeComputePipelineWithSourceOption(render_shader_radixsort_reorder_option_metal, "main0", compileOptions);
                        NSArray *arr_reorder = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)dstIndex.first)->getBuffer(),
                                                (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)srcIndex.first)->getBuffer(),
                                                (id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)histogramSum.first)->getBuffer(),
                                                sortNumber, pass, nil];
                        time += [context PipelinetimeUsed:radixsort_reorder global:MTLSizeMake(g, 1, 1) local:MTLSizeMake(l, 1, 1) loop:loopNumber buffer:arr_reorder queue:mtbn->queue()];
                    }
                    time *= numerPass;
                    if(time < min_cost){
                        min_cost = time;
                        mLocalSize = l;
                        mGroupSize = g;
                    }
                }
            }
        }
    
        memPool->free(histogram);
        memPool->free(histogramSum);
        
        size_t histogramSize = binSize * mLocalSize * mGroupSize * sizeof(uint32_t);
        mBuffer.histogram = memPool->alloc(histogramSize);
        mBuffer.histogramSum = memPool->alloc(histogramSize);
        mConst.historyCumSumSize = [context newDeviceBuffer:defaultConstantSize access:CPUWriteOnly];
        ((uint32_t*)[mConst.historyCumSumSize contents])[0] = binSize * mLocalSize * mGroupSize;
        
        mConst.pass.resize(numerPass);
        for (int i=0; i<numerPass; ++i) {
            mConst.pass[i] = [context newDeviceBuffer:defaultConstantSize access:CPUWriteOnly];
            auto pass = mConst.pass[i];
            auto ptr = (uint32_t*)[pass contents];
            ptr[0] = i * mPerSortBit;
        }
        
        {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"BIN_NUMBER" : @(binSize).stringValue,
                @"LOCAL_SIZE" : @(mLocalSize).stringValue
            };
            mPipeline.radixsort_histogram = mtbn->makeComputePipelineWithSourceOption(render_shader_radixsort_histogram_option_metal, "main0", compileOptions);
        }
        {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"BIN_NUMBER" : @(binSize).stringValue,
                @"LOCAL_SIZE" : @(mLocalSize).stringValue
            };
            mPipeline.radixsort_reorder = mtbn->makeComputePipelineWithSourceOption(render_shader_radixsort_reorder_option_metal, "main0", compileOptions);
        }
        
        memPool->free(mBuffer.histogram);
        memPool->free(mBuffer.histogramSum);
        return NO_ERROR;
    }
    void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder, MemChunk &srcIndex, MemChunk &dstIndex,
                  Tensor* sortNumber) {
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        auto attr = inputs[0];
        auto viewProj = inputs[1];
        // Radix sort
        int numerPass = UP_DIV(mNeedBits, mPerSortBit);
        for (int i=0; i<numerPass; ++i) {
            auto pass = mConst.pass[i];
            // compute histogram
            {
                [encoder setComputePipelineState:mPipeline.radixsort_histogram];
                _setMemChunk(mBuffer.histogram, encoder, 0);
                _setMemChunk(srcIndex, encoder, 1);
                _setTensor(sortNumber, encoder, 2);
                [encoder setBuffer:pass offset:0 atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake(mGroupSize, 1, 1) threadsPerThreadgroup:MTLSizeMake(mLocalSize, 1, 1)];
            }
            // cumsum histogram
            {
                [encoder setComputePipelineState:mPipeline.cumsum];
                _setMemChunk(mBuffer.histogramSum, encoder, 0);
                _setMemChunk(mBuffer.histogram, encoder, 1);
                [encoder setBuffer:mConst.historyCumSumSize offset:0 atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(mCumsumLocalSize, 1, 1)];
            }
            // reorder
            {
                [encoder setComputePipelineState:mPipeline.radixsort_reorder];
                _setMemChunk(dstIndex, encoder, 0);
                _setMemChunk(srcIndex, encoder, 1);
                _setMemChunk(mBuffer.histogramSum, encoder, 2);
                _setTensor(sortNumber, encoder, 3);
                [encoder setBuffer:pass offset:0 atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(mGroupSize, 1, 1) threadsPerThreadgroup:MTLSizeMake(mLocalSize, 1, 1)];
            }
            // Swap dst/src
            auto temp = srcIndex;
            srcIndex = dstIndex;
            dstIndex = temp;
        }
    }
};

class MetalRasterSort : public MetalExecution {
private:
    struct Shaders {
        id<MTLComputePipelineState> rastersort_collect_key;
        id<MTLComputePipelineState> rastersort_count_valid_number;
    };
    Shaders mPipeline;
    
    struct MidBuffers {
        MemChunk pointOffsets;
        MemChunk pointOffsetSum;
        MemChunk pointKeysMid;
    };
    MidBuffers mBuffer;
    
    struct ConstBuffer {
        id<MTLBuffer> imageConstant;
        id<MTLBuffer> pointClipHistormSize;
        id<MTLBuffer> blit;
    };
    ConstBuffer mConst;
    int mPrepareLocalSize = 256;
    int mPrepareGroupSize = 16;
    std::shared_ptr<MetalRadixSort> mRadixSort;

public:
    MetalRasterSort(Backend *backend, const MNN::Op *op) : MetalExecution(backend) {
        auto mtbn = static_cast<MetalBackend*>(backend);
        mRadixSort.reset(new MetalRadixSort(backend, op, 16));
    }
    virtual ~MetalRasterSort() {
        // Do nothing
    }
    void prepare(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
        auto attr = inputs[0];
        auto viewProj = inputs[1];
        auto numberPoint = attr->length(0);

        // Count prepare, Compute point offset, rect and newposition
        {
            [encoder setComputePipelineState:mPipeline.rastersort_count_valid_number];
            _setMemChunk(mBuffer.pointOffsets, encoder, 0);
            _setTensor(attr, encoder, 1);
            _setTensor(viewProj, encoder, 2);
            [encoder setBuffer:mConst.imageConstant offset:0 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(mPrepareGroupSize, 1, 1) threadsPerThreadgroup:MTLSizeMake(mPrepareLocalSize, 1, 1)];
        }
        
        // Compute cusum of point offset
        {
            [encoder setComputePipelineState:mRadixSort->mPipeline.cumsum];
            _setMemChunk(mBuffer.pointOffsetSum, encoder, 0);
            _setMemChunk(mBuffer.pointOffsets, encoder, 1);
            [encoder setBuffer:mConst.pointClipHistormSize offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(mRadixSort->mCumsumLocalSize, 1, 1)];
        }
        // Compute pointKeys
        {
            [encoder setComputePipelineState:mPipeline.rastersort_collect_key];
            _setTensor(outputs[1], encoder, 0);
            _setTensor(attr, encoder, 1);
            _setTensor(viewProj, encoder, 2);
            _setMemChunk(mBuffer.pointOffsetSum, encoder, 3);
            [encoder setBuffer:mConst.imageConstant offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(mPrepareGroupSize, 1, 1) threadsPerThreadgroup:MTLSizeMake(mPrepareLocalSize, 1, 1)];
        }
    }
    void setupPrepare(NSString* T) {
        auto mtbn = static_cast<MetalBackend*>(backend());
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"TYPE" : T,
                @"LOCAL_SIZE" : @(mPrepareLocalSize).stringValue
            };
            auto pipeline = mtbn->makeComputePipelineWithSourceOption(render_shader_rastersort_collect_key_metal, "main0", compileOptions);
            mPipeline.rastersort_collect_key = pipeline;
        }
        {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"TYPE" : T,
                @"LOCAL_SIZE" : @(mPrepareLocalSize).stringValue
            };
            auto pipeline = mtbn->makeComputePipelineWithSourceOption(render_shader_rastersort_count_valid_number_metal, "main0", compileOptions);
            mPipeline.rastersort_count_valid_number = pipeline;
        }
        ((uint32_t*)[mConst.pointClipHistormSize contents])[0] = mPrepareGroupSize * mPrepareLocalSize;

    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto mtbn = static_cast<MetalBackend*>(backend());
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        auto attr = inputs[0];
        auto numAttr = attr->length(1);
        bool autoTune = !mtbn->isIphone();

        NSString* T = nil;
        if (4 == numAttr) {
            T = @"float4";
        } else {
            T = @"half4";
        }
        auto viewProj = inputs[1];
        auto numberPoint = attr->length(0);
        auto memPool = mtbn->getBufferPool();

        // gaussian prepare, Compute point offset, rect and newposition
        const int defaultConstantSize = 128;
        // gaussian prepare, Compute point offset, rect and newposition
        mConst.imageConstant = [context newDeviceBuffer:defaultConstantSize access:CPUWriteOnly];
        {
            auto ptr = (ImageConstant*)[mConst.imageConstant contents];
            ptr->point[0] = numberPoint;
        }
        mConst.pointClipHistormSize = [context newDeviceBuffer:defaultConstantSize access:CPUWriteOnly];
        // Alloc Max Size
        auto pointOffsetBytes = 1024 * 256 * sizeof(uint32_t);
        mBuffer.pointOffsets = memPool->alloc(pointOffsetBytes);
        // Compute cusum of point offset
        mBuffer.pointOffsetSum = memPool->alloc(pointOffsetBytes);

        memPool->free(mBuffer.pointOffsets);
        // Collect pointKeys
        auto keySize = UP_DIV(numberPoint, 2) * 2 * sizeof(uint32_t) * 2;

        // Radix Sort
        memPool->free(mBuffer.pointOffsetSum);
        mBuffer.pointKeysMid = memPool->alloc(keySize);
        
        MemChunk srcIndex;
        srcIndex.first = (void*)outputs[1]->deviceId();
        srcIndex.second = TensorUtils::getDescribe(outputs[1])->extra.offset;
        auto dstIndex = mBuffer.pointKeysMid;
        mRadixSort->onResize(inputs, outputs, srcIndex, dstIndex);
        memPool->free(mBuffer.pointKeysMid);

        // Reset mGroupSize and mLocalSize
        if(autoTune) {
            int bestGroup = mPrepareGroupSize;
            int bestLocal = mPrepareLocalSize;
            int loop = 10;
            auto queue = mtbn->queue();
            uint32_t min_cost = UINT_MAX;
            for(int g = 8; g <= 512; g *= 2){
                for(int l = 32; l <= 512; l *= 2){
                    uint32_t time = 0;
                    mPrepareGroupSize = g;
                    mPrepareLocalSize = l;
                    setupPrepare(T);
                    auto commamd_buffer = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [commamd_buffer computeCommandEncoder];

                    for (int v=0; v<loop; ++v) {
                        prepare(inputs, outputs, encoder);
                    }
                    [encoder endEncoding];
                    time = [context timeUsed:commamd_buffer];
                    // compute histogram
                    if(time < min_cost){
                        min_cost = time;
                        bestLocal = l;
                        bestGroup = g;
                    }
                }
            }
            mPrepareGroupSize = bestGroup;
            mPrepareLocalSize = bestLocal;
        }
        setupPrepare(T);

        SamplerInfo info;
        info.extent[0] = 0;
        info.extent[1] = 0;
        info.extent[2] = 0;
        info.extent[3] = 0;
        info.stride[0] = 1;
        info.stride[1] = 0;
        info.stride[2] = 0;
        info.stride[3] = (mPrepareGroupSize * mPrepareLocalSize - 1);
        info.size[0] = 1;
        info.size[1] = 1;
        info.size[2] = 1;
        mConst.blit = [context newDeviceBuffer:sizeof(SamplerInfo) bytes:&info access:CPUWriteOnly];

        return NO_ERROR;
    }
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override {
        auto backend = static_cast<MetalBackend *>(this->backend());
        auto context = (__bridge MNNMetalContext *)backend->context();
        auto attr = inputs[0];
        auto viewProj = inputs[1];
        auto numberPoint = attr->length(0);
        prepare(inputs, outputs, encoder);
        {
            auto blitPipeline = MetalRaster::getBlitPipeline(4, backend, false);
            [encoder setComputePipelineState:blitPipeline];
            _setMemChunk(mBuffer.pointOffsetSum, encoder, 0);
            _setTensor(outputs[0], encoder, 1);
            [encoder setBuffer:mConst.blit offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }
        // Radix sort
        MemChunk srcIndex;
        srcIndex.first = (void*)outputs[1]->deviceId();
        srcIndex.second = TensorUtils::getDescribe(outputs[1])->extra.offset;
        auto dstIndex = mBuffer.pointKeysMid;
        mRadixSort->onEncode(inputs, outputs, encoder, srcIndex, dstIndex, outputs[0]);
    }
};


class MetalRasterAndInterpolateCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        int type = 4;
        if (op->main_type() == OpParameter_Extra) {
            auto extra = op->main_as_Extra();
            if (nullptr != extra->attr()) {
                for (int i=0; i<extra->attr()->size(); ++i) {
                    auto attr = extra->attr()->GetAs<Attribute>(i);
                    if (attr->key()->str() == "primitiveType") {
                        type = attr->i();
                        continue;
                    }
                }
            }
        }
        if (6 == type) {
            return new MetalRasterSort(backend, op);
        }
        return nullptr;
    }
};
REGISTER_METAL_OP_CREATOR(MetalRasterAndInterpolateCreator, OpType_RasterAndInterpolate);
};

#endif

