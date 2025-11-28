//
//  MetalReduction.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalReduction.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "MetalCast.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "core/TensorUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {
static const char* gReduceTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct constBuffer
{
    // outside_size, axis_size, inside_size, outside_step
    int4 size;
};

#define SIMD_GROUP_WIDTH 32
kernel void reduce_shader(const device T* uInput [[buffer(0)]],
        device T* uOutput [[buffer(1)]], 
        constant constBuffer& uConst [[buffer(2)]], 
#ifdef SIMD_GROUP_REDUCE
    uint3 gid[[threadgroup_position_in_grid]],
    uint  tiisg[[thread_index_in_simdgroup]],
    uint  sgitg[[simdgroup_index_in_threadgroup]]
#else
    uint3 gid[[thread_position_in_grid]]
#endif
) {
    int outside_size = uConst.size.x;
    if(gid.x >= outside_size) {
        return;
    }
    int axis_size = uConst.size.y;
    int inside_size = uConst.size.z;
    int outside_step = uConst.size.w;
    auto axis_in = uInput + gid.x * outside_step + gid.y;
#ifdef SIMD_GROUP_REDUCE
    #ifdef COMPUTE_REDUCE_MAX
        T res = (T)(-FLT_MAX);
        for(int i = tiisg; i < axis_size; i+=SIMD_GROUP_WIDTH){
            T data = axis_in[i * inside_size];
            res = max(res, data);
        }
        res = simd_max(res);
    #elif defined(COMPUTE_REDUCE_SUM)
        T res = (T)0;
        for(int i = tiisg; i < axis_size; i+=SIMD_GROUP_WIDTH){
            T data = axis_in[i * inside_size];
            res += data;
        }
        res = simd_sum(res);
    #elif defined(COMPUTE_REDUCE_MEAN)
        T res = (T)0;
        for(int i = tiisg; i < axis_size; i+=SIMD_GROUP_WIDTH){
            T data = axis_in[i * inside_size];
            res += data;
        }
        res = simd_sum(res);
        res = res / axis_size;
    #elif defined(COMPUTE_REDUCE_MIN)
        T res = (T)(FLT_MAX);
        for(int i = tiisg; i < axis_size; i+=SIMD_GROUP_WIDTH){
            T data = axis_in[i * inside_size];
            res = min(res, data);
        }
        res = simd_min(res);
    #elif defined(COMPUTE_REDUCE_PROD)
        T res = (T)1;
        for(int i = tiisg; i < axis_size; i+=SIMD_GROUP_WIDTH){
            T data = axis_in[i * inside_size];
            res *= data;
        }
        res = simd_product(res);
    #endif
    if(tiisg == 0) {
        uOutput[int(gid.x) * inside_size + int(gid.y)] = (T)res;
    }
#else
    #ifdef COMPUTE_REDUCE_MAX
        T res = (T)(-FLT_MAX);
        for (int i = 0; i < axis_size; i++) {
            T data = axis_in[i * inside_size];
            res = max(res, data);
        }
    #elif defined(COMPUTE_REDUCE_SUM)
        M res = (M)0;
        for(int i = 0; i < axis_size; i++){
            T data = axis_in[i * inside_size];
            res += (M)data;
        }
    #elif defined(COMPUTE_REDUCE_MEAN)
        T res = (T)0;
        for(int i = 0; i < axis_size; i++){
            T data = axis_in[i * inside_size];
            res += (M)data;
        }
        res = res / axis_size;
    #elif defined(COMPUTE_REDUCE_MIN)
        T res = (T)(FLT_MAX);
        for(int i = 0; i < axis_size; i++){
            T data = axis_in[i * inside_size];
            res = min(res, data);
        }
    #elif defined(COMPUTE_REDUCE_PROD)
        M res = (M)1;
        for(int i = 0; i < axis_size; i++){
            T data = axis_in[i * inside_size];
            res *= (M)data;
        }
        res = simd_product(res);
    #endif
    uOutput[int(gid.x) * inside_size + int(gid.y)] = (T)res;
#endif

}
)metal";
MetalReduction::MetalReduction(Backend *backend, const ReductionParam *p) : MetalExecution(backend) {
    // The reduce after geometry compute has only one axis
    mAxis = p->dim()->data()[0];
    mReduceType = p->operation();
    auto mkbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mkbn->context();
    mConst = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
}

ErrorCode MetalReduction::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int outsideSize = 1, axisSize = 1, insideSize = 1;
    for (int i = 0; i < mAxis; i++) {
        outsideSize *= inputs[0]->length(i);
    }
    axisSize = inputs[0]->length(mAxis);
    for (int i = mAxis + 1; i < inputs[0]->dimensions(); i++) {
        insideSize *= inputs[0]->length(i);
    }

    auto mtbn = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    ((int *)mConst.contents)[0] = outsideSize;
    ((int *)mConst.contents)[1] = axisSize;
    ((int *)mConst.contents)[2] = insideSize;
    ((int *)mConst.contents)[3] = axisSize * insideSize;
    
    bool useFp16 = mtbn->useFp16InsteadFp32();
    auto type = inputs[0]->getType();
    NSString* T = MetalCast::getScalarType(type, useFp16);
    NSString* M = @"float";
    if(type.code != halide_type_float) {
        M = @"int";
    }
    std::vector<std::string> keys = {
        std::string([T UTF8String]),
        std::string([M UTF8String]),
        "reduce_shader",
    };
    
    switch (mReduceType) {
        case ReductionType_SUM:
            keys.emplace_back("COMPUTE_REDUCE_SUM");
            break;
        case ReductionType_ASUM:
        case ReductionType_SUMSQ:
            MNN_ASSERT(false); // both un-supported
            break;
        case ReductionType_MEAN:
            keys.emplace_back("COMPUTE_REDUCE_MEAN");
            break;
        case ReductionType_MAXIMUM:
            keys.emplace_back("COMPUTE_REDUCE_MAX");
            break;
        case ReductionType_MINIMUM:
            keys.emplace_back("COMPUTE_REDUCE_MIN");
            break;
        case ReductionType_PROD:
            keys.emplace_back("COMPUTE_REDUCE_PROD");
            break;
        default:
            break;
    }

    if(((MetalRuntime*)mtbn->runtime())->supportSimdGroupReduce()) {
        // reduce dimension is large than thread number
        if(axisSize > outsideSize * insideSize) {
            mUseSimdReduce = true;
        }
    }
    if(mUseSimdReduce) {
        keys.emplace_back("SIMD_GROUP_REDUCE");
    }
    auto pipeline = mtbn->runtime()->findPipeline(keys);
    if (nil == pipeline) {
        MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        [dic setValue:T forKey:@"T"];
        [dic setValue:M forKey:@"M"];
        [dic setValue:@"1" forKey:@(keys[3].c_str())];
        if(mUseSimdReduce) {
            [dic setValue:@"1" forKey:@"SIMD_GROUP_REDUCE"];
        }
        compileOptions.preprocessorMacros = dic;

        pipeline = mtbn->makeComputePipelineWithSourceOption(gReduceTemplate, "reduce_shader", compileOptions);
        mtbn->runtime()->insertPipeline(keys, pipeline);
    }
    if (nil == pipeline) {
        MNN_ERROR("Create gather reduce pipeline error\n");
    }
    mPipeline = pipeline;
    if(mUseSimdReduce) {
        mThreads = std::make_pair(MTLSizeMake(outsideSize, insideSize, 1), MTLSizeMake(32, 1, 1));
    } else {
        mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(outsideSize, insideSize, 1)];
    }

    return NO_ERROR;
}

void MetalReduction::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto &input = inputs[0], &output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribe(input)->extra.offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribe(output)->extra.offset atIndex:1];
    [encoder setBuffer:mConst offset:0 atIndex:2];
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}

class MetalReductionCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        auto param = op->main_as_ReductionParam();
        switch (param->operation()) {
            case ReductionType_ALL:
            case ReductionType_ANY:
            case ReductionType_ASUM:
            case ReductionType_SUMSQ:
                return nullptr;
            default:
                break;
        };

        return new MetalReduction(backend, op->main_as_ReductionParam());
    }
};
REGISTER_METAL_OP_CREATOR(MetalReductionCreator, OpType_Reduction);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
