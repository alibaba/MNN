//
//  MetalLoop.mm
//  MNN
//
//  Created by MNN on 2023/12/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "core/Macro.h"
#import "MetalCast.hpp"
#import "MetalBinary.hpp"
#import "MetalBackend.hpp"
#import "MNNMetalContext.h"
#include "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {
static const char* gMatMulUnitTemplate = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct constBuffer
{
    int4 size;
    int4 stride_o;
    int4 stride_a;
    int4 stride_b;
    int4 stride_c;
    int4 _step;
    int4 iter;
};

struct s3
{
    T data[1];
};

struct s4
{
    T data[1];
};

struct s5
{
    T data[1];
};

struct s6
{
    T data[1];
};

struct s0
{
    T data[1];
};

struct s1
{
    T data[1];
};

struct s2
{
    T data[1];
};

struct d0
{
    T data[1];
};

kernel void main0(device d0& uOutput [[buffer(0)]], const device s0& uInputA [[buffer(1)]], const device s1& uInputB [[buffer(2)]],
#ifdef HAS_BIAS
    const device s2& uInputC [[buffer(3)]],
    const device s3& uOOffset [[buffer(4)]],
    const device s4& uAOffset [[buffer(5)]],
    const device s5& uBOffset [[buffer(6)]],
    const device s6& uCOffset [[buffer(7)]],
    constant constBuffer& uConstant [[buffer(8)]],
#else
    const device s3& uOOffset [[buffer(3)]],
    const device s4& uAOffset [[buffer(4)]],
    const device s5& uBOffset [[buffer(5)]],
    constant constBuffer& uConstant [[buffer(6)]],
#endif
    uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int3 posTmp = int3(gl_GlobalInvocationID);
    int e = uConstant.size.x;
    int l = uConstant.size.y;
    int h = uConstant.size.z;
    int n = uConstant.size.w;
    int eh = e * h;
    if (posTmp.x < (eh * n))
    {
        int regionInsideIndex = posTmp.x % eh;
        int regionOutsideIndex = posTmp.x / eh;
        int X = regionInsideIndex % e;
        int Y = regionInsideIndex / e;
        int4 index = int4(regionOutsideIndex, regionOutsideIndex, regionOutsideIndex, regionOutsideIndex);
        if (uConstant.iter.x >= 0)
        {
            index.x = int(uOOffset.data[regionOutsideIndex]);
        }
        if (uConstant.iter.y >= 0)
        {
            index.y = int(uAOffset.data[regionOutsideIndex]);
        }
        if (uConstant.iter.z >= 0)
        {
            index.z = int(uBOffset.data[regionOutsideIndex]);
        }
#ifdef HAS_BIAS
        if (uConstant.iter.w >= 0)
        {
            index.w = int(uCOffset.data[regionOutsideIndex]);
        }
#endif
        int4 offset = index * uConstant._step;
        T value = 0.0;
        int aOffset = (offset.y + uConstant.stride_a.w) + (X * uConstant.stride_a.x);
        int bOffset = (offset.z + uConstant.stride_b.w) + (Y * uConstant.stride_b.z);
        for (int i = 0; i < l; i++)
        {
            value += (uInputA.data[aOffset + (i * uConstant.stride_a.y)] * uInputB.data[bOffset + (i * uConstant.stride_b.y)]);
        }
#ifdef HAS_BIAS
        value += uInputC.data[(offset.w + (Y * uConstant.stride_c.z)) + (X * uConstant.stride_c.x)];
#endif
        uOutput.data[((offset.x + uConstant.stride_o.w) + (X * uConstant.stride_o.x)) + (Y * uConstant.stride_o.z)] = value;
    }
}
)metal";

struct VulkanBatchMatMulInfo {
    int size[4];
    int stride_o[4];
    int stride_a[4];
    int stride_b[4];
    int stride_c[4];
    int step[4];
    int iter[4];
};
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

class MetalBatchMatMul : public MetalExecution {
private:
    const LoopParam* mLoop;
    id<MTLBuffer> mParam;
    id<MTLComputePipelineState> mPipeline;
    std::vector<Tensor*> mTensors;
    bool mHasBias = false;

public:
    MetalBatchMatMul(const LoopParam* loop, Backend *bn) : MetalExecution(bn) {
        mLoop = loop;
        auto mtbn = static_cast<MetalBackend *>(bn);
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        mParam = [context newDeviceBuffer:sizeof(VulkanBatchMatMulInfo) access:CPUWriteOnly];
        bool useFp16 = mtbn->useFp16InsteadFp32();
        NSString* T = nil;
        if (useFp16) {
            T = @"half";
        } else {
            T = @"float";
        }
        std::vector<std::string> keys = {
            std::string([T UTF8String]),
            "matmulunit"
        };
        auto cmd = loop->commands()->GetAs<RegionCommand>(0);
        mHasBias = cmd->indexes()->size() > 3;
        if (mHasBias) {
            keys.emplace_back("BIAS");
        }
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            if (!mHasBias) {
                compileOptions.preprocessorMacros = @{
                    @"T" : T,
                };
            } else {
                compileOptions.preprocessorMacros = @{
                    @"T" : T,
                    @"HAS_BIAS":@"1",
                };
            }
            pipeline = mtbn->makeComputePipelineWithSourceOption(gMatMulUnitTemplate, "main0", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Create batch matmul pipeline error\n");
        }
        mPipeline = pipeline;
        mTensors.resize(mLoop->tensorNumber());
    }
    virtual ~MetalBatchMatMul() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs) override {
        _setTensorStack(mTensors, inputs, outputs, mLoop);
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto size = cmd->size()->data();
        auto AStride = cmd->view()->GetAs<View>(1)->stride()->data();
        auto BStride = cmd->view()->GetAs<View>(2)->stride()->data();
        auto OStride = cmd->view()->GetAs<View>(0)->stride()->data();
        int totalSize = mLoop->loopNumber() * size[0] * size[1] * size[2];
        auto param = reinterpret_cast<VulkanBatchMatMulInfo*>([mParam contents]);
        param->size[3] = mLoop->loopNumber();
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
        return NO_ERROR;
    }
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override {
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto size = cmd->size()->data();
        auto AStride = cmd->view()->GetAs<View>(1)->stride()->data();
        auto BStride = cmd->view()->GetAs<View>(2)->stride()->data();
        auto OStride = cmd->view()->GetAs<View>(0)->stride()->data();
        int totalSize = mLoop->loopNumber() * size[0] * size[1] * size[2];
        [encoder setComputePipelineState:mPipeline];
        for (int i=0; i<cmd->indexes()->size(); ++i) {
            MetalBackend::setTensor(mTensors[cmd->indexes()->data()[i]], encoder, i);
        }
        auto iter = cmd->iterIndexes()->data();
        for (int i=0; i<cmd->indexes()->size(); ++i) {
            if (iter[i] >= 0) {
                MetalBackend::setTensor(mTensors[iter[i]], encoder, cmd->indexes()->size() + i);
            } else {
                MetalBackend::setTensor(inputs[0], encoder, cmd->indexes()->size() + i);
            }
        }
        [encoder setBuffer:mParam offset:0 atIndex:cmd->indexes()->size() * 2];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(totalSize, 256), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
};

static const char* gBlitRegion = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct constBuffer
{
    int4 stride;
    int4 size;
    int4 extent;
    int4 _step;
    int4 iter;
};

struct s1
{
    int data[1];
};

struct s2
{
    int data[1];
};

struct sourceBuffer
{
    T data[1];
};

struct s0
{
    T data[1];
};

kernel void main0(device sourceBuffer& uOutput [[buffer(0)]], const device s0& uInput [[buffer(1)]], const device s1& uSrcOffset [[buffer(2)]], const device s2& uDstOffset [[buffer(3)]], constant constBuffer& uConstant [[buffer(4)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int3 posTmp = int3(gl_GlobalInvocationID);
    if (posTmp.x < uConstant._step.w)
    {
        int regionInsideIndex = posTmp.x % uConstant.size.w;
        int regionOutsideIndex = posTmp.x / uConstant.size.w;
        int3 pos;
        pos.x = regionInsideIndex / (uConstant.size.y * uConstant.size.z);
        int subIndex = regionInsideIndex % (uConstant.size.y * uConstant.size.z);
        pos.z = subIndex % uConstant.size.z;
        pos.y = subIndex / uConstant.size.z;
        int srcBasicOffset;
        if (uConstant.iter.y > 0)
        {
            srcBasicOffset = uConstant._step.y * int(uSrcOffset.data[regionOutsideIndex]);
        }
        else
        {
            srcBasicOffset = uConstant._step.y * regionOutsideIndex;
        }
        int dstBasicOffset;
        if (uConstant.iter.x > 0)
        {
            dstBasicOffset = uConstant._step.x * int(uDstOffset.data[regionOutsideIndex]);
        }
        else
        {
            dstBasicOffset = uConstant._step.x * regionOutsideIndex;
        }
        int srcOffset = (((srcBasicOffset + uConstant.stride.w) + (uConstant.stride.z * pos.z)) + (uConstant.stride.y * pos.y)) + (uConstant.stride.x * pos.x);
        int dstOffset = (((dstBasicOffset + uConstant.extent.w) + (pos.x * uConstant.extent.x)) + (pos.y * uConstant.extent.y)) + (pos.z * uConstant.extent.z);
        uOutput.data[dstOffset] = uInput.data[srcOffset];
    }
}
)metal";

struct GatherInfo {
    int stride[4];
    int size[4];
    int extent[4];
    int step[4];
    int iter[4];
};

class MetalGather : public MetalExecution {
private:
    const LoopParam* mLoop;
    id<MTLBuffer> mParam;
    id<MTLComputePipelineState> mPipeline;
    std::vector<Tensor*> mTensors;
public:
    MetalGather(const LoopParam* loop, Backend *bn, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) : MetalExecution(bn) {
        mLoop = loop;
        auto mtbn = static_cast<MetalBackend *>(bn);
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        mParam = [context newDeviceBuffer:sizeof(GatherInfo) access:CPUWriteOnly];
        bool useFp16 = mtbn->useFp16InsteadFp32();
        mTensors.resize(mLoop->tensorNumber());
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        _setTensorStack(mTensors, inputs, outputs, mLoop);
        auto dstTensor = mTensors[cmd->indexes()->data()[0]];

        NSString* T = MetalCast::getScalarType(dstTensor->getType(), useFp16);
        std::vector<std::string> keys = {
            std::string([T UTF8String]),
            "blitregion"
        };
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"T" : T,
            };
            pipeline = mtbn->makeComputePipelineWithSourceOption(gBlitRegion, "main0", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Create gather pipeline error\n");
        }
        mPipeline = pipeline;
    }
    virtual ~MetalGather() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs) override {
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        _setTensorStack(mTensors, inputs, outputs, mLoop);
        auto srcStride = cmd->view()->GetAs<View>(1)->stride()->data();
        auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
        auto size = cmd->size()->data();
        int totalSize = mLoop->loopNumber() * size[0] * size[1] * size[2];
        auto param = reinterpret_cast<GatherInfo*>([mParam contents]);
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
            param->iter[0] = 1;
        }
        if (iterIndex[1] >= 0) {
            param->iter[1] = 1;
        }
        return NO_ERROR;
    }
    virtual void onEncode(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override {
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto size = cmd->size()->data();
        auto srcStride = cmd->view()->GetAs<View>(1)->stride()->data();
        auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
        int totalSize = mLoop->loopNumber() * size[0] * size[1] * size[2];
        
        [encoder setComputePipelineState:mPipeline];
        auto dstTensor = mTensors[cmd->indexes()->data()[0]];
        auto srcTensor = mTensors[cmd->indexes()->data()[1]];
        MetalBackend::setTensor(dstTensor, encoder, 0);
        MetalBackend::setTensor(srcTensor, encoder, 1);

        auto iterIndex = cmd->iterIndexes()->data();
        if (iterIndex[0] >= 0) {
            MetalBackend::setTensor(mTensors[iterIndex[0]], encoder, 3);
        } else {
            MetalBackend::setTensor(dstTensor, encoder, 3);
        }
        if (iterIndex[1] >= 0) {
            MetalBackend::setTensor(mTensors[iterIndex[1]], encoder, 2);
        } else {
            MetalBackend::setTensor(srcTensor, encoder, 2);
        }
        [encoder setBuffer:mParam offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(totalSize, 256), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
};


static const char* gBinaryBroadcast = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct constBuffer
{
    int4 srcview0;
    int4 srcview1;
    int4 dstview;
    int4 size;
};

static inline __attribute__((always_inline))
int computeVec4dot(thread const int4& a, thread const int4& b)
{
    return (((a.x * b.x) + (a.y * b.y)) + (a.z * b.z)) + (a.w * b.w);
}

kernel void main0(device T1* uOutput [[buffer(0)]], const device T0* uInput0 [[buffer(1)]], const device T0* uInput1 [[buffer(2)]], constant constBuffer& uConstant [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int3 posTmp = int3(gl_GlobalInvocationID);
    if (posTmp.x < uConstant.size.w)
    {
        int4 pos;
        pos.x = posTmp.x / (uConstant.size.y * uConstant.size.z);
        int subIndex = posTmp.x % (uConstant.size.y * uConstant.size.z);
        pos.z = subIndex % uConstant.size.z;
        pos.y = subIndex / uConstant.size.z;
        pos.w = 1;
        int4 param = uConstant.srcview0;
        int4 param_1 = pos;
        int s0 = computeVec4dot(param, param_1);
        int4 param_2 = uConstant.srcview1;
        int4 param_3 = pos;
        int s1 = computeVec4dot(param_2, param_3);
        int4 param_4 = uConstant.dstview;
        int4 param_5 = pos;
        int d = computeVec4dot(param_4, param_5);
        T0 V0 = uInput0[s0];
        T0 V1 = uInput1[s1];
        uOutput[d] = CUSTOM;
    }
}
)metal";

struct BinaryBroadCastInfo {
    int srcview0[4];
    int srcview1[4];
    int dstview[4];
    int size[4];
};

class MetalBinaryBroadCast : public MetalExecution {
public:
    MetalBinaryBroadCast(const LoopParam* loop, Backend *bn, std::vector<Tensor*>&& tensors, NSString* CUSTOM) : MetalExecution(bn) {
        mLoop = loop;
        auto mtbn = static_cast<MetalBackend *>(bn);
        auto context = (__bridge MNNMetalContext *)mtbn->context();
        mParam = mtbn->getConstBuffer(sizeof(BinaryBroadCastInfo));
        mTensors = std::move(tensors);
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto dstTensor = mTensors[cmd->indexes()->data()[0]];
        auto srcTensor = mTensors[cmd->indexes()->data()[1]];
        auto srcTensor1 = mTensors[cmd->indexes()->data()[2]];

        NSString* T1 = MetalCast::getScalarType(dstTensor->getType(), mtbn->useFp16InsteadFp32());
        NSString* T0 = MetalCast::getScalarType(srcTensor->getType(), mtbn->useFp16InsteadFp32());
        std::vector<std::string> keys = {
            std::string([T0 UTF8String]),
            std::string([T1 UTF8String]),
            std::string([CUSTOM UTF8String]),
            "binary_broadcast"
        };
        auto pipeline = mtbn->runtime()->findPipeline(keys);
        if (nil == pipeline) {
            MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
            compileOptions.preprocessorMacros = @{
                @"T0" : T0,
                @"T1" : T1,
                @"CUSTOM" : CUSTOM,
            };
            pipeline = mtbn->makeComputePipelineWithSourceOption(gBinaryBroadcast, "main0", compileOptions);
            mtbn->runtime()->insertPipeline(keys, pipeline);
        }
        if (nil == pipeline) {
            MNN_ERROR("Create Binary Broadcast pipeline error\n");
        }
        mPipeline = pipeline;
    }
    virtual ~MetalBinaryBroadCast() {
        auto mtbn = static_cast<MetalBackend*>(backend());
        mtbn->returnConstBuffer(mParam);
    }
    virtual ErrorCode onResize(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs) override {
        _setTensorStack(mTensors, inputs, outputs, mLoop);
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto size = cmd->size()->data();
        auto srcStride0 = cmd->view()->GetAs<View>(1)->stride()->data();
        auto srcStride1 = cmd->view()->GetAs<View>(2)->stride()->data();
        auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
        mTotalSize = size[0] * size[1] * size[2];
        auto param = reinterpret_cast<BinaryBroadCastInfo*>([mParam contents]);
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
        return NO_ERROR;
    }
    virtual void onEncode(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs,
                               id<MTLComputeCommandEncoder> encoder) override {
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto dstTensor = mTensors[cmd->indexes()->data()[0]];
        auto srcTensor = mTensors[cmd->indexes()->data()[1]];
        auto srcTensor1 = mTensors[cmd->indexes()->data()[2]];
        [encoder setComputePipelineState:mPipeline];
        MetalBackend::setTensor(dstTensor, encoder, 0);
        MetalBackend::setTensor(srcTensor, encoder, 1);
        MetalBackend::setTensor(srcTensor1, encoder, 2);
        [encoder setBuffer:mParam offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(UP_DIV(mTotalSize, 256), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
private:
    const LoopParam* mLoop;
    id<MTLComputePipelineState> mPipeline;
    id<MTLBuffer> mParam;
    std::vector<Tensor*> mTensors;
    int mTotalSize;
};

class MetalLoopCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *bn, const std::vector<Tensor *> &outputs) const {
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
                return new MetalGather(loop, bn, inputs, outputs);
            }
            if (OpType_MatMul == subop->type() && loop->parallel()) {
                return new MetalBatchMatMul(loop, bn);
            }
            if (OpType_BinaryOp == subop->type() && cmd->fuse() < 0 && 1 == loop->loopNumber()) {
                std::vector<MNN::Tensor*> tensors(loop->tensorNumber());
                _setTensorStack(tensors, inputs, outputs, loop);
                auto srcTensor = tensors[cmd->indexes()->data()[1]];

                NSString* CUSTOM = MetalBinary::convert(cmd->op()->main_as_BinaryOp()->opType(), srcTensor->getType().code == halide_type_float);
                if (nil == CUSTOM) {
                    MNN_ERROR("Metal Don't support binary - %d \n", cmd->op()->main_as_BinaryOp()->opType());
                    return nullptr;
                }
                return new MetalBinaryBroadCast(loop, bn, std::move(tensors), CUSTOM);
            }
        }
        return nullptr;
    }
};
REGISTER_METAL_OP_CREATOR(MetalLoopCreator, OpType_While);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
