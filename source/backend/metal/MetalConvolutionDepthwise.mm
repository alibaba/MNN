//
//  MetalConvolutionDepthwise.mm
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalConvolutionDepthwise.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED

namespace MNN {
static const char* gDepthwiseMultiInputTransform = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void depthwise_weight_pack(const device IType* src [[buffer(0)]],
                                  device OType4* dst [[buffer(1)]],
                                  constant int2& cst [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    int z = (int)gid.x;
    int k = (int)gid.y;
    int base = z * 4;
    int channel = cst.x;
    int kernelSize = cst.y;
    if (base >= channel || k >= kernelSize) {
        return;
    }
    OType4 value = OType4(0);
    value.x = (OType)src[(base + 0) * kernelSize + k];
    if (base + 1 < channel) {
        value.y = (OType)src[(base + 1) * kernelSize + k];
    }
    if (base + 2 < channel) {
        value.z = (OType)src[(base + 2) * kernelSize + k];
    }
    if (base + 3 < channel) {
        value.w = (OType)src[(base + 3) * kernelSize + k];
    }
    dst[z * kernelSize + k] = value;
}

kernel void depthwise_bias_pack(const device IType* src [[buffer(0)]],
                                device OType4* dst [[buffer(1)]],
                                constant int& channel [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    int base = (int)gid * 4;
    if (base >= channel) {
        return;
    }
    OType4 value = OType4(0);
    value.x = (OType)src[base + 0];
    if (base + 1 < channel) {
        value.y = (OType)src[base + 1];
    }
    if (base + 2 < channel) {
        value.z = (OType)src[base + 2];
    }
    if (base + 3 < channel) {
        value.w = (OType)src[base + 3];
    }
    dst[(int)gid] = value;
}

kernel void depthwise_bias_zero(device OType4* dst [[buffer(0)]],
                                constant int& channel [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
    if ((int)gid * 4 >= channel) {
        return;
    }
    dst[(int)gid] = OType4(0);
}
)metal";

MetalConvolutionDepthwise::MetalConvolutionDepthwise(Backend *backend, const MNN::Op *op)
    : MetalConvolutionCommon(backend, op, nullptr) {
    loadWeight(op);
}
MetalConvolutionDepthwise::MetalConvolutionDepthwise(Backend *backend, const MNN::Op *op, bool dynamicWeight)
    : MetalConvolutionCommon(backend, op, nullptr) {
    mDynamicWeight = dynamicWeight;
    if (!mDynamicWeight) {
        loadWeight(op);
    }
}
MetalConvolutionDepthwise::MetalConvolutionDepthwise(Backend *backend, const MNN::Op *op, std::shared_ptr<MNN::Tensor> weight,
                          std::shared_ptr<MNN::Tensor> bias) : MetalConvolutionCommon(backend, op, bias) {
    mWeight = weight;
}

ErrorCode MetalConvolutionDepthwise::onResize(const std::vector<Tensor *> &inputs,
                                              const std::vector<Tensor *> &outputs) {
    MetalConvolutionCommon::onResize(inputs, outputs);
    auto backend = static_cast<MetalBackend *>(this->backend());

    // prepare
    auto input = inputs[0], output = outputs[0];
    auto iw = input->width();
    auto ih = input->height();
    auto ic_4 = UP_DIV(input->channel(), 4);
    auto ow = output->width();
    auto oh = output->height();
    auto ob = output->batch();
    auto oc_4 = UP_DIV(output->channel(), 4);

    if (mDynamicWeight) {
        if (inputs.size() < 2 || inputs[1]->getType().code != halide_type_float) {
            return NOT_SUPPORT;
        }
        auto context = (__bridge MNNMetalContext *)backend->context();
        auto rt = (MetalRuntime *)backend->runtime();
        const int channel = output->channel();
        const int kernelSize = mKernelX * mKernelY;
        const int weightLength = oc_4 * 4 * kernelSize;
        const int biasLength = UP_DIV(channel, 16) * 16;
        mWeight.reset(MNN::Tensor::createDevice<float>({weightLength}));
        mBias.reset(MNN::Tensor::createDevice<float>({biasLength}));
        bool res = backend->onAcquireBuffer(mWeight.get(), Backend::DYNAMIC);
        res = res && backend->onAcquireBuffer(mBias.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        backend->onReleaseBuffer(mWeight.get(), Backend::DYNAMIC);
        backend->onReleaseBuffer(mBias.get(), Backend::DYNAMIC);

        int weightConstants[] = {channel, kernelSize};
        mWeightTransformConstBuffer = backend->getConstBuffer(sizeof(weightConstants));
        ::memcpy(mWeightTransformConstBuffer.contents, weightConstants, sizeof(weightConstants));
        mBiasTransformConstBuffer = backend->getConstBuffer(sizeof(channel));
        ::memcpy(mBiasTransformConstBuffer.contents, &channel, sizeof(channel));

        auto inputType = backend->useFp16InsteadFp32() ? @"half" : @"float";
        auto inputType4 = backend->useFp16InsteadFp32() ? @"half4" : @"float4";
        std::vector<std::string> keys = {
            "depthwise_multi_input_transform",
            backend->useFp16InsteadFp32() ? "fp16" : "fp32"
        };
        auto weightKeys = keys;
        weightKeys.emplace_back("weight");
        mWeightTransformPipeline = rt->findPipeline(weightKeys);
        if (nil == mWeightTransformPipeline) {
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:inputType forKey:@"IType"];
            [dic setValue:inputType forKey:@"OType"];
            [dic setValue:inputType4 forKey:@"OType4"];
            option.preprocessorMacros = dic;
            mWeightTransformPipeline = backend->makeComputePipelineWithSourceOption(gDepthwiseMultiInputTransform, "depthwise_weight_pack", option);
            rt->insertPipeline(weightKeys, mWeightTransformPipeline);
        }
        auto biasKeys = keys;
        biasKeys.emplace_back(inputs.size() > 2 ? "bias" : "zero_bias");
        mBiasTransformPipeline = rt->findPipeline(biasKeys);
        if (nil == mBiasTransformPipeline) {
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:inputType forKey:@"IType"];
            [dic setValue:inputType forKey:@"OType"];
            [dic setValue:inputType4 forKey:@"OType4"];
            option.preprocessorMacros = dic;
            mBiasTransformPipeline = backend->makeComputePipelineWithSourceOption(gDepthwiseMultiInputTransform, inputs.size() > 2 ? "depthwise_bias_pack" : "depthwise_bias_zero", option);
            rt->insertPipeline(biasKeys, mBiasTransformPipeline);
        }
        mWeightTransformThreads = [context computeBestGroupAndLocal:mWeightTransformPipeline threads:MTLSizeMake(oc_4, kernelSize, 1)];
        mBiasTransformThreads = [context computeBestGroupAndLocal:mBiasTransformPipeline threads:MTLSizeMake(oc_4, 1, 1)];
    }

    auto pads = ConvolutionCommon::convolutionPad(input, output, mOp->main_as_Convolution2D()->common());
    auto padX = pads.first;
    auto padY = pads.second;

    // create const buffer
    int constants[] = {iw,
                       ih,
                       iw * ih,
                       ow,
                       oh,
                       ow * oh,
                       ic_4,
                       ob,

                       mKernelX,
                       mKernelY,
                       mKernelX * mKernelY,
                       mStrideX,
                       mStrideY,
                       padX,
                       padY,
                       mDilateX,
                       mDilateY,
                       mActivationType};
    mConstBuffer = backend->getConstBuffer(sizeof(constants));

    ::memcpy(mConstBuffer.contents, constants, sizeof(constants));

    auto context = (__bridge MNNMetalContext *)backend->context();
    mPipeline = [context pipelineWithName:@"conv_depthwise" fp16:backend->useFp16InsteadFp32()];

    NSUInteger gid_x = ow;
    NSUInteger gid_y = oh;
    NSUInteger gid_z = oc_4*ob;

    NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                    (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                    mConstBuffer, (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
    const Tensor* weight = mWeight.get();
    const Tensor* bias = mBias.get();
    int buffer_offset[] = {
        TensorUtils::getDescribeOrigin(input)->offset,
        TensorUtils::getDescribeOrigin(output)->offset,
        0,
        TensorUtils::getDescribeOrigin(weight)->offset,
        TensorUtils::getDescribeOrigin(bias)->offset
    };

    std::string name = "conv_depthwise";
    MetalRuntime *rt = (MetalRuntime *)backend->runtime();
    auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets:buffer_offset queue:backend->queue()];
    mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
    return NO_ERROR;
}

void MetalConvolutionDepthwise::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    if (mDynamicWeight) {
        [encoder setComputePipelineState:mWeightTransformPipeline];
        MetalBackend::setTensor(inputs[1], encoder, 0);
        MetalBackend::setTensor(mWeight.get(), encoder, 1);
        [encoder setBuffer:mWeightTransformConstBuffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:mWeightTransformThreads.first threadsPerThreadgroup:mWeightTransformThreads.second];

        [encoder setComputePipelineState:mBiasTransformPipeline];
        if (inputs.size() > 2) {
            MetalBackend::setTensor(inputs[2], encoder, 0);
            MetalBackend::setTensor(mBias.get(), encoder, 1);
            [encoder setBuffer:mBiasTransformConstBuffer offset:0 atIndex:2];
        } else {
            MetalBackend::setTensor(mBias.get(), encoder, 0);
            [encoder setBuffer:mBiasTransformConstBuffer offset:0 atIndex:1];
        }
        [encoder dispatchThreadgroups:mBiasTransformThreads.first threadsPerThreadgroup:mBiasTransformThreads.second];
    }

    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(inputs[0], encoder, 0);
    MetalBackend::setTensor(outputs[0], encoder, 1);
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    MetalBackend::setTensor(mWeight.get(), encoder, 3);
    MetalBackend::setTensor(mBias.get(), encoder, 4);
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}

template <typename FType, typename TType>
static void weightInBlock(int group, int kh, int kw, const FType *src, uint8_t* dstOrigin) {
    auto dst    = (TType *)dstOrigin;
    for (int g = 0; g < group; g++) {
        auto z = g / 4, r = g % 4;
        auto z_dst = dst + z * kh * kw * 4 + r;
        for (int h = 0; h < kh; h++) {
            for (int w = 0; w < kw; w++) {
                // to   [g/4][h][w][4]
                // from [g][h][w]
                // dst[(z * kh * kw + h * kw + w) * 4 + r] =
                // src[ g * kh * kw + h * kw + w];
                z_dst[(h * kw + w) * 4] = *src++;
            }
        }
    }
}

bool MetalConvolutionDepthwise::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    if (mDynamicWeight) {
        *dst = new MetalConvolutionDepthwise(bn, op, true);
        return true;
    }
    auto exe = new MetalConvolutionDepthwise(bn, op, mWeight, mBias);
    *dst = exe;
    return true;
}

std::shared_ptr<MNN::Tensor> MetalConvolutionDepthwise::weightTransform(int group, int oc, int ic, int kh, int kw, const float *src, bool int8Weight, bool int4Weight, id<MTLBuffer> srcGpuBuffer, int subBits) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto length = UP_DIV(group, 4) * 4 * kw * kh;
    std::shared_ptr<MNN::Tensor> t(MNN::Tensor::createDevice<float>({length}));
    auto res = backend->onAcquireBuffer(t.get(), Backend::STATIC);
    if (!res) {
        MNN_ERROR("Alloca gpu memory error in MetalConvolutionDepthwise\n");
        return nullptr;
    }
    auto buffer = MetalBackend::getBuffer(t.get());
    auto content = (uint8_t*)[buffer.first contents] + buffer.second;
    if (backend->useFp16InsteadFp32()) {
        weightInBlock<float, __fp16>(group, kh, kw, src, content);
    } else {
        weightInBlock<float, float>(group, kh, kw, src, content);
    }
    return t;
}

class MetalConvolutionDepthwiseCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        if (inputs.size() > 1) {
            auto common = op->main_as_Convolution2D()->common();
            if (inputs[1]->getType().code != halide_type_float || common->group() != common->outputCount()) {
                return nullptr;
            }
            return new MetalConvolutionDepthwise(backend, op, true);
        }
        return new MetalConvolutionDepthwise(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalConvolutionDepthwiseCreator, OpType_ConvolutionDepthwise);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
