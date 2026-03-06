//
//  MetalScale.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalScale.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {
static const char* shader_MetalScale_metal =
"struct scale_shape {\n"
" int size;\n"
" int steps;\n"
" int batch;\n"
" int offset;\n"
"};\n"
"kernel void scale_ca(const device M4 *in [[buffer(0)]],\n"
" device M4 *out [[buffer(1)]],\n"
" constant scale_shape &s [[buffer(2)]],\n"
" const device float4 *scalesbias[[buffer(3)]],\n"
" uint2 gid [[thread_position_in_grid]]) {\n"
" if ((int)gid.x >= s.size || (int)gid.y >= s.steps*s.batch) return;\n"
" int z=gid.y/s.batch;\n"
" int offset=s.offset;\n"
" float4 scale=scalesbias[z];\n"
"#ifdef BIAS\n"
" float4 bias=scalesbias[z+offset];\n"
" out[int(gid.y)*s.size+int(gid.x)] = (M4)((float4)in[int(gid.y)*s.size+int(gid.x)]*scale+bias);\n"
"#else \n"
" out[int(gid.y)*s.size+int(gid.x)] = (M4)((float4)in[int(gid.y)*s.size+int(gid.x)]*scale);\n"
"#endif \n"
"}\n"
;

MetalScale::MetalScale(Backend *backend, const Scale *scale) : MetalExecution(backend) {
    auto mtbn = static_cast<MetalBackend *>(backend);
    auto bufferAlloc = mtbn->getStaticBufferPool();
    auto context  = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto channel4 = UP_DIV(scale->channels(), 4) * 4;
    mBiasOffset = channel4 / 4;
    auto rt = (MetalRuntime*)mtbn->runtime();
    MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
    auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
    std::vector<std::string> keys = {"scale"};
    if (mtbn->useFp16InsteadFp32()) {
        [dic setValue:@"half4" forKey:@"M4"];
        keys.emplace_back("fp16");
    } else {
        [dic setValue:@"float4" forKey:@"M4"];
        keys.emplace_back("fp32");
    }
    int scaleBiasNumber = 1;
    if (nullptr != scale->biasData()) {
        [dic setValue:@"1" forKey:@"BIAS"];
        keys.emplace_back("BIAS");
        scaleBiasNumber = 2;
    }
    option.preprocessorMacros = dic;
    auto pipeline = rt->findPipeline(keys);
    if (nil == pipeline) {
        // Rebuild Pipeline
        pipeline = mtbn->makeComputePipelineWithSourceOption(shader_MetalScale_metal, "scale_ca", option);
        rt->insertPipeline(keys, pipeline);
    }
    mPipeline = pipeline;
    mScaleBias = bufferAlloc->alloc(scaleBiasNumber * channel4 * sizeof(float));
    if (mScaleBias.first == nullptr) {
        mValid = false;
        return;
    }
    auto scalePtr = MetalBackend::getMemPtr(mScaleBias);
    ::memset(scalePtr, 0, channel4 * sizeof(float));
    ::memcpy(scalePtr, scale->scaleData()->data(), scale->channels() * sizeof(float));
    if (nullptr != scale->biasData()) {
        auto biasPtr = scalePtr + channel4 * sizeof(float);
        ::memset(biasPtr, 0, channel4 * sizeof(float));
        ::memcpy(biasPtr, scale->biasData()->data(), scale->channels() * sizeof(float));
    }
    mConst = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
}
MetalScale::~MetalScale() {
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto bufferAlloc = mtbn->getStaticBufferPool();
    if (nullptr != mScaleBias.first) {
        bufferAlloc->free(mScaleBias);
    }
}

ErrorCode MetalScale::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto output = outputs[0];

    // shape
    int w   = output->width();
    int h   = output->height();
    int c   = output->channel();
    int z   = UP_DIV(c, 4);
    ((int *)mConst.contents)[0] = w*h;
    ((int *)mConst.contents)[1] = z;
    ((int *)mConst.contents)[2] = output->batch();
    ((int *)mConst.contents)[3] = mBiasOffset;
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(w*h, z * outputs[0]->batch(), 1)];
    return NO_ERROR;
}

void MetalScale::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0], output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);
    MetalBackend::setTensor(output, encoder, 1);
    [encoder setBuffer:mConst offset:0 atIndex:2];
    MetalBackend::setMem(mScaleBias, encoder, 3);
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];

}

class MetalScaleCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *>& outputs) const {
        return new MetalScale(backend, op->main_as_Scale());
    }
};
REGISTER_METAL_OP_CREATOR(MetalScaleCreator, OpType_Scale);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
