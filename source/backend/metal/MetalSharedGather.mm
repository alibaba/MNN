//
//  MetalSharedGather.mm
//  MNN

#import "backend/metal/MetalSharedGather.hpp"
#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MNNMetalContext.h"
#import "core/Macro.h"
#import "backend/metal/ConvSimdGroupShader.hpp"

#if MNN_METAL_ENABLED

namespace MNN {

// gSharedGatherQuant: directly decode int4/int8 weights and gather on-the-fly.
// Layout and dequant parameters follow conv1x1 low-memory path.
// Weight layout: [N/4, K/4, N4, K4] (packed), linear index for a pack:
//   offset = ((idx_n4 * cst.input_slice + idx_k4) * 4 + idx_nl)
//   - idx_n4 = n / 4, idx_nl = n % 4
//   - idx_k4 = k / 4, comp = k % 4
// W_QUANT_4:
//   uchar2 pack = wt[offset];
//   w0 = (pack.x >> 4) - 8; w1 = (pack.x & 15) - 8;
//   w2 = (pack.y >> 4) - 8; w3 = (pack.y & 15) - 8;
//   choose w = w{comp}.
// W_QUANT_8:
//   char4 pack = wt[offset]; choose pack.{x,y,z,w} by comp.
// Dequant scale/bias:
//   blockK4PerBi = (cst.input_slice + cst.block_size - 1) / cst.block_size;
//   bi = clamp(idx_k4 / blockK4PerBi, 0, cst.block_size-1);
//   sbIndex = idx_n4 * cst.block_size + bi;
//   scaleVec = dequantScale[2*sbIndex+0] / cst.scale_coef;  // ftype4
//   biasVec  = dequantScale[2*sbIndex+1] / cst.scale_coef;
//   out = w * scaleVec[idx_nl] + biasVec[idx_nl].
// Thread grid: 1D over all elements (selectSize * ic).
static const char* gSharedGatherQuant = R"metal(
kernel void shared_gather_quant(
    device ftype4 *wf                             [[buffer(0)]],
#ifdef W_QUANT_4
    const device uchar2 *wi                       [[buffer(1)]],
#elif defined(W_QUANT_8)
    const device char4 *wi                        [[buffer(1)]],
#else
    const device ftype4 *wi                     [[buffer(1)]],// [N/4, K/4, N4, K4]
#endif
    const device int *indices                     [[buffer(2)]],
    constant conv1x1_constants& cst               [[buffer(3)]],
    const device ftype4 *dequantScale             [[buffer(4)]],
    uint2 gid                                      [[thread_position_in_grid]]) {
    int ic = cst.input_size;
    int selectSize = cst.output_width;
    int idx_k16 = gid.y; // K/16

    int idx_k4 = idx_k16 * 4;

    if(idx_k4 >= cst.input_slice || gid.x >= selectSize) {
        return;
    }

    int idx_n = indices[gid.x]; // N

    int idx_n4 = idx_n/4;
    int idx_nl = idx_n%4;

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;


    int bi = idx_k4 / block;
    // [N/4, cst.block_size, 2/*scale_bias*/, N4]
    FLOAT scale = FLOAT(((const device ftype *)dequantScale)[((idx_n4 * cst.block_size + bi) * 2 + 0) * 4 + idx_nl]) / (FLOAT)cst.scale_coef;
    FLOAT dequant_bias = FLOAT(((const device ftype *)dequantScale)[((idx_n4 * cst.block_size + bi) * 2 + 1) * 4 + idx_nl]) / (FLOAT)cst.scale_coef;

    auto xy_wi = wi + (idx_n4 * cst.input_slice + idx_k4) * 4 + idx_nl;// [N/4, K/4, N4, K4]
    auto xy_wf = wf + (ic * gid.x + idx_k16 * 16) / 4;

    #ifdef W_QUANT_4
    for(int k = 0; k < 4; k++) {
        uchar2 w_int4 = xy_wi[4*k]; // [N/4, K/4, N4, K4]
        FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
        FLOAT4 res = w4 * scale + dequant_bias;
        xy_wf[k] = (ftype4)res;
    }
    #elif defined(W_QUANT_8)
    for(int k = 0; k < 4; k++) {
        char4 w_int4 = xy_wi[4*k]; // [N/4, K/4, N4, K4]
        FLOAT4 w4 = FLOAT4((float)w_int4[0], (float)w_int4[1], (float)w_int4[2], (float)w_int4[3]);
        FLOAT4 res = w4 * scale + dequant_bias;
        xy_wf[k] = (ftype4)res;
    }
    #endif
}
)metal";

MetalSharedGather::MetalSharedGather(Backend *backend,
                                     int oc,
                                     std::shared_ptr<Tensor> weight,
                                     std::shared_ptr<Tensor> dequantScaleBias,
                                     int dequantBits,
                                     float scaleCoef)
    : MetalExecution(backend) {
    mOc = oc;
    mWeight = std::move(weight);
    mDequantScaleBias = std::move(dequantScaleBias);
    mDequantBits = dequantBits;
    mScaleCoef = scaleCoef;
}

ErrorCode MetalSharedGather::onResize(const std::vector<Tensor *> &inputs,
                                      const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    auto input = inputs[0];   // indices tensor
    auto output = outputs[0]; // gathered weight rows

    if (nullptr == mWeight.get() || nullptr == mDequantScaleBias.get()) {
        // Only support quantized weights for SharedGather
        return NOT_SUPPORT;
    }

    // Logical sizes
    int selectSize = input->elementSize();
    int ic = output->length(output->dimensions() - 1);
    int oc = mOc;
    int oc_4 = UP_DIV(oc, 4);
    int ic_4 = UP_DIV(ic, 4);

    int bytes = backend->useFp16InsteadFp32() ? 2 : 4;
    int blockSize = 1;
    if (mDequantScaleBias.get()) {
        // Layout in MetalConvolutionCommon::getDequantScale: [alignOutputCount, blockSize, 2, 4]
        blockSize = (int)(mDequantScaleBias->usize() / bytes / oc_4 / 2 / 4);
        if (blockSize <= 0) {
            blockSize = 1;
        }
    }
    if (ic % 16 != 0) {
        MNN_PRINT("Currnetly metal shared gather don's support ic not align to 16: %d\n", ic);
        return NOT_SUPPORT;
    }

    // Prepare constant buffer shared by quant/dequant and gather kernels
    mConstBuffer = backend->getConstBuffer(sizeof(Conv1x1Constants));
    auto param = (Conv1x1Constants *)mConstBuffer.contents;
    ::memset(param, 0, sizeof(Conv1x1Constants));
    param->input_size     = ic;     // reinterpret as ic
    param->input_slice    = ic_4;   // ic_4
    param->output_width   = selectSize;
    param->output_height  = 1;
    param->output_size    = selectSize * ic;
    param->output_slice   = oc_4;
    param->output_channel = oc;
    param->batch          = 1;
    param->block_size     = blockSize;
    param->activation     = 0;
    param->scale_coef     = mScaleCoef;

    // basic macro info for fp16/fp32
    std::string ftype = "float";
    std::string ftype2 = "float2";
    std::string ftype4 = "float4";
    std::string ftype2x4 = "float2x4";
    std::string ftype4x4 = "float4x4";
    if (backend->useFp16InsteadFp32()) {
        ftype = "half";
        ftype2 = "half2";
        ftype4 = "half4";
        ftype2x4 = "half2x4";
        ftype4x4 = "half4x4";
    }

    auto baseDic = [NSMutableDictionary dictionaryWithCapacity:0];
    [baseDic setValue:@(ftype.c_str())   forKey:@"ftype"];
    [baseDic setValue:@(ftype2.c_str())  forKey:@"ftype2"];
    [baseDic setValue:@(ftype4.c_str())  forKey:@"ftype4"];
    [baseDic setValue:@(ftype2x4.c_str()) forKey:@"ftype2x4"];
    [baseDic setValue:@(ftype4x4.c_str()) forKey:@"ftype4x4"];
    [baseDic setValue:@"1" forKey:@"MNN_METAL_FLOAT32_COMPUTER"];
    if (backend->useFp16InsteadFp32()) {
        [baseDic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
    }

    MetalRuntime *rt = (MetalRuntime *)backend->runtime();
    std::string basicShaderPrefix = gBasicConvPrefix;

    // Preferred path: direct int4/int8 quant gather in shader
    mQuantPipeline = nil;

    MTLCompileOptions *optionQuant = [[MTLCompileOptions alloc] init];
    NSMutableDictionary *dic = [baseDic mutableCopy];
    std::vector<std::string> keys = {ftype4, "MNN_METAL_FLOAT32_COMPUTER", "shared_gather_quant"};
    if (mDequantBits == 4) {
        [dic setValue:@"1" forKey:@"W_QUANT_4"];
        keys.emplace_back("W_QUANT_4");
    } else {
        [dic setValue:@"1" forKey:@"W_QUANT_8"];
        keys.emplace_back("W_QUANT_8");
    }
    optionQuant.preprocessorMacros = dic;

    auto pipeline = rt->findPipeline(keys);
    if (nil == pipeline) {
        std::string shader = basicShaderPrefix + gSharedGatherQuant;
        pipeline = backend->makeComputePipelineWithSourceOption(shader.c_str(), "shared_gather_quant", optionQuant);
        rt->insertPipeline(keys, pipeline);
    }
    mQuantPipeline = pipeline;

    auto threads = MTLSizeMake((NSUInteger)selectSize, UP_DIV(ic, 16), 1);
    mQuantThreads = [context computeBestGroupAndLocal:pipeline threads:threads];

    // In int4/int8 path we do not build global dequant + blit by default.
    return NO_ERROR;
}

void MetalSharedGather::onEncode(const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs,
                                 id<MTLComputeCommandEncoder> encoder) {
    auto backend = static_cast<MetalBackend *>(this->backend());

    auto input = inputs[0];
    auto output = outputs[0];

    // Preferred path: direct quant gather
    [encoder setComputePipelineState:mQuantPipeline];
    MetalBackend::setTensor(output, encoder, 0);         // out
    MetalBackend::setTensor(mWeight.get(), encoder, 1);  // quant weight
    MetalBackend::setTensor(input, encoder, 2);          // indices
    [encoder setBuffer:mConstBuffer offset:0 atIndex:3];
    if (nullptr != mDequantScaleBias.get()) {
        MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 4); // dequantScaleBias
    }
    [encoder dispatchThreadgroups:mQuantThreads.first threadsPerThreadgroup:mQuantThreads.second];
    MNN_PRINT_ENCODER((__bridge MNNMetalContext *)backend->context(), encoder);
    return;
}

bool MetalSharedGather::onClone(Backend *bn, const Op *op, Execution **dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new MetalSharedGather(bn, mOc, mWeight, mDequantScaleBias, mDequantBits, mScaleCoef);
    return true;
}

} // namespace MNN

#endif /* MNN_METAL_ENABLED */
