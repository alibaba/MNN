//
//  MetalConvolution1x1.mm
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalConvolution1x1.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalSharedGather.hpp"
#import "ConvSimdGroupShader.hpp"

#if MNN_METAL_ENABLED

#if MNN_METAL_OP_PROFILE
#define CONV1X1_SET_TAG(name) mProfileTag = (name)
#else
#define CONV1X1_SET_TAG(name) do {} while(0)
#endif

namespace MNN {
bool MetalConvolution1x1::isValid(const Convolution2D *conv, const Tensor *input) {
    auto common = conv->common();
    auto kx = common->kernelX(), ky = common->kernelY();
    auto dx = common->dilateX(), dy = common->dilateY();
    auto sx = common->strideX(), sy = common->strideY();
    auto px = common->padX(), py = common->padY();
    return kx == 1 && ky == 1 && dx == 1 && dy == 1 && px == 0 && py == 0 && sx == 1 && sy == 1;
}

MetalConvolution1x1::MetalConvolution1x1(Backend *backend, const MNN::Op *op) : MetalConvolutionCommon(backend, op, nullptr) {
    auto conv2D = op->main_as_Convolution2D();
    bool ldInt8Weight = false;
    if(static_cast<MetalBackend*>(backend)->getMemoryMode() == BackendConfig::Memory_Low) {
        if (conv2D->quanParameter() && (conv2D->external() || conv2D->quanParameter()->buffer())) {
            // quant type equal to 3 means fp16, fallback to float weight
            if(conv2D->quanParameter()->type() != 3 && conv2D->quanParameter()->type() != 8) {
            	ldInt8Weight = true;
            }
        }
    }
    loadWeight(op, ldInt8Weight);
}

MetalConvolution1x1::MetalConvolution1x1(Backend *backend, const MNN::Op *op, std::shared_ptr<MNN::Tensor> weight, std::shared_ptr<MNN::Tensor> bias, std::shared_ptr<MNN::Tensor> dequantScale, int dequantBits, float scaleCoef) : MetalConvolutionCommon(backend, op, bias) {
    mWeight = weight;
    mBias = bias;
    mDequantScaleBias = dequantScale;
    mDequantBits = dequantBits;
    mScaleCoef = scaleCoef;
}


bool MetalConvolution1x1::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    if (op->type() == OpType_GatherV2) {
        // SharedGather path: reuse quantized weight and dequant resources
        if (!mDequantScaleBias.get() || (mDequantBits != 4 && mDequantBits != 8)) {
            // Quantized weight is required for SharedGather
            return false;
        }
        auto conv2D = mOp->main_as_Convolution2D();
        int oc = conv2D->common()->outputCount();
        *dst = new MetalSharedGather(bn, oc, mWeight, mDequantScaleBias, mDequantBits, mScaleCoef);
        MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
        return true;
    }
    *dst = new MetalConvolution1x1(bn, op, mWeight, mBias, mDequantScaleBias, mDequantBits, mScaleCoef);
    MNN_METAL_PROFILE_REGISTER_CLONE(bn, op, *dst);
    return true;
}

bool MetalConvolution1x1::setupGateUpFusion(MetalConvolution1x1* peer, const Tensor* peerOutput) {
    if (!mIs2sgDecode || !peer->mIs2sgDecode) {
        return false;
    }
    // Leader = gate (this), Follower = up (peer)
    mIsGateUpLeader = true;
    mGateUpPeer = peer;
    mGateUpPeerOutput = peerOutput;
    peer->mIsGateUpFollower = true;

    // Build fused pipeline with GATE_UP_FUSED macro
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    // Store up's scale_coef separately: gate uses cst.scale_coef (via buffer(2)),
    // but up needs its own tensor-specific coefficient. Without this, up's dequant
    // is scaled by gate's coefficient and any range mismatch drifts decode into
    // garbage on models like Qwen3.5-2B.
    mGateUpSegBuffer = backend->getConstBuffer(sizeof(float));
    ((float *)mGateUpSegBuffer.contents)[0] = peer->mScaleCoef;
    MetalRuntime* rt = (MetalRuntime *)backend->runtime();

    std::string ftype4 = backend->useFp16InsteadFp32() ? "half4" : "float4";
    std::vector<std::string> keys = {ftype4, "MNN_METAL_FLOAT32_COMPUTER"};
    if (backend->useFp16InsteadFp32()) {
        keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
    }
    if (mDequantBits == 4) {
        keys.emplace_back("conv1x1_wquant_4");
    } else if (mDequantBits == 8) {
        keys.emplace_back("conv1x1_wquant_8");
    }
    keys.emplace_back("conv1x1_wquant_sg_reduce");
    keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
    keys.emplace_back("GATE_UP_FUSED");

    mGateUpFusedPipeline = rt->findPipeline(keys);
    if (nil == mGateUpFusedPipeline) {
        std::string ftype = backend->useFp16InsteadFp32() ? "half" : "float";
        std::string ftype2 = backend->useFp16InsteadFp32() ? "half2" : "float2";
        std::string ftype2x4 = backend->useFp16InsteadFp32() ? "half2x4" : "float2x4";
        std::string ftype4x4 = backend->useFp16InsteadFp32() ? "half4x4" : "float4x4";

        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
        [dic setValue:@(ftype2.c_str()) forKey:@"ftype2"];
        [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
        [dic setValue:@(ftype2x4.c_str()) forKey:@"ftype2x4"];
        [dic setValue:@(ftype4x4.c_str()) forKey:@"ftype4x4"];
        [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT32_COMPUTER"];
        if (backend->useFp16InsteadFp32()) {
            [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
        }
        if (mDequantBits == 4) {
            [dic setValue:@"1" forKey:@"W_QUANT_4"];
        } else if (mDequantBits == 8) {
            [dic setValue:@"1" forKey:@"W_QUANT_8"];
        }
        [dic setValue:@"1" forKey:@"GATE_UP_FUSED"];
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mGateUpFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mGateUpFusedPipeline);
    }

    if (nil == mGateUpFusedPipeline) {
        // Compilation failed, revert fusion
        mIsGateUpLeader = false;
        mGateUpPeer = nullptr;
        mGateUpSegBuffer = nil;
        peer->mIsGateUpFollower = false;
        return false;
    }

    // Update grid: add z=2 dimension for gate/up selection
    auto gridSize = mThreads.first;
    mThreads.first = MTLSizeMake(gridSize.width, gridSize.height, 2);

    return true;
}

bool MetalConvolution1x1::setupQKVFusion(MetalConvolution1x1* peerK, const Tensor* peerKOutput,
                                          MetalConvolution1x1* peerV, const Tensor* peerVOutput) {
    if (!mIs2sgDecode || !peerK->mIs2sgDecode || !peerV->mIs2sgDecode) {
        return false;
    }
    // Leader = Q (this), Followers = K (peerK), V (peerV)
    mIsQKVLeader = true;
    mQKVPeerK = peerK;
    mQKVPeerV = peerV;
    mQKVPeerKOutput = peerKOutput;
    mQKVPeerVOutput = peerVOutput;
    peerK->mIsQKVFollower = true;
    peerV->mIsQKVFollower = true;

    // Build fused pipeline with QKV_FUSED macro
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    MetalRuntime* rt = (MetalRuntime *)backend->runtime();

    std::string ftype4 = backend->useFp16InsteadFp32() ? "half4" : "float4";
    std::vector<std::string> keys = {ftype4, "MNN_METAL_FLOAT32_COMPUTER"};
    if (backend->useFp16InsteadFp32()) {
        keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
    }
    if (mDequantBits == 4) {
        keys.emplace_back("conv1x1_wquant_4");
    } else if (mDequantBits == 8) {
        keys.emplace_back("conv1x1_wquant_8");
    }
    keys.emplace_back("conv1x1_wquant_sg_reduce");
    keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
    keys.emplace_back("QKV_FUSED");

    mQKVFusedPipeline = rt->findPipeline(keys);
    if (nil == mQKVFusedPipeline) {
        std::string ftype = backend->useFp16InsteadFp32() ? "half" : "float";
        std::string ftype2 = backend->useFp16InsteadFp32() ? "half2" : "float2";
        std::string ftype2x4 = backend->useFp16InsteadFp32() ? "half2x4" : "float2x4";
        std::string ftype4x4 = backend->useFp16InsteadFp32() ? "half4x4" : "float4x4";

        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
        [dic setValue:@(ftype2.c_str()) forKey:@"ftype2"];
        [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
        [dic setValue:@(ftype2x4.c_str()) forKey:@"ftype2x4"];
        [dic setValue:@(ftype4x4.c_str()) forKey:@"ftype4x4"];
        [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT32_COMPUTER"];
        if (backend->useFp16InsteadFp32()) {
            [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
        }
        if (mDequantBits == 4) {
            [dic setValue:@"1" forKey:@"W_QUANT_4"];
        } else if (mDequantBits == 8) {
            [dic setValue:@"1" forKey:@"W_QUANT_8"];
        }
        [dic setValue:@"1" forKey:@"QKV_FUSED"];
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mQKVFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mQKVFusedPipeline);
    }

    if (nil == mQKVFusedPipeline) {
        // Compilation failed, revert fusion
        mIsQKVLeader = false;
        mQKVPeerK = nullptr;
        mQKVPeerV = nullptr;
        peerK->mIsQKVFollower = false;
        peerV->mIsQKVFollower = false;
        return false;
    }

    // Compute segment info: Q groups, K groups, and output_slice for each
    auto qParam = (Param *)mConstBuffer.contents;
    auto kParam = (Param *)peerK->getConstBuffer().contents;
    auto vParam = (Param *)peerV->getConstBuffer().contents;
    int q_groups = (int)mThreads.first.width;   // UP_DIV(q_oc, 8)
    int k_groups = (int)UP_DIV(peerK->getBias()->length(0), 2); // UP_DIV(k_oc_4, 2) = UP_DIV(k_oc, 8)
    int v_groups = (int)UP_DIV(peerV->getBias()->length(0), 2);
    // Recalculate from const buffer for accuracy
    q_groups = UP_DIV(qParam->output_slice, 2);
    k_groups = UP_DIV(kParam->output_slice, 2);
    v_groups = UP_DIV(vParam->output_slice, 2);
    int total_groups = q_groups + k_groups + v_groups;

    // Create segment info buffer as float array:
    // [0]=q_groups, [1]=k_groups, [2]=k_oc_slice, [3]=v_oc_slice, [4]=k_scale_coef, [5]=v_scale_coef
    mQKVSegBuffer = backend->getConstBuffer(6 * sizeof(float));
    auto segPtr = (float *)mQKVSegBuffer.contents;
    segPtr[0] = (float)q_groups;
    segPtr[1] = (float)k_groups;
    segPtr[2] = (float)kParam->output_slice;
    segPtr[3] = (float)vParam->output_slice;
    segPtr[4] = kParam->scale_coef;
    segPtr[5] = vParam->scale_coef;

    // Save original grid before modifying (needed for fallback if buffer overlap detected)
    mQKVOriginalGrid = mThreads.first;
    // Update grid: total_groups in x, 1 in y (same as original 2sg decode)
    mThreads.first = MTLSizeMake(total_groups, 1, 1);

    return true;
}

bool MetalConvolution1x1::setupLNFusion(const Tensor* hiddenInput, const Tensor* residualInput,
                                        const Tensor* residualOutput, std::shared_ptr<Tensor> gamma, float eps) {
    if (!mIs2sgDecode) {
        return false;
    }

    mLNHiddenInput = hiddenInput;
    mLNResidualInput = residualInput;
    mLNResidualOutput = residualOutput;
    mLNGamma = gamma;
    mHasLNFusion = true;

    auto backend = static_cast<MetalBackend *>(this->backend());
    MetalRuntime* rt = (MetalRuntime *)backend->runtime();
    mLNEpsBuffer = backend->getConstBuffer(sizeof(float));
    *((float *)mLNEpsBuffer.contents) = eps;

    std::string ftype = backend->useFp16InsteadFp32() ? "half" : "float";
    std::string ftype2 = backend->useFp16InsteadFp32() ? "half2" : "float2";
    std::string ftype4 = backend->useFp16InsteadFp32() ? "half4" : "float4";
    std::string ftype2x4 = backend->useFp16InsteadFp32() ? "half2x4" : "float2x4";
    std::string ftype4x4 = backend->useFp16InsteadFp32() ? "half4x4" : "float4x4";

    std::vector<std::string> keys = {ftype4, "MNN_METAL_FLOAT32_COMPUTER"};
    if (backend->useFp16InsteadFp32()) {
        keys.emplace_back("MNN_METAL_FLOAT16_STORAGE");
    }
    if (mDequantBits == 4) {
        keys.emplace_back("conv1x1_wquant_4");
    } else if (mDequantBits == 8) {
        keys.emplace_back("conv1x1_wquant_8");
    }
    keys.emplace_back("conv1x1_wquant_sg_reduce");
    keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
    if (mIsQKVLeader) {
        keys.emplace_back("QKV_FUSED");
    } else if (mIsGateUpLeader) {
        keys.emplace_back("GATE_UP_FUSED");
    }
    keys.emplace_back("LN_FUSED");

    mLNFusedPipeline = rt->findPipeline(keys);
    if (nil == mLNFusedPipeline) {
        MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
        auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
        [dic setValue:@(ftype.c_str()) forKey:@"ftype"];
        [dic setValue:@(ftype2.c_str()) forKey:@"ftype2"];
        [dic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
        [dic setValue:@(ftype2x4.c_str()) forKey:@"ftype2x4"];
        [dic setValue:@(ftype4x4.c_str()) forKey:@"ftype4x4"];
        [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT32_COMPUTER"];
        if (backend->useFp16InsteadFp32()) {
            [dic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
        }
        if (mDequantBits == 4) {
            [dic setValue:@"1" forKey:@"W_QUANT_4"];
        } else if (mDequantBits == 8) {
            [dic setValue:@"1" forKey:@"W_QUANT_8"];
        }
        if (mIsQKVLeader) {
            [dic setValue:@"1" forKey:@"QKV_FUSED"];
        } else if (mIsGateUpLeader) {
            [dic setValue:@"1" forKey:@"GATE_UP_FUSED"];
        }
        [dic setValue:@"1" forKey:@"LN_FUSED"];
        option.preprocessorMacros = dic;

        std::string sgrWqStr = std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        mLNFusedPipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
        rt->insertPipeline(keys, mLNFusedPipeline);
    }

    if (nil == mLNFusedPipeline) {
        mHasLNFusion = false;
        return false;
    }
    return true;
}

void MetalConvolution1x1::bindLNBuffers(id<MTLComputeCommandEncoder> encoder) {
    MetalBackend::setTensor(mLNResidualInput, encoder, 20);
    MetalBackend::setTensor(mLNGamma.get(), encoder, 21);
    MetalBackend::setTensor(mLNResidualOutput, encoder, 22);
    [encoder setBuffer:mLNEpsBuffer offset:0 atIndex:23];
}

ErrorCode MetalConvolution1x1::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MetalConvolutionCommon::onResize(inputs, outputs);

    // Reset Gate/Up fusion state on each resize
    mIs2sgDecode = false;
    mIsGateUpLeader = false;
    mIsGateUpFollower = false;
    mGateUpPeer = nullptr;
    mGateUpFusedPipeline = nil;
    mGateUpSegBuffer = nil;

    // Reset QKV fusion state on each resize
    mIsQKVLeader = false;
    mIsQKVFollower = false;
    mQKVPeerK = nullptr;
    mQKVPeerV = nullptr;
    mQKVPeerKOutput = nullptr;
    mQKVPeerVOutput = nullptr;
    mQKVFusedPipeline = nil;
    mQKVSegBuffer = nil;

    mHasLNFusion = false;
    mLNFusedPipeline = nil;
    mLNHiddenInput = nullptr;
    mLNResidualInput = nullptr;
    mLNResidualOutput = nullptr;
    mLNGamma = nullptr;
    mLNEpsBuffer = nil;

    // prepare
    // For C4NHW4 format, NHW can be fuse to W
    auto input = inputs[0];
    auto output = outputs[0];
    int is = input->batch();
    for (int i=2; i<input->dimensions(); ++i) {
        is *= input->length(i);
    }
    int ic  = input->channel();
    int ic_4  = UP_DIV(input->channel(), 4);
    int ow  = is;
    int oh  = 1;
    int os  = ow;
    int ob  = 1;
    auto oc  = output->channel();
    auto oc_4  = UP_DIV(output->channel(), 4);
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    int blockSize = 1;
    if (mDequantScaleBias.get()) {
        int bytes = sizeof(float);
        if(backend->useFp16InsteadFp32()) {
            bytes = sizeof(__fp16);
        }
        blockSize = (int)(mDequantScaleBias->usize() / bytes / oc_4 / 2 / 4);
    }
    // create const buffer
    mConstBuffer = backend->getConstBuffer(sizeof(Param));
    auto param = (Param *)mConstBuffer.contents;
    param->input_size = is;
    param->input_slice = ic_4;
    param->output_width = ow;
    param->output_height = oh;
    param->output_size = os;
    param->output_slice = oc_4;
    param->output_channel = oc;
    param->batch = ob;
    param->block_size = blockSize;
    param->activation = mActivationType;
    param->scale_coef = mScaleCoef;
    int area = ob * ow * oh;
    // basic marco info
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

    MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
    auto baseDic = [NSMutableDictionary dictionaryWithCapacity:0];
    [baseDic setValue:@(ftype.c_str()) forKey:@"ftype"];
    [baseDic setValue:@(ftype2.c_str()) forKey:@"ftype2"];
    [baseDic setValue:@(ftype4.c_str()) forKey:@"ftype4"];
    [baseDic setValue:@(ftype2x4.c_str()) forKey:@"ftype2x4"];
    [baseDic setValue:@(ftype4x4.c_str()) forKey:@"ftype4x4"];
    [baseDic setValue:@"1" forKey:@"MNN_METAL_FLOAT32_COMPUTER"];
    if (backend->useFp16InsteadFp32()) {
        [baseDic setValue:@"1" forKey:@"MNN_METAL_FLOAT16_STORAGE"];
    }
    std::vector<std::string> baseKeys = {ftype4, "MNN_METAL_FLOAT32_COMPUTER"};

    MetalRuntime* rt = (MetalRuntime *)backend->runtime();
    std::string basicShaderPrefix = gBasicConvPrefix;

    // if M is small, dequant weight in shader
    // if device not support simdgroup matrix, only support dequant in shader
    bool dequantInShader = (area < 64) || !(rt->supportSimdGroupMatrix());
    // Native W_QUANT_2/3 paths are only implemented in conv1x1_gemv_g8_wquant_sg (decode,
    // area==1). For multi-token prefill we route through the outer-dequant + fp gemm
    // path instead, which has a real W_QUANT_2/3 dequant in conv1x1_w_dequant.
    // The outer-dequant path itself uses simdgroup-matrix; only override when the device
    // supports it, otherwise stay on the in-shader path.
    if ((mDequantBits == 2 || mDequantBits == 3) && area > 1 && rt->supportSimdGroupMatrix()) {
        dequantInShader = false;
    }
    // On tensor API devices (M5/A19+), prefer outer-dequant + tensor API GEMM for all
    // non-decode workloads. The tensor API path uses AI TensorCore hardware acceleration,
    // which outperforms in-shader dequant + simdgroup matrix for all GEMM sizes.
    if (backend->isSupportTensorApi() && area > 1 && (mDequantBits == 4 || mDequantBits == 8)) {
        dequantInShader = false;
    }
    // On non-tensor-API devices (M4 and below), choose in-shader vs outer-dequant
    // based on weight size. For large weights (ic*oc > 4M), in-shader dequant avoids
    // the outer-dequant double-pass (dequant Q4→fp16 then GEMM). For small weights,
    // the optimized fp GEMM kernel outperforms in-shader dequant despite the extra pass.
    // Env MNN_METAL_PREFILL_INSHADER_DEQUANT=1 forces on, =0 forces off.
    if (!backend->isSupportTensorApi() && rt->supportSimdGroupMatrix() && area > 1 &&
        (mDequantBits == 4 || mDequantBits == 8)) {
        static const int kForceInShader = []{
            const char* e = getenv("MNN_METAL_PREFILL_INSHADER_DEQUANT");
            if (e && e[0] == '1') return 1;
            if (e && e[0] == '0') return -1;
            return 0;
        }();
        if (kForceInShader == 1) {
            dequantInShader = true;
        } else if (kForceInShader == -1) {
            dequantInShader = false;
        } else if ((size_t)ic * oc > 4 * 1024 * 1024) {
            dequantInShader = true;
        }
    }
    mPreDequantWeight = false;
    mUseFusedDecode = false;

#ifdef MNN_LOW_MEMORY
    if (mDequantScaleBias.get() && dequantInShader) {
        //printf("inner dequant MNK: %d %d %d %d\n", area, oc, ic, blockSize);

        std::string sgmWqShader  = gConv1x1WqSgMatrix;
        std::string sgrWqShader  = gConv1x1WqSgReduce;

        NSMutableDictionary *dic = [baseDic mutableCopy];
        if(mDequantBits == 2) {
            [dic setValue:@"1" forKey:@"W_QUANT_2"];
        } else if(mDequantBits == 3) {
            [dic setValue:@"1" forKey:@"W_QUANT_3"];
        } else if(mDequantBits == 4) {
            [dic setValue:@"1" forKey:@"W_QUANT_4"];
        } else if(mDequantBits == 8) {
            [dic setValue:@"1" forKey:@"W_QUANT_8"];
        }
        option.preprocessorMacros = dic;

        NSUInteger gid_x = UP_DIV(ow * oh, 4);
        NSUInteger gid_y = oc_4;
        NSUInteger gid_z = ob;
        std::string name = "conv1x1_g1z4_w8";
        mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w8" fp16:backend->useFp16InsteadFp32()];

        if (mDequantBits == 2 || mDequantBits == 3 || mDequantBits == 4 || mDequantBits == 8) {
            // TODO: define short_seq more accurately
            int short_seq = 16;

            if(mDequantBits == 2) {
                baseKeys.emplace_back("conv1x1_wquant_2");
            } else if(mDequantBits == 3) {
                baseKeys.emplace_back("conv1x1_wquant_3");
            } else if(mDequantBits == 4) {
                baseKeys.emplace_back("conv1x1_wquant_4");
            } else if(mDequantBits == 8) {
                baseKeys.emplace_back("conv1x1_wquant_8");
            }
            if(rt->supportSimdGroupReduce() && area <= short_seq) {
                baseKeys.emplace_back("conv1x1_wquant_sg_reduce");

                std::string sgrWqStr = basicShaderPrefix + sgrWqShader;
                if(area > 1) {
                    auto keys = baseKeys;
                    int piece = 1;
                    // memory bound not so seriously, can add more thread to reduce computation in each thread
                    float ratio = 1.0 * ic_4 / 2048.0 * oc / 2048.0;
                    bool heavyMemory = ratio > 1.0;
                    if(area > 5 && !heavyMemory) {
                        if(area % 2 != 0) {
                            keys.emplace_back("MNN_METAL_SRC_PROTECT");
                            [dic setValue:@"1" forKey:@"MNN_METAL_SRC_PROTECT"];;
                            option.preprocessorMacros = dic;
                        }
                        area = UP_DIV(area, 2);
                        piece = 2;
                    }
//                    MNN_PRINT("Conv1x1 Oc:%d Ic:%d\n", oc, ic_4*4);
                    std::string kernel_name = "conv1x1_gemv_g4m" + std::to_string(area) + "_wquant_sg";
                    keys.emplace_back(kernel_name);
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), kernel_name.c_str(), option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 4), piece, 1), MTLSizeMake(32, 1, 1));
                } else if(mDequantBits != 2 && mDequantBits != 3 && oc > 16384 && oc_4 % 2 == 0) {
                    // g16 path not extended for W_QUANT_2/3, fall back to g8.
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemv_g16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 16), area, 1), MTLSizeMake(64, 1, 1));
                } else if(mDequantBits != 2 && mDequantBits != 3 && area == 1) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemv_g4m1_2sg_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g4m1_2sg_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    // 2 simdgroups per threadgroup, each handles 4 OC independently
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), 1, 1), MTLSizeMake(64, 1, 1));
                    mIs2sgDecode = true;
                    // Register this Conv1x1 for Gate/Up fusion lookup
                    backend->registerConv1x1ForOutput(output, this);
                    // Register for QKV fusion grouping (by input tensor)
                    backend->registerConv1x1ForQKV(input, this, output, oc);
                } else {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemv_g8_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), "conv1x1_gemv_g8_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
//                    MNN_PRINT("g8  ic: %d oc: %d\n", input->channel(), oc);
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 8), area, 1), MTLSizeMake(128, 1, 1));
                }
                return NO_ERROR;
            } else if(rt->supportSimdGroupMatrix()  && area > short_seq && oc > 8 && (ic_4 % 8 == 0 || ic_4 % 2 == 0)) {
                baseKeys.emplace_back("conv1x1_wquant_sg_matrix");

                std::string sgmWqStr = basicShaderPrefix + sgmWqShader;

                // Generally threadgroup memory >= 16KB
                auto smem_size = [[context device] maxThreadgroupMemoryLength];
                // choose different tile for different computation
                if(ic_4 % 8 != 0) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_8x16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_8x16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 8), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
                } else if(area >= 128 && oc >= 512 && area * oc > 512 * 2048 && smem_size >= 8192) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_32x64_wquant_split_k_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_32x64_wquant_split_k_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 64), 1), MTLSizeMake(128, 1, 1));

                } else if(area >= 32 && area * oc > 128 * 2048) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_32x16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_32x16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
                } else if(oc > 512 && area * oc > 128 * 2048) {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_16x32_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_16x32_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 16), UP_DIV(oc, 32), 1), MTLSizeMake(32, 1, 1));
                } else if(area < 16) {
                    // TODO: define useMatrix more accurate
                    bool useMatrix = area > 6 && oc > 2048 && ic*2 < oc;
                    if(useMatrix) {
                        auto keys = baseKeys;
                        int oc_block = (oc > 4096) ? 32 : 16;
                        std::string kernel_name = "conv1x1_gemm_8x" + std::to_string(oc_block) + "_wquant_sg";

                        keys.emplace_back(kernel_name);
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), kernel_name.c_str(), option);
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 8), UP_DIV(oc, oc_block), 1), MTLSizeMake(32, 1, 1));
                    } else {
                        std::string sgrWqStr = basicShaderPrefix + sgrWqShader;

                        auto keys = baseKeys;
                        std::string kernel_name = "conv1x1_gemv_g4m" + std::to_string(area) + "_wquant_sg";
                        keys.emplace_back(kernel_name);
                        auto pipeline = rt->findPipeline(keys);
                        if (nil == pipeline) {
                            pipeline = backend->makeComputePipelineWithSourceOption(sgrWqStr.c_str(), kernel_name.c_str(), option);
                            rt->insertPipeline(keys, pipeline);
                        }
                        mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                        mThreads = std::make_pair(MTLSizeMake(UP_DIV(oc, 4), 1, 1), MTLSizeMake(32, 1, 1));
                    }
                } else {
                    auto keys = baseKeys;
                    keys.emplace_back("conv1x1_gemm_16x16_wquant_sg");
                    auto pipeline = rt->findPipeline(keys);
                    if (nil == pipeline) {
                        pipeline = backend->makeComputePipelineWithSourceOption(sgmWqStr.c_str(), "conv1x1_gemm_16x16_wquant_sg", option);
                        rt->insertPipeline(keys, pipeline);
                    }
                    mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
//                                    MNN_PRINT("gemm M: %d N: %d\n", area, oc);
                    mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 16), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
                }
                return NO_ERROR;
            } else if(mDequantBits == 4) {
                mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w4" fp16:backend->useFp16InsteadFp32()];
                name = "conv1x1_g1z4_w4";
            } else {
                // mDequantBits == 8
                mPipeline = [context pipelineWithName:@"conv1x1_g1z4_w8" fp16:backend->useFp16InsteadFp32()];
                name = "conv1x1_g1z4_w8";
            }
        } else {
            MNN_ERROR("metal conv weight quant not support %d bits yet!\n", mDequantBits);
        }
        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(),
                        ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(),
                        (((MetalRuntimeAllocator::MetalBufferAlloc *)mDequantScaleBias->deviceId()))->getBuffer(),
                        nil];
        const Tensor* weight = mWeight.get();
        const Tensor* bias = mBias.get();
        int buffer_offset[] = {
            TensorUtils::getDescribeOrigin(input)->offset,
            TensorUtils::getDescribeOrigin(output)->offset,
            0,
            TensorUtils::getDescribeOrigin(weight)->offset,
            TensorUtils::getDescribeOrigin(bias)->offset,
            TensorUtils::getDescribeOrigin(mDequantScaleBias.get())->offset,
            0};

        MetalRuntime *rt = (MetalRuntime *)backend->runtime();
        auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets:buffer_offset  queue:backend->queue()];
        mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
        CONV1X1_SET_TAG(name);
        return NO_ERROR;
    }
#endif

    std::string sgmWfpShader = gConv1x1WfpSgMatrix;
    std::string sgrWfpShader = gConv1x1WfpSgReduce;

    // Dequant using single shader
    if (mDequantScaleBias.get()) {
        baseKeys.emplace_back("conv1x1_dequant_weight_outter");
        std::string sgmWfpStr = basicShaderPrefix + sgmWfpShader;

        mPreDequantWeight = true;
        {
            NSMutableDictionary *dic = [baseDic mutableCopy];

            auto keys = baseKeys;
            keys.emplace_back("conv1x1_w_dequant");
            if(mDequantBits == 2) {
                [dic setValue:@"1" forKey:@"W_QUANT_2"];
                keys.emplace_back("W_QUANT_2");
            } else if(mDequantBits == 3) {
                [dic setValue:@"1" forKey:@"W_QUANT_3"];
                keys.emplace_back("W_QUANT_3");
            } else if(mDequantBits == 4) {
                [dic setValue:@"1" forKey:@"W_QUANT_4"];
                keys.emplace_back("W_QUANT_4");
            } else if(mDequantBits == 8) {
                [dic setValue:@"1" forKey:@"W_QUANT_8"];
                keys.emplace_back("W_QUANT_8");
            }
            if(ic % 16 != 0) {
                [dic setValue:@"1" forKey:@"W_ALIGN_K16_PROTECT"];
                keys.emplace_back("W_ALIGN_K16_PROTECT");
            }
            option.preprocessorMacros = dic;

            int bytes = backend->useFp16InsteadFp32() ? 2 : 4;
            // accquire space
            mTempWeight.reset(Tensor::createDevice<uint8_t>(std::vector<int>{ROUND_UP(oc, 4) * ROUND_UP(ic, 32) * bytes}));
            backend->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);
            backend->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);

            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_w_dequant", option);
                rt->insertPipeline(keys, pipeline);
            }
            mDequantPipeline = pipeline;

            mDequantThreads = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(UP_DIV(oc, 1),  UP_DIV(ic, 16), 1)];
        }

        {
            auto keys = baseKeys;
            keys.emplace_back("conv1x1_gemm_32x64_split_k_sg");

            NSMutableDictionary *dic = [baseDic mutableCopy];
            if (ic_4 % 8 != 0) {
                [dic setValue:@"1" forKey:@"MNN_METAL_SRC_PROTECT"];
                keys.emplace_back("MNN_METAL_SRC_PROTECT");
            }
            if(backend->isSupportTensorApi() == true) {
                [dic setValue:@"1" forKey:@"USE_METAL_TENSOR_OPS"];
                keys.emplace_back("USE_METAL_TENSOR_OPS");
                if(ic > oc && ic > 2048 && (ic / blockSize) % 64 == 0) {
                    [dic setValue:@"1" forKey:@"LOOP_K64"];
                    keys.emplace_back("LOOP_K64");
                }
            }
            option.preprocessorMacros = dic;

            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_gemm_32x64_split_k_sg", option);
                rt->insertPipeline(keys, pipeline);
            }
            mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
            mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 64), 1), MTLSizeMake(128, 1, 1));
            //printf("out dequant MNK: %d %d %d %d\n", area, oc, ic, blockSize);
        }

        return NO_ERROR;
    }

    option.preprocessorMacros = baseDic;

    if(rt->supportSimdGroupMatrix()) {
        std::string sgmWfpStr = basicShaderPrefix + sgmWfpShader;

        baseKeys.emplace_back("conv1x1_float_sg_matrix");
        // total computation not too small
        if(area >= 16 && ic_4 >= 4 && ic_4 % 2 == 0 && oc_4 >= 4 && area * ic_4 * oc_4 >= 64 * 64 * 64) {
            // Enough threads
            if(area * oc_4 / ic_4 >= 1024) {
                auto keys = baseKeys;
                keys.emplace_back("conv1x1_gemm_32x16_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_gemm_32x16_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 32), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
            } else {
                auto keys = baseKeys;
                keys.emplace_back("conv1x1_gemm_16x16_sg");
                auto pipeline = rt->findPipeline(keys);
                if (nil == pipeline) {
                    pipeline = backend->makeComputePipelineWithSourceOption(sgmWfpStr.c_str(), "conv1x1_gemm_16x16_sg", option);
                    rt->insertPipeline(keys, pipeline);
                }
                mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
                mThreads = std::make_pair(MTLSizeMake(UP_DIV(area, 16), UP_DIV(oc, 16), 1), MTLSizeMake(32, 1, 1));
            }
            return NO_ERROR;
        }
    }
    if(rt->supportSimdGroupReduce()) {
        std::string sgrWfpStr = basicShaderPrefix + sgrWfpShader;

        baseKeys.emplace_back("conv1x1_float_sg_reduce");
        // do input_channel reduce
        auto magic_num = 4.0; // total threads pretty small and loop pretty large
        if(ic_4 >= 32 && ic_4 % 2 == 0 && 1.0 * area * oc_4 / ic_4 < magic_num) {
            auto keys = baseKeys;
            keys.emplace_back("conv1x1_z4_sg");
            auto pipeline = rt->findPipeline(keys);
            if (nil == pipeline) {
                pipeline = backend->makeComputePipelineWithSourceOption(sgrWfpStr.c_str(), "conv1x1_z4_sg", option);
                rt->insertPipeline(keys, pipeline);
            }
            mPipeline = pipeline; CONV1X1_SET_TAG(keys.back());
            mThreads = std::make_pair(MTLSizeMake(ow * oh, oc_4, ob), MTLSizeMake(32, 1, 1));
            return NO_ERROR;
        }
    }
//    printf("lora: %d %d %d %d %d\n", ob, oh, ow, oc, input->channel());
    if(rt->getTuneLevel() == Never) {
        if (ow * oh >= 128) {
            NSUInteger gid_x = UP_DIV(ow * oh, 8);
            NSUInteger gid_y = oc_4;
            NSUInteger gid_z = ob;

            mPipeline = [context pipelineWithName:@"conv1x1_g1z8" fp16:backend->useFp16InsteadFp32()];

            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];

            const Tensor* weight = mWeight.get();
            const Tensor* bias = mBias.get();
            int buffer_offset[] = {TensorUtils::getDescribeOrigin(input)->offset, TensorUtils::getDescribeOrigin(output)->offset, 0, TensorUtils::getDescribeOrigin(weight)->offset, TensorUtils::getDescribeOrigin(bias)->offset, 0};
            std::string name = "conv1x1_g1z8";
            MetalRuntime *rt = (MetalRuntime *)backend->runtime();
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets: buffer_offset queue:backend->queue()];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            CONV1X1_SET_TAG(name);
        } else {
            NSUInteger gid_x = UP_DIV(ow * oh, 4);
            NSUInteger gid_y = oc_4;
            NSUInteger gid_z = ob;

            mPipeline = [context pipelineWithName:@"conv1x1_g1z4" fp16:backend->useFp16InsteadFp32()];

            NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                            (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                            mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
            const Tensor* weight = mWeight.get();
            const Tensor* bias = mBias.get();
            int buffer_offset[] = {TensorUtils::getDescribeOrigin(input)->offset, TensorUtils::getDescribeOrigin(output)->offset, 0,  TensorUtils::getDescribeOrigin(weight)->offset, TensorUtils::getDescribeOrigin(bias)->offset, 0};
            std::string name = "conv1x1_g1z4";
            MetalRuntime *rt = (MetalRuntime *)backend->runtime();
            auto ret = [context getGridAndThreadgroup:mPipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets: buffer_offset queue:backend->queue()];
            mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            CONV1X1_SET_TAG(name);
            //printf("conv1x1_z4, %d %d %d %d\n", ow, oh, oc_4, ic_4);
        }
    } else {
        NSString* shaderName[] = {@"conv1x1_g1z8", @"conv1x1_g1z4", @"conv1x1_w4h4",  @"conv1x1_w2c2", @"conv1x1_w4c2"};
        int itemW[] = {8, 4, 16, 2, 4};
        int itemC[] = {4, 4, 4, 8, 8};
        int actual_kernel = 5;
        if (oc_4 % 2 != 0) {
            // Don't unrool c for avoid memory exceed
            actual_kernel = 3;
        }
        std::pair<NSUInteger, int> min_cost(INT_MAX, 0);//(min_time, min_index)

        NSArray *arr = [NSArray arrayWithObjects:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer(),
                        (id<MTLBuffer>)(((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId()))->getBuffer(),
                        mConstBuffer, (((MetalRuntimeAllocator::MetalBufferAlloc *)mWeight->deviceId()))->getBuffer(), ((MetalRuntimeAllocator::MetalBufferAlloc *)mBias->deviceId())->getBuffer(), nil];
        const Tensor* weight = mWeight.get();
        const Tensor* bias = mBias.get();
        int buffer_offset[] = {TensorUtils::getDescribeOrigin(input)->offset, TensorUtils::getDescribeOrigin(output)->offset, 0, TensorUtils::getDescribeOrigin(weight)->offset, TensorUtils::getDescribeOrigin(bias)->offset, 0};

        for(int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
            id<MTLComputePipelineState> pipeline = [context pipelineWithName:shaderName[knl_idx] fp16:backend->useFp16InsteadFp32()];
            NSUInteger gid_x = UP_DIV(ow, itemW[knl_idx]);
            NSUInteger gid_y = UP_DIV(oc, itemC[knl_idx]);
            NSUInteger gid_z = 1;

            std::string name = [shaderName[knl_idx] UTF8String];
            auto ret = [context getGridAndThreadgroup:pipeline gid:MTLSizeMake(gid_x, gid_y, gid_z) loop:10 buffer:arr runtime:rt shaderName:name offsets:buffer_offset queue:backend->queue()];

            if(min_cost.first > std::get<2>(ret)) {
                min_cost.first = std::get<2>(ret);
                min_cost.second = knl_idx;
                mThreads = std::make_pair(std::get<0>(ret), std::get<1>(ret));
            }
            //printf("conv1x1 idx:%d, global:%d %d %d, local:%d %d %d, min_cost:%d\n", knl_idx, (int)retTune.second.first.width, (int)retTune.second.first.height, (int)retTune.second.first.depth, (int)retTune.second.second.width, (int)retTune.second.second.height, (int)retTune.second.second.depth, (int)retTune.first);
        }
        //printf("conv1x1 idx:%d, min_cost:%d\n", (int)min_cost.second, (int)min_cost.first);
        mPipeline = [context pipelineWithName:shaderName[min_cost.second] fp16:backend->useFp16InsteadFp32()];
        CONV1X1_SET_TAG(std::string([shaderName[min_cost.second] UTF8String]));
    }

    return NO_ERROR;
}

void MetalConvolution1x1::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    // Gate/Up follower: the leader already dispatched this projection
    if (mIsGateUpFollower) {
        return;
    }
    // QKV follower: the leader already dispatched Q/K/V via fused pipeline
    if (mIsQKVFollower) {
        return;
    }

#if MNN_METAL_OP_PROFILE
    // Report kernel-variant tag so the profile output can distinguish shader paths
    // (e.g. Convolution/gemm_32x64_split_k_sg vs Convolution/gemv_g4m1_2sg_wquant_sg).
    {
        std::string subtag = mProfileTag;
        if (mIsGateUpLeader) {
            subtag = "gate_up_fused_" + subtag;
        } else if (mIsQKVLeader) {
            subtag = "qkv_leader_" + subtag;
        } else if (mPreDequantWeight) {
            subtag = "outdeq+" + subtag;
        }
        static_cast<MetalBackend *>(backend())->setProfileSubtag(subtag);
    }
#endif

    auto input = inputs[0];
    auto output = outputs[0];

    // QKV leader: dispatch Q/K/V in a single fused dispatch
    if (mIsQKVLeader && mQKVPeerK && mQKVPeerV && nil != (mHasLNFusion ? mLNFusedPipeline : mQKVFusedPipeline) && mQKVPeerKOutput && mQKVPeerVOutput) {
        [encoder setComputePipelineState:(mHasLNFusion ? mLNFusedPipeline : mQKVFusedPipeline)];
        // buffer(0): input (shared by Q/K/V) — with LN fusion, use hidden input
        {
            auto inTensor = mHasLNFusion ? mLNHiddenInput : input;
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)inTensor->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(inTensor)->offset atIndex:0];
        }
        // buffer(1): Q output
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
        // buffer(2): Q const params
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        // buffers(3-5): Q weight, bias, dequant
        MetalBackend::setTensor(mWeight.get(), encoder, 3);
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 5);
        // buffers(6-9): K output, weight, bias, dequant
        MetalBackend::setTensor(mQKVPeerKOutput, encoder, 6);
        MetalBackend::setTensor(mQKVPeerK->getWeight().get(), encoder, 7);
        MetalBackend::setTensor(mQKVPeerK->getBias().get(), encoder, 8);
        MetalBackend::setTensor(mQKVPeerK->getDequantScale().get(), encoder, 9);
        // buffers(10-13): V output, weight, bias, dequant
        MetalBackend::setTensor(mQKVPeerVOutput, encoder, 10);
        MetalBackend::setTensor(mQKVPeerV->getWeight().get(), encoder, 11);
        MetalBackend::setTensor(mQKVPeerV->getBias().get(), encoder, 12);
        MetalBackend::setTensor(mQKVPeerV->getDequantScale().get(), encoder, 13);
        // buffer(14): segment info
        [encoder setBuffer:mQKVSegBuffer offset:0 atIndex:14];
        if (mHasLNFusion) {
            bindLNBuffers(encoder);
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        return;
    }

    // Gate/Up leader: dispatch fused kernel covering both gate and up projections
    if (mIsGateUpLeader && mGateUpPeer && nil != (mHasLNFusion ? mLNFusedPipeline : mGateUpFusedPipeline) && mGateUpPeerOutput) {
        [encoder setComputePipelineState:(mHasLNFusion ? mLNFusedPipeline : mGateUpFusedPipeline)];
        // buffer(0): input (shared by gate and up) — with LN fusion, use hidden input
        {
            auto inTensor = mHasLNFusion ? mLNHiddenInput : input;
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)inTensor->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(inTensor)->offset atIndex:0];
        }
        // buffer(1): gate output (this)
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
        // buffer(2): gate params (also used by up since dimensions are identical)
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        // buffer(3): gate weight
        MetalBackend::setTensor(mWeight.get(), encoder, 3);
        // buffer(4): gate bias
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        // buffer(5): gate dequant scale
        MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 5);
        // buffer(6): up output
        MetalBackend::setTensor(mGateUpPeerOutput, encoder, 6);
        // buffer(7): up weight
        MetalBackend::setTensor(mGateUpPeer->getWeight().get(), encoder, 7);
        // buffer(8): up bias
        MetalBackend::setTensor(mGateUpPeer->getBias().get(), encoder, 8);
        // buffer(9): up dequant scale
        MetalBackend::setTensor(mGateUpPeer->getDequantScale().get(), encoder, 9);
        // buffer(14): {up_scale_coef} - per-tensor coefficient used by up branch
        [encoder setBuffer:mGateUpSegBuffer offset:0 atIndex:14];
        if (mHasLNFusion) {
            bindLNBuffers(encoder);
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        return;
    }

    if(mPreDequantWeight) {
#if MNN_METAL_OP_PROFILE
        // In profile mode, split the two sub-passes (weight dequant + gemm) into
        // independent command buffers so each shows up as its own profile row.
        static_cast<MetalBackend*>(backend())->setProfileSubtag("outdeq_wdq");
#endif
        // pre dequant weight pipeline
        {
            [encoder setComputePipelineState:mDequantPipeline];
            MetalBackend::setTensor(mWeight.get(), encoder, 0);
            MetalBackend::setTensor(mTempWeight.get(), encoder, 1);
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 3);
            [encoder dispatchThreadgroups:mDequantThreads.first threadsPerThreadgroup:mDequantThreads.second];
        }
#if MNN_METAL_OP_PROFILE
        {
            auto* mtbn = static_cast<MetalBackend*>(backend());
            mtbn->flushEncoder();
            mtbn->commit_net();
            mtbn->setProfileSubtag(std::string("outdeq_gemm_") + mProfileTag);
            encoder = mtbn->encoder_for_net();
        }
#endif
        // convolution pipeline
        {
            [encoder setComputePipelineState:mPipeline];
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(input)->offset atIndex:0];
            [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
            [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
            MetalBackend::setTensor(mTempWeight.get(), encoder, 3);
            MetalBackend::setTensor(mBias.get(), encoder, 4);
            MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 5);
            [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
        }
    } else if (mUseFusedDecode) {
        // Fused weight+scale decode path: single buffer contains interleaved scale/bias/weights
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(input)->offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        MetalBackend::setTensor(mFusedWeightScale.get(), encoder, 3);
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    } else {
        [encoder setComputePipelineState:mPipeline];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)input->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(input)->offset atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)((MetalRuntimeAllocator::MetalBufferAlloc *)output->deviceId())->getBuffer() offset:TensorUtils::getDescribeOrigin(output)->offset atIndex:1];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        MetalBackend::setTensor(mWeight.get(), encoder, 3);
        MetalBackend::setTensor(mBias.get(), encoder, 4);
        if (mDequantScaleBias) {
            MetalBackend::setTensor(mDequantScaleBias.get(), encoder, 5);
        }
        [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
    }
#ifdef MNN_METAL_DEBUG_INFO
    if(!static_cast<MetalBackend*>(backend())->useFp16InsteadFp32()) {
        {
            static_cast<MetalBackend*>(backend())->flushEncoder();
            static_cast<MetalBackend*>(backend())->commit_net();
            static_cast<MetalBackend*>(backend())->wait();

            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(input);
            auto ptr = (float*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%f ", ptr[i]);
            }
            printf("\n\n");
        }
        {
            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(mWeight.get());
            auto ptr = (int8_t*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%d ", ptr[i]);
            }
            printf("\n\n");
        }
        {
            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(mDequantScaleBias.get());
            auto ptr = (float*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%f ", ptr[i]);
            }
            printf("\n\n");
        }

        {
            auto buffer = static_cast<MetalBackend*>(backend())->getBuffer(output);
            auto ptr = (float*)((int8_t*)buffer.first.contents + buffer.second);
            for(int i=0; i<64; i++) {
                printf("%f ", ptr[i]);
            }
            printf("\n\n");
        }
    }
#endif
}
} // namespace MNN
#endif /* MNN_METAL_ENABLED */