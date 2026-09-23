//  ConvInt8TiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvInt8TiledExecutor.hpp"
#include "ConvolutionTiledExecutor.hpp"
#include "core/Macro.h"
#include "core/BufferAllocator.hpp"
#include "SharedGather.hpp"

#include <atomic>
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "CPUExtension.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "half.hpp"

#define WEIGHT_ONLINE_REORDER 8
namespace MNN {

// The fp16-metadata pack/execute paths inherit the caller's FPCR; reject anything but the default
// (AHP/FZ16/FZ/DN/RMode would silently corrupt the fp16 conversions the kernels rely on).
static bool _compactFpEnvironment() {
#if defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr == 0;
#else
    return false;
#endif
}

static bool _compactCanRebuild(const Convolution2D* conv) {
    if (!conv || !conv->external() || conv->external()->size() < 3 ||
        conv->external()->Get(1) <= 0 || conv->external()->Get(2) <= 0) {
        return false;
    }
    if (conv->symmetricQuan() && conv->symmetricQuan()->bias() && conv->symmetricQuan()->bias()->size() > 0) {
        return false;
    }
    if (conv->external()->size() > 3) {
        const auto biasBytes = conv->external()->Get(3);
        const auto wanted = static_cast<int64_t>(conv->common()->outputCount()) * sizeof(float);
        if (biasBytes != 0 && biasBytes != wanted) {
            return false;
        }
        if (biasBytes == wanted) {
            return true;
        }
    }
    // Rearrangement drops inline bias; without external bias only positive zero is exactly recoverable.
    if (conv->bias()) {
        for (auto value : *conv->bias()) {
            uint32_t bits;
            ::memcpy(&bits, &value, sizeof(bits));
            if (bits != 0) {
                return false;
            }
        }
    }
    return true;
}

static bool _compactResourceCompatible(Backend* backend, const CPUConvolution::ResourceInt8& resource) {
    if (!backend || backend->type() != MNN_FORWARD_CPU_EXTENSION) {
        return false;
    }
    auto cpu = static_cast<CPUBackend*>(backend);
    const auto core = cpu->functions();
    const auto functions = cpu->int8GemmFunctions();
    const auto& hint = cpu->getRuntime()->hint();
    // A persistent mmap weight pool (useCachedMmap>=1) is trusted verbatim by later runs whose
    // weight load is skipped, so they cannot reproduce the compact layout decision; the packed
    // resource is only reusable when this run could pack the compact layout itself.
    if (!(core->bytes == 2 && core->pack == 8 && core->supportFp16arith && functions &&
          cpu->threadNumber() == resource.mCompactThreads && core->smeCoreNumber == resource.mCompactSmeCores &&
          hint.dynamicQuantOption == resource.mCompactDynamicQuantOption &&
          hint.divisionRatio == resource.mCompactDivisionRatio &&
          (hint.useCachedMmap == 0 || hint.weightMemoryPath.empty()))) {
        return false;
    }
    if (resource.mCompactNeonKernel != nullptr) {
        // SME2 mixed-kernel packing: DecodeMax on SME cores, ARMV82 NEON on the branch.
        if (resource.mWeightBits == 8) {
            return core->supportSME2 && core->supportSDot &&
                   functions->MNNGemmInt8AddBiasScale_Unit_FP16_DecodeMax == resource.mCompactSmeKernel &&
                   core->arm82MatmulRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP16 ==
                       resource.mCompactNeonKernel;
        }
        return core->supportSME2 && core->supportSDot &&
               functions->MNNGemmInt8AddBiasScale_w4_Unit_FP16_DecodeMax == resource.mCompactSmeKernel &&
               core->arm82MatmulRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16 ==
                   resource.mCompactNeonKernel;
    }
    // Single-kernel packing (i8mm or sdot): one kernel expands the fp16 metadata in every
    // tile path. i8mm implies sdot, and the pointer equality pins the exact kernel family.
    if (resource.mWeightBits == 8) {
        return core->supportSDot && functions->MNNGemmInt8AddBiasScale_Unit_FP16 == resource.mCompactSmeKernel;
    }
    return core->supportSDot && functions->MNNGemmInt8AddBiasScale_w4_Unit_FP16 == resource.mCompactSmeKernel;
}

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Op* op)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend) {}

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ResourceInt8> res)
    : CPUConvolution(op->main_as_Convolution2D()->common(), backend), mResourceInt8(res) {
    if (!res->mDynamicQuant) {
        mMutableResource.reset(new MutableResourceInt8(res, backend));
        mValid = mMutableResource->mValid;
    }
}

ConvInt8TiledExecutor::~ConvInt8TiledExecutor() {
    // Do nothing
}

bool ConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    return false;
}

ErrorCode ConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (nullptr != mMutableResource) {
        mMutableResource->updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]),
                                                 TensorUtils::getQuantInfo(outputs[0]));
    }
    CPUConvolution::onResize(inputs, outputs);
    ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY,
                                                 static_cast<CPUBackend*>(backend())->functions(),
                                                 static_cast<CPUBackend*>(backend())->int8Functions());
    return NO_ERROR;
}

void ConvInt8TiledExecutor::initializeConvInt8QuantInfo(std::shared_ptr<CPUConvolution::ResourceInt8>& resourceInt8,
                                                        const Convolution2D* conv2D,
                                                        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon) {
    // input/output scale&zeorpoint
    if (conv2D->symmetricQuan()) {
        resourceInt8->mWeightBits = conv2D->symmetricQuan()->nbits();
    }
    if (conv2D->bias() && (conv2D->quanParameter()->alpha() || quanCommon->alpha.get() || quanCommon->alphaHalf.get())) {
        resourceInt8->mUseConvQuan = false;
    }
    resourceInt8->mInputZeroPoint = 0;
    resourceInt8->mOutputZeroPoint = 0;
    resourceInt8->mClampMin = -128;
    resourceInt8->mClampMax = 127;
    if (conv2D->symmetricQuan()) {
        resourceInt8->mInputZeroPoint = conv2D->symmetricQuan()->zeroPoint();
        resourceInt8->mOutputZeroPoint = conv2D->symmetricQuan()->outputZeroPoint();
        resourceInt8->mClampMin = conv2D->symmetricQuan()->clampMin();
        resourceInt8->mClampMax = conv2D->symmetricQuan()->clampMax();
    }
    if (conv2D->quanParameter() != nullptr) {
        resourceInt8->mInputScale = conv2D->quanParameter()->scaleIn();
        resourceInt8->mOutputScale = conv2D->quanParameter()->scaleOut();
    }
    resourceInt8->mRelu = conv2D->common()->relu() || conv2D->common()->relu6();
    if (conv2D->symmetricQuan() && conv2D->symmetricQuan()->outputDataType() == MNN::DataType_DT_FLOAT) {
        resourceInt8->mOutputZeroPoint = 0;
        resourceInt8->mOutputScale = 1.0f;
    }
}

void ConvInt8TiledExecutor::reorderWeight(uint8_t* dst, const uint8_t* src, int32_t* info, int32_t initval,
                                          float* kernelsum, weightSummerFuncion summerFunc) {
    // weight shape = {UP_DIV(oc, UNIT), blockNum, kernelCount* UP_DIV(ic / blockNum, SRC_UNIT), UNIT, SRC_UNIT};
    MNN_ASSERT(dst != nullptr && src != nullptr);

    int blockNum = info[0];
    int oc = info[1];
    int ic = info[2];
    int kernelCount = info[3];
    int UNIT = info[4];
    int SRC_UNIT = info[5];

    int blockL = UP_DIV(ic / blockNum, SRC_UNIT) * kernelCount;
    int stride0 = blockNum * SRC_UNIT * blockL * UNIT; // weight->stride(0)
    int stride1 = blockL * SRC_UNIT * UNIT;            // weight->stride(1)
    int stride2 = UNIT * SRC_UNIT;                     // weight->stride(2)
    int weightlen = stride0 * UP_DIV(oc, UNIT);
    memset(dst, initval, weightlen);

    auto hU = UP_DIV(oc, UNIT);
    auto lU = UP_DIV(ic / blockNum, SRC_UNIT) * kernelCount;
    bool fast = (kernelCount == 1 && ROUND_UP(oc, UNIT) == oc && (ic % (blockNum * SRC_UNIT)) == 0);
    if (fast) {
        for (int i = 0; i < hU; ++i) {
            for (int k = 0; k < UNIT; ++k) {
                for (int bl = 0; bl < blockNum; ++bl) {
                    for (int j = 0; j < blockL; ++j) {
                        int srcindex = (i * UNIT + k) * ic + bl * (lU * SRC_UNIT) + j * SRC_UNIT;
                        int dstindex = i * stride0 + bl * stride1 + j * stride2 + k * SRC_UNIT;
                        memcpy(dst + dstindex, src + srcindex, SRC_UNIT);
                    }
                }
            }
        }
    } else {
        auto blockic = ic / blockNum;
        auto bklU = UP_DIV(blockic, SRC_UNIT);
        for (int i_hU = 0; i_hU < hU; ++i_hU) {
            int dst_i_hU = i_hU * stride0;
            for (int bk = 0; bk < blockNum; ++bk) {
                int dst_bk = dst_i_hU + bk * stride1;
                int src_bk = bk * blockic * kernelCount;
                for (int k2 = 0; k2 < kernelCount; ++k2) {
                    int dst_k2 = dst_bk + k2 * bklU * stride2;
                    int src_k2 = src_bk + k2;
                    for (int blu = 0; blu < bklU; ++blu) {
                        int dst_blu = dst_k2 + blu * stride2;
                        int src_blu = src_k2 + blu * SRC_UNIT * kernelCount;
                        for (int inId = 0; inId < UNIT; ++inId) {
                            int i = i_hU * UNIT + inId;
                            if (i >= oc)
                                continue;
                            int dst_inId = dst_blu + inId * SRC_UNIT;
                            int src_inId = src_blu + i * ic * kernelCount;

                            for (int blp = 0; blp < SRC_UNIT; ++blp) {
                                int j_in_block = blu * SRC_UNIT + blp;
                                if (j_in_block >= blockic)
                                    continue;

                                int dstindex = dst_inId + blp;
                                int srcindex = src_inId + blp * kernelCount;
                                dst[dstindex] = src[srcindex];
                            }
                        }
                    }
                }
            }
        }
    } // not fast

    if (summerFunc != nullptr && kernelsum != nullptr) {
        summerFunc(kernelsum, (int8_t*)dst, blockNum * hU, blockL, UNIT, SRC_UNIT);
    }
}

void ConvInt8TiledExecutor::packWeightAndQuantInfo(int8_t* dstbuffer, const int8_t* weight, const int8_t* quantInfo,
                                                   int32_t* info, int infoBytes) {
    int blockNum = info[0];
    int ocDiv = info[1];
    int blockL = info[2];
    int UNIT = info[3];
    int SRC_UNIT = info[4];
    auto ocUp4 = info[5];
    auto src0 = weight;                              // int8 weight: [oc/hp, blocknum, ic/lp*(kx*ky)/blocknum, hp, lp]
    auto src1 = quantInfo;                           // dequant scale: [blocknum, ocUp4]
    auto src2 = src1 + infoBytes * ocUp4 * blockNum; // dequant bias: [blocknum, ocUp4]
    int stride0 = info[0] * info[2] * info[3] * info[4];
    int stride1 = info[2] * info[3] * info[4];

    // dst: [oc/hp, blocknum, packedUnit]
    // packedUnit: [ic/lp*(kx*ky)/blocknum, hp, lp] + [hp] + [hp]

    for (int hU = 0; hU < ocDiv; ++hU) {
        auto huPtr = dstbuffer + hU * blockNum * (stride1 + 2 * UNIT * infoBytes);
        int scaleCount = ALIMIN(ocUp4 - hU * UNIT, UNIT);
        for (int bl = 0; bl < blockNum; ++bl) {
            auto blockPtr = huPtr + bl * (stride1 + 2 * UNIT * infoBytes);
            memcpy(blockPtr, src0 + bl * stride1 + hU * stride0, stride1);
            if (scaleCount < UNIT) {
                ::memset(blockPtr + stride1, 0, 2 * UNIT * infoBytes);
            }
            memcpy(blockPtr + stride1, src1 + (bl * ocUp4 + hU * UNIT) * infoBytes, scaleCount * infoBytes);
            memcpy(blockPtr + stride1 + UNIT * infoBytes, src2 + (bl * ocUp4 + hU * UNIT) * infoBytes,
                   scaleCount * infoBytes);
        }
    }
}

static void _computeReorderQuantInfo(float* weightKernelSum, int32_t* paramsKernelSum, bool blockQuantInput,
                                     int weightBits, bool asyQuantWeight, float* quanInfoPtr, int outputCount,
                                     int kernelCount, int pack, AutoStorage<int8_t>& reorderedQuantInfo,
                                     float* ikernelSum, int HP, bool realInt4OrInt8) {
    // Only used for dynamic quant:
    // copy gemm bias
    // copy/compute real dequant bias/scale
    // dequant weight kernel sum
    int ocUp4 = ROUND_UP(outputCount, pack);
    int ocUpHp = ROUND_UP(outputCount, HP);

    int blockNum = paramsKernelSum[0];
    int kernelSumSize = paramsKernelSum[1];
    int scaleSize = blockNum * ocUp4; // pack size.
    int blockSize = kernelCount / blockNum;
    int originOffset = 0;
    if (weightBits == 4) {
        originOffset = -8;
    } else if (weightBits == 3) {
        // w3 kernels produce unsigned 0..7; -4 re-centers to signed [-4, 3].
        originOffset = -4;
    } else if (weightBits == 2) {
        originOffset = -2;
    }
    // Save weight quant alpha and zero: wf=alpha*wi+zero
    auto alphaPtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + scaleSize * sizeof(float));
    if (outputCount % pack != 0) {
        ::memset(alphaPtr, 0, scaleSize * sizeof(float));
        ::memset(biasPtr, 0, scaleSize * sizeof(float));
    }
    ::memset(weightKernelSum, 0, kernelSumSize * sizeof(float));

    int ocDiv4 = UP_DIV(outputCount, pack);
    // resource->mWeightKernelSum: [hU,blocknum,hP]
    if (asyQuantWeight) {
        for (int i = 0; i < outputCount; ++i) {
            float accum = 0.f;
            auto ocOutside = i / HP;
            auto ocInside = i % HP;
            for (int j = 0; j < blockNum; ++j) {
                int index = i * blockNum + j;
                int srcSumIndex = ocOutside * blockNum * HP + j * HP + ocInside; // ikernelsum: [hU,blocknum,hP]
                alphaPtr[j * ocUp4 + i] = quanInfoPtr[2 * index + 1];
                biasPtr[j * ocUp4 + i] = quanInfoPtr[2 * index] + (float)originOffset * quanInfoPtr[2 * index + 1];
                if (realInt4OrInt8) {
                    accum +=
                        (ikernelSum[srcSumIndex] * quanInfoPtr[2 * index + 1] + blockSize * biasPtr[j * ocUp4 + i]);
                } else {
                    accum += ((ikernelSum[srcSumIndex] - blockSize * 8) * quanInfoPtr[2 * index + 1] +
                              blockSize * quanInfoPtr[2 * index]);
                }
                if (blockQuantInput) {
                    int dstSumIndex = ocOutside * blockNum * HP + j * HP + ocInside;
                    weightKernelSum[dstSumIndex] = accum;
                    accum = 0;
                }
            }
            if (!blockQuantInput) {
                weightKernelSum[i] = accum;
            }
        }
    } else {
        for (int i = 0; i < outputCount; ++i) {
            float accum = 0.f;
            auto ocOutside = i / HP;
            auto ocInside = i % HP;
            for (int j = 0; j < blockNum; ++j) {
                int index = i * blockNum + j;
                int srcSumIndex = ocOutside * blockNum * HP + j * HP + ocInside; // ikernelsum: [hU,blocknum,hP]
                alphaPtr[j * ocUp4 + i] = quanInfoPtr[index];
                biasPtr[j * ocUp4 + i] = (float)originOffset * quanInfoPtr[index];
                if (realInt4OrInt8) {
                    accum += (ikernelSum[srcSumIndex] * quanInfoPtr[index] + blockSize * biasPtr[j * ocUp4 + i]);
                } else {
                    accum += ((ikernelSum[srcSumIndex] - blockSize * 8) * quanInfoPtr[index]);
                }
                if (blockQuantInput) {
                    int dstSumIndex = ocOutside * blockNum * HP + j * HP + ocInside;
                    weightKernelSum[dstSumIndex] = accum;
                    accum = 0;
                }
            }
            if (!blockQuantInput) {
                weightKernelSum[i] = accum;
            }
        }
    }
}

// fp16-source twin of the asymmetric branch of _computeReorderQuantInfo, used when the packed
// quant info is built straight from alphaHalf. Only weightKernelSum is needed in that mode;
// scale/zero are upconverted on the fly (widening is exact), so the fp32 math is identical.
static void _computeReorderKernelSumFp16(float* weightKernelSum, int32_t* paramsKernelSum, bool blockQuantInput,
                                         int weightBits, const int16_t* quanInfoHalf, int outputCount, int kernelCount,
                                         float* ikernelSum, int HP, bool realInt4OrInt8) {
    int blockNum = paramsKernelSum[0];
    int kernelSumSize = paramsKernelSum[1];
    int blockSize = kernelCount / blockNum;
    int originOffset = 0;
    if (weightBits == 4) {
        originOffset = -8;
    } else if (weightBits == 3) {
        originOffset = -4;
    } else if (weightBits == 2) {
        originOffset = -2;
    }
    ::memset(weightKernelSum, 0, kernelSumSize * sizeof(float));
    auto quanHalf = reinterpret_cast<const half_float::half*>(quanInfoHalf);
    for (int i = 0; i < outputCount; ++i) {
        float accum = 0.f;
        auto ocOutside = i / HP;
        auto ocInside = i % HP;
        for (int j = 0; j < blockNum; ++j) {
            int index = i * blockNum + j;
            int srcSumIndex = ocOutside * blockNum * HP + j * HP + ocInside; // ikernelsum: [hU,blocknum,hP]
            const float minValue = (float)quanHalf[2 * index];
            const float scale = (float)quanHalf[2 * index + 1];
            const float bias = minValue + (float)originOffset * scale;
            if (realInt4OrInt8) {
                accum += (ikernelSum[srcSumIndex] * scale + blockSize * bias);
            } else {
                accum += ((ikernelSum[srcSumIndex] - blockSize * 8) * scale + blockSize * minValue);
            }
            if (blockQuantInput) {
                int dstSumIndex = ocOutside * blockNum * HP + j * HP + ocInside;
                weightKernelSum[dstSumIndex] = accum;
                accum = 0;
            }
        }
        if (!blockQuantInput) {
            weightKernelSum[i] = accum;
        }
    }
}

static inline void calculateSmeNeonWorkDivision(int& ocMain, int& ocBranch, std::vector<int>& divides, int oc,
                                                int threads, int pack, int planeSize, int divisionRatio, int smeCores,
                                                int ocMainThreads) {
    // workload
    // ocMain is the SME/NEON weight-packing boundary chosen at construction time with the pool's
    // full thread count; derive it with ocMainThreads so the boundary stays identical even when the
    // execute-time thread count is capped (computeThreadNumber). Otherwise SME threads would read
    // NEON-packed weights past the boundary.
    auto ocDivPack = UP_DIV(oc, pack);
    auto workUnit = UP_DIV(ocDivPack, divisionRatio * smeCores + 1 * (ocMainThreads - smeCores));
    int calOcMain = ALIMIN(ROUND_UP(workUnit * pack * smeCores * divisionRatio, GEMM_INT8_UNIT_SME2_128), oc);
    if (calOcMain <= ocMain) { // The purpose of this function is to increase the value of ocMain.
        return;
    }
    ocMain = calOcMain;
    ocBranch = oc - ocMain;
    divides.assign(threads + 1, ocDivPack);
    divides[0] = 0;

    // runtime UNIT for different core and different process(prefill or decode)
    auto rtUnit4Sme = planeSize == 1 ? GEMM_INT8_UNIT_SME2_128 : GEMM_INT8_UNIT_SME2;
    // mOcMain
    auto ocPerSmeCore = ALIMIN(UP_DIV(UP_DIV(ROUND_UP(ocMain, pack), rtUnit4Sme), smeCores) * (rtUnit4Sme / pack),
                               UP_DIV(ocMain, pack));
    for (int i = 0; i < smeCores; ++i) {
        divides[i + 1] = ALIMIN(divides[i] + ocPerSmeCore, UP_DIV(ocMain, pack));
    }

    // ocRemain
    if (ocBranch > 0) {
        auto ocPerNeonCore = UP_DIV(UP_DIV(ROUND_UP(ocBranch, pack), GEMM_INT8_UNIT_ARM82), threads - smeCores) *
                             (GEMM_INT8_UNIT_ARM82 / pack);
        for (int i = smeCores + 1; i < threads + 1; ++i) {
            divides[i] = ALIMIN(divides[i - 1] + ocPerNeonCore, ocDivPack);
        }
    }
}

static inline void _getProportions(int totalProp, int& intensiveProp, int& lightProp) {
    // compute the proportions of different kernels
    lightProp = totalProp % 8;
    intensiveProp = totalProp / 8 % 8;
    if (lightProp == 0 && intensiveProp == 0) {
        // pass
        // Don't use mixed kernels
    } else if (lightProp == 0) {
        lightProp = 1;
    } else if (intensiveProp == 0) {
        intensiveProp = 6;
    } else if (lightProp > intensiveProp) {
        lightProp = 1;
    }
}

static inline void _computeDivides4Sme(std::vector<int>& divides, int threads, int smeCoreNums, int size) {
    divides.resize(threads + 1);
    divides[0] = 0;
    auto length = UP_DIV(size, smeCoreNums);
    auto cur = length;
    for (int i = 1; i < smeCoreNums + 1; ++i) {
        divides[i] = cur;
        cur = ALIMIN(cur + length, size);
    }
}

static inline void _updateMixedKernelFlag(bool& mixedKernel, bool& onlineReorderWeightSme, int threads, int eP,
                                          bool isDynamciQuant, bool postiveBothProp) {
    mixedKernel = false;
    if (threads >= 4 && eP == GEMM_INT8_DST_XUNIT_SME2 && isDynamciQuant && postiveBothProp) {
        mixedKernel = true;
        onlineReorderWeightSme = true;
    }
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Op* op,
                                                       std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon,
                                                       bool isDynamicQuant)
    : DenseConvInt8TiledExecutor(backend, op, std::move(quanCommon), isDynamicQuant, nullptr) {}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Op* op,
                                                       std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon,
                                                       bool isDynamicQuant, WeightReorderComplete weightReorderComplete)
    : ConvInt8TiledExecutor(backend, op) {
    // convolution info
    auto convOp = op->main_as_Convolution2D();
    int kernelCount = mCommon->kernelX() * mCommon->kernelY();
    int oc = convOp->common()->outputCount();
    int ic = convOp->common()->inputCount();
    bool asyWeight = quanCommon ? quanCommon->asymmetric : false;

    mOcBranch = 0;
    mOcMain = oc;

    int blockNum = 1;
    int inputBlockNum = 1;
    if (quanCommon) {
        int dequantCnt = quanCommon->alphaSize;
        if (quanCommon->asymmetric) {
            dequantCnt /= 2;
        }
        blockNum = dequantCnt / oc;
    }
    mBlockNum = blockNum;

    // backend info
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend)->functions();
    const int threads = static_cast<CPUBackend*>(backend)->threadNumber();
    const int pack = gcore->pack;

    // runtime hint
    auto option = static_cast<CPUBackend*>(backend)->getRuntime()->hint().dynamicQuantOption;
    auto weightOnlineReorderOption = WEIGHT_ONLINE_REORDER & option;
    auto inputBlockQuantOption = option % WEIGHT_ONLINE_REORDER;
    if (inputBlockQuantOption == 2) {
        inputBlockNum = blockNum;
    }

    _getProportions(static_cast<CPUBackend*>(backend)->getRuntime()->hint().divisionRatio, mRatioPrefill, mRatioDecode);
    mSmeCores = gcore->smeCoreNumber;

    mRelatedFunctions = *(static_cast<CPUBackend*>(backend)->int8GemmFunctions());
    mArm82Functions = gcore->arm82MatmulRelatedFunctions;

    int UNITMain, SRC_UNITMain, DST_XUNITMain;
    int UNITBranch = 0;
    int SRC_UNITBranch = 0, DST_XUNITBranch = 0;
    mRelatedFunctions.MNNGetGemmUnit(&UNITMain, &SRC_UNITMain, &DST_XUNITMain);

    if (mArm82Functions.MNNGetGemmUnit != nullptr) { // exclude cpu does not support arm82
        mArm82Functions.MNNGetGemmUnit(&UNITBranch, &SRC_UNITBranch, &DST_XUNITBranch);
    }

    // prefer to maximum decode performance & the machine supports 'sme2' & the runtime backend is 'sme2' ->
    // mOnlineReorderWeightSme=true
    mOnlineReorderWeightSme = (weightOnlineReorderOption > 0 && DST_XUNITMain == GEMM_INT8_DST_XUNIT_SME2);
    if (isDynamicQuant == false) {
        mOnlineReorderWeightSme = false;
    }
    _updateMixedKernelFlag(mMixedKernel, mOnlineReorderWeightSme, threads, DST_XUNITMain, isDynamicQuant,
                           mRatioDecode && mRatioPrefill);

    if (mMixedKernel) {
        // total work: UP_DIV(oc, pack)
        // (sme's work / neon's work) = divisionRatio
        auto workUnit = UP_DIV(UP_DIV(oc, pack), mRatioDecode * mSmeCores + 1 * (threads - mSmeCores));
        mOcMain = ALIMIN(ROUND_UP(workUnit * pack * mSmeCores * mRatioDecode, GEMM_INT8_UNIT_SME2_128), oc);
        ;
        mOcBranch = oc - mOcMain;
    }
    if (mOnlineReorderWeightSme) {
        UNITMain = GEMM_INT8_UNIT_SME2_128;
    }

    // compute info
    int ocUp4Main = ROUND_UP(mOcMain, pack);
    int ocUpHpMain = ROUND_UP(mOcMain, UNITMain);
    int lUMain = UP_DIV(ic / blockNum, SRC_UNITMain) * kernelCount;
    int scaleSizeMain = ocUp4Main * blockNum;

    int ocUp4Branch = ROUND_UP(mOcBranch, pack);
    int ocUpHpBranch = UNITBranch != 0 ? ROUND_UP(mOcBranch, UNITBranch) : 0;
    int ocDivHpBranch = UNITBranch != 0 ? UP_DIV(mOcBranch, UNITBranch) : 0;
    int lUBranch = UNITBranch != 0 ? UP_DIV(ic / blockNum, SRC_UNITBranch) * kernelCount : 0;
    int scaleSizeBranch = ocUp4Branch * blockNum;

    std::vector<int> shapeMain = {blockNum, UP_DIV(mOcMain, UNITMain), lUMain, UNITMain, SRC_UNITMain};
    std::vector<int> shapeBranch = {blockNum, ocDivHpBranch, lUBranch, UNITBranch, SRC_UNITBranch};
    mResourceInt8.reset(new CPUConvolution::ResourceInt8);
    mResourceInt8->mWeightAsymmetricQuant = asyWeight;
    mResourceInt8->mWeightBits = 8;
    mResourceInt8->mBlockNum = blockNum;
    mResourceInt8->mHp = UNITMain;
    mResourceInt8->mLp = SRC_UNITMain;
    if (DST_XUNITMain == GEMM_INT8_DST_XUNIT_SME2) {
        mResourceInt8->mPackMode = 1;
    }
    if (quanCommon && quanCommon->canUseInt4) {
        shapeMain[4] = SRC_UNITMain / 2;
        shapeBranch[4] = SRC_UNITBranch / 2;
        mResourceInt8->mWeightBits = 4;
        mResourceInt8->mWeightAsymmetricQuant = true; // offset: 8 from uint8_t
    } else if (quanCommon && quanCommon->canUseInt2 &&
               ((gcore->bytes == 2 && gcore->pack == 8 &&
                 mRelatedFunctions.MNNGemmInt8AddBiasScale_w2_Unit_FP16 != nullptr) ||
                mRelatedFunctions.Int8GemmKernel_W2 != nullptr)) {
        // FP16 path uses MNNGemmInt8AddBiasScale_w2_Unit_FP16; FP32 path uses Int8GemmKernel_W2.
        // Backends without either kernel fall through to the int8 (loader-expanded) path.
        shapeMain[4] = SRC_UNITMain / 4;
        shapeBranch[4] = SRC_UNITBranch / 4;
        mResourceInt8->mWeightBits = 2;
        mResourceInt8->mWeightAsymmetricQuant = true; // offset: 2 from uint8_t
    } else if (quanCommon && quanCommon->canUseInt3 &&
               ((gcore->bytes == 2 && gcore->pack == 8 &&
                 mRelatedFunctions.MNNGemmInt8AddBiasScale_w3_Unit_FP16 != nullptr) ||
                mRelatedFunctions.Int8GemmKernel_W3 != nullptr)) {
        // bit-plane (2 + 1) split: per-cell layout is [main | aux] + optional padding.
        //   i8mm (SRC_UNIT=8): 16B main + 8B aux = 24 bytes (shape[4] = 3, exact).
        //   sdot (SRC_UNIT=4):  8B main + 4B aux = 12 bytes; 12/8 isn't integer, so we
        //     ceil-round shape[4] to 2 (= 16-byte cell) and pad the trailing 4 bytes with
        //     zero.  The kernel skips those bytes after the aux load.  Without ceiling,
        //     truncating to 1 undersizes the buffer by 33% and the packer/kernel write
        //     past the end → SIGSEGV in a later malloc/memset.
        shapeMain[4] = (SRC_UNITMain * 3 + 7) / 8;
        shapeBranch[4] = (SRC_UNITBranch * 3 + 7) / 8;
        mResourceInt8->mWeightBits = 3;
        mResourceInt8->mWeightAsymmetricQuant = true; // offset: 4 from uint8_t
    }
    mResourceInt8->mDynamicQuant = isDynamicQuant ? true : false;
    const WeightReorderInfo weightReorderInfo = {static_cast<size_t>(shapeMain[2]),
                                                 static_cast<size_t>(shapeMain[1]),
                                                 static_cast<size_t>(shapeMain[0]),
                                                 shapeMain[3],
                                                 shapeMain[4],
                                                 mOcBranch};

    // Relu/Relu6 post parameters
    auto postPtr = getPostParameters();
    mResourceInt8->mReluThreshold.resize(2);
    mResourceInt8->mReluThreshold[0] = postPtr[2];
    mResourceInt8->mReluThreshold[1] = postPtr[3];
    if (gcore->bytes == 2) {
        gcore->MNNFp32ToLowp(mResourceInt8->mReluThreshold.data(),
                             reinterpret_cast<int16_t*>(mResourceInt8->mReluThreshold.data()), 2);
    }
    const auto& hint = static_cast<CPUBackend*>(backend)->getRuntime()->hint();
    // Rearrangement strips inline weights; incompatible clones need an external source to rebuild.
    const bool canRebuild = _compactCanRebuild(convOp);
    // Only fp16-disk alpha is packed as fp16 metadata (the converter already guaranteed its
    // precision); fp32-disk scale/bias keeps the legacy kernels that read fp32 metadata directly.
    // Requiring that no fp32 view was materialized also proves no fp32 post-treat ran on alpha.
    const bool alphaFp16Direct = quanCommon && quanCommon->alphaIsFp16 && quanCommon->alphaHalf.get() &&
                                 nullptr == quanCommon->alpha.get();
    // Single-thread decode is measurably slower with compact metadata (rebuild cost exceeds
    // the byte savings when one core cannot saturate DRAM); enable for multi-thread only.
    // A persistent mmap weight pool (useCachedMmap>=1) is trusted verbatim by later runs whose
    // weight load is skipped, so they cannot reproduce this layout decision; only pack compact
    // when the pool is absent or auto-removed (useCachedMmap==0), never trusted across runs.
    const bool compactCommon = threads > 1 && isDynamicQuant && quanCommon && quanCommon->asymmetric &&
                               (mResourceInt8->mWeightBits == 4 || mResourceInt8->mWeightBits == 8) &&
                               gcore->bytes == 2 && pack == 8 &&
                               gcore->supportFp16arith && gcore->MNNFp32ToLowp && gcore->MNNLowpToFp32 &&
                               (hint.useCachedMmap == 0 || hint.weightMemoryPath.empty()) && canRebuild &&
                               quanCommon->alphaSize == 2 * oc * blockNum && alphaFp16Direct;
    if (compactCommon) {
        const bool w4 = mResourceInt8->mWeightBits == 4;
        const bool compactSme = mOnlineReorderWeightSme && gcore->supportSME2 && gcore->supportSDot &&
                                UNITMain == GEMM_INT8_UNIT_SME2_128 && SRC_UNITMain == GEMM_INT8_SRC_UNIT_SME2 &&
                                DST_XUNITMain == GEMM_INT8_DST_XUNIT_SME2 && UNITBranch == GEMM_INT8_UNIT_ARM82 &&
                                SRC_UNITBranch == GEMM_INT8_SRC_UNIT_ARM82 &&
                                DST_XUNITBranch == GEMM_INT8_DST_XUNIT_ARM82 &&
                                (w4 ? (mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16_DecodeMax &&
                                       mArm82Functions.MNNGemmInt8AddBiasScale_w4_Unit_FP16)
                                    : (mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP16_DecodeMax &&
                                       mArm82Functions.MNNGemmInt8AddBiasScale_Unit_FP16));
        // i8mm machines without SME2: the ARMV86 kernel expands the fp16 metadata in every
        // tile path, so prefill needs no repack; single kernel, no NEON branch.
        const bool compactArmv86 = !mOnlineReorderWeightSme && !mMixedKernel && gcore->supportI8mm &&
                                   UNITMain == GEMM_INT8_UNIT_ARM86 && SRC_UNITMain == GEMM_INT8_SRC_UNIT_ARM86 &&
                                   DST_XUNITMain == GEMM_INT8_DST_XUNIT_ARM86 &&
                                   (w4 ? mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16
                                       : mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP16);
        // sdot machines without i8mm/SME2: same single-kernel shape — the ARMV82 kernel
        // expands the fp16 metadata in every tile path (TILE_12/8/4/1).
        const bool compactArmv82 = !mOnlineReorderWeightSme && !mMixedKernel && gcore->supportSDot &&
                                   UNITMain == GEMM_INT8_UNIT_ARM82 && SRC_UNITMain == GEMM_INT8_SRC_UNIT_ARM82 &&
                                   DST_XUNITMain == GEMM_INT8_DST_XUNIT_ARM82 &&
                                   (w4 ? mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16
                                       : mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP16);
        if (compactSme || compactArmv86 || compactArmv82) {
            mResourceInt8->mWeightQuantInfoMode = 1;
            mResourceInt8->mCompactThreads = threads;
            mResourceInt8->mCompactSmeCores = mSmeCores;
            mResourceInt8->mCompactOcMain = mOcMain;
            mResourceInt8->mCompactDynamicQuantOption = hint.dynamicQuantOption;
            mResourceInt8->mCompactDivisionRatio = hint.divisionRatio;
            mResourceInt8->mCompactSmeKernel =
                compactSme ? (w4 ? mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16_DecodeMax
                                 : mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP16_DecodeMax)
                           : (w4 ? mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16
                                 : mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP16);
            mResourceInt8->mCompactNeonKernel =
                compactSme ? (w4 ? mArm82Functions.MNNGemmInt8AddBiasScale_w4_Unit_FP16
                                 : mArm82Functions.MNNGemmInt8AddBiasScale_Unit_FP16)
                           : nullptr;
        }
    }
    const int quantInfoBytes = mResourceInt8->mWeightQuantInfoMode ? (int)sizeof(int16_t) : (int)sizeof(float);
    // buffer allocate
    auto quantlenMain = 2 * blockNum * ROUND_UP(mOcMain, UNITMain) * quantInfoBytes;
    auto weightlenMain = shapeMain[0] * shapeMain[1] * shapeMain[2] * shapeMain[3] * shapeMain[4];
    auto quantlenBranch = 2 * blockNum * ocUpHpBranch * quantInfoBytes;
    auto weightlenBranch = shapeBranch[0] * shapeBranch[1] * shapeBranch[2] * shapeBranch[3] * shapeBranch[4];

    mResourceInt8->mWeightInt8.reset(
        Tensor::createDevice<uint8_t>({weightlenMain + quantlenMain + weightlenBranch + quantlenBranch}));
    mResourceInt8->mOriginBias.reset(Tensor::createDevice<int32_t>({ocUp4Main + ocUpHpBranch})); // float
    mResourceInt8->mWeightKernelSum.reset(
        Tensor::createDevice<uint8_t>({inputBlockNum * (int)sizeof(float) * (ocUpHpMain + ocUpHpBranch)}));

    auto res = backend->onAcquireBuffer(mResourceInt8->mOriginBias.get(), Backend::STATIC);
    res &= backend->onAcquireBuffer(mResourceInt8->mWeightKernelSum.get(), Backend::STATIC);
    res &= backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);

    if (!res) {
        MNN_ERROR("weight acquire buffer error\n");
        if (weightReorderComplete) {
            weightReorderComplete(false, mResourceInt8, weightReorderInfo);
        }
        return;
    }
    bool useCachedMmap = backend->getRuntime()->hint().useCachedMmap > 1;
    if (useCachedMmap) {
        if (weightReorderComplete) {
            weightReorderComplete(true, mResourceInt8, weightReorderInfo);
        }
        return;
    }

    // read weight, weight's scale&bias, convolution bias
    ::memset(mResourceInt8->mOriginBias->host<float>(), 0, mResourceInt8->mOriginBias->size());

    // dynamic quant
    bool directReadInt4weight = (kernelCount == 1 && ROUND_UP(mOcMain, UNITMain) == mOcMain &&
                                 ROUND_UP(ic, SRC_UNITMain) == ic); // TODO:fix this
    auto ocMain = mOcMain;
    auto ocBranch = mOcBranch;
    auto target = mResourceInt8;
    auto funcsMain = mRelatedFunctions;
    auto funcsBranch = mArm82Functions;
    auto needToReorderWeightOnline4Sme = mOnlineReorderWeightSme;
    // Save bias
    if (convOp->bias()) {
        ::memcpy(mResourceInt8->mOriginBias->host<float>(), convOp->bias()->data(),
                 convOp->bias()->size() * sizeof(float));
    }
    auto coreFuncs = static_cast<CPUBackend*>(backend)->functions();

    auto reorderFunc = [=](decltype(mRelatedFunctions) funcs, std::vector<int> shape, int UNIT, int SRC_UNIT,
                           int DST_XUNIT, int weightlen, int scaleSize, int oc, int offsetTg, bool fastReadWeight,
                           int8_t** addressPtr, weightSummerFuncion sumFunc) -> int {
        auto sh = shape;
        AutoStorage<int8_t> weightReordered(weightlen);
        AutoStorage<int8_t> reorderedQuantInfo(2 * scaleSize * sizeof(float));
        AutoStorage<int8_t> kernelsum(blockNum * ROUND_UP(oc, UNIT) * sizeof(float));
        if (weightReordered.get() == nullptr || reorderedQuantInfo.get() == nullptr || kernelsum.get() == nullptr) {
            MNN_ERROR("Memory not enough\n");
            return -1;
        }
        memset(kernelsum.get(), 0, blockNum * ROUND_UP(oc, UNIT) * sizeof(float));

        /* 1. reorder weight */
        auto srcPtr = (uint8_t*)addressPtr[0];
        if (target->mWeightBits == 4 && fastReadWeight) {
            auto dstPtr = (uint8_t*)weightReordered.get();
            ::memset(dstPtr, 0, weightlen);
            funcs.MNNReorderWeightInt4(dstPtr, srcPtr, sh.data(), sh.size(), (float*)kernelsum.get());
        } else { // int4 weight but oc/ic not packed
            int blocksize = ic * kernelCount / blockNum;
            int originOffset = 0;
            int32_t info[6] = {blockNum, oc, ic, kernelCount, UNIT, SRC_UNIT};
            if (target->mWeightBits == 4) {
                if (kernelCount == 1 && ROUND_UP(ic, SRC_UNIT) == ic) {
                    int ocUp = ROUND_UP(oc, UNIT);
                    int rowBytes = ic * kernelCount / 2;
                    AutoStorage<uint8_t> paddedWeight(ocUp * rowBytes);
                    if (paddedWeight.get() == nullptr) {
                        MNN_ERROR("Weight reorder memory not enough!\n");
                        return -1;
                    }
                    ::memset(paddedWeight.get(), 0, ocUp * rowBytes);
                    for (int o = 0; o < oc; ++o) {
                        ::memcpy(paddedWeight.get() + o * rowBytes, srcPtr + o * rowBytes, rowBytes);
                    }
                    funcs.MNNReorderWeightInt4(reinterpret_cast<uint8_t*>(weightReordered.get()), paddedWeight.get(),
                                               sh.data(), sh.size(), reinterpret_cast<float*>(kernelsum.get()));
                } else {
                    originOffset = -8;
                    std::vector<uint8_t> tmpWeight(oc * ic * kernelCount);
                    for (int j = 0; j < oc; ++j) {
                        for (int k = 0; k < blockNum; ++k) {
                            for (int i = 0; i < blocksize; ++i) {
                                int index = j * blockNum * blocksize + k * blocksize + i;
                                uint8_t w_ = srcPtr[index / 2];
                                int truew = index % 2 ? (w_ & 0x0f) : (w_ >> 4);
                                tmpWeight[index] = truew;
                            }
                        }
                    }
                    AutoStorage<uint8_t> packedInt8weight(weightlen * 2);
                    if (packedInt8weight.get() == nullptr) {
                        MNN_ERROR("Weight reorder memory not enough!\n");
                        return -1;
                    }

                    reorderWeight(packedInt8weight.get(), (uint8_t*)tmpWeight.data(), info, 0, (float*)kernelsum.get(),
                                  sumFunc);

                    // pack two int4 to int8
                    int leng = weightlen * 2;
                    auto srcint4Ptr = (uint8_t*)packedInt8weight.get();
                    auto dstint4Ptr = (uint8_t*)weightReordered.get();
                    int permuteUnit = UNIT * SRC_UNIT;
                    int halfPermuteStride = static_cast<int32_t>(permuteUnit / 2);
                    for (int i = 0; i < leng / permuteUnit; ++i) {
                        auto src0 = srcint4Ptr + i * permuteUnit;
                        auto dst0 = dstint4Ptr + i * halfPermuteStride;
                        for (int j = 0; j < halfPermuteStride; ++j) {
                            int s0, s1, d;
                            if (DST_XUNIT == GEMM_INT8_DST_XUNIT_SME2) { // SME2
                                s0 = src0[2 * j + 0];
                                s1 = src0[2 * j + 1];
                                d = s0 + (s1) * 16;
                            } else {
                                s0 = src0[j];
                                s1 = src0[j + halfPermuteStride];
                                d = (s0) * 16 + (s1);
                            }
                            dst0[j] = d;
                        }
                    }
                }
            } else if (target->mWeightBits == 2) {
                // Loader gave us signed int8 in [-2, 1]; convert to unsigned [0, 3], reorder via int8
                // path, then pack 4 weights/byte into the kernel-expected layout.
                originOffset = -2;
                std::vector<uint8_t> tmpWeight(oc * ic * kernelCount);
                for (int idx = 0; idx < oc * ic * kernelCount; ++idx) {
                    tmpWeight[idx] = (uint8_t)((int)((int8_t*)srcPtr)[idx] + 2);
                }
                AutoStorage<uint8_t> packedInt8weight(weightlen * 4);
                if (packedInt8weight.get() == nullptr) {
                    MNN_ERROR("Weight reorder memory not enough!\n");
                    return -1;
                }
                reorderWeight(packedInt8weight.get(), (uint8_t*)tmpWeight.data(), info, 0, (float*)kernelsum.get(),
                              sumFunc);

                int permuteUnit = UNIT * SRC_UNIT; // 64 for ARMV86 (UNIT=8, SRC_UNIT=8); 32 for ARMV82 sdot (8x4)
                int quarterPermuteStride = permuteUnit / 4; // 16 bytes for ARMV86, 8 bytes for sdot
                int leng = weightlen * 4;                   // unsigned int8 stage size
                auto srcw2Ptr = (uint8_t*)packedInt8weight.get();
                auto dstw2Ptr = (uint8_t*)weightReordered.get();
                MNN_ASSERT(UNIT == 8 && (SRC_UNIT == 8 || SRC_UNIT == 4));
                if (SRC_UNIT == 8) {
                    // ARMV86 i8mm layout. 16 bytes per cell:
                    //   bytes 0..7  : bits[6:7]=oc0, [4:5]=oc2, [2:3]=oc4, [0:1]=oc6  (at IC = b)
                    //   bytes 8..15 : bits[6:7]=oc1, [4:5]=oc3, [2:3]=oc5, [0:1]=oc7  (at IC = b-8)
                    for (int i = 0; i < leng / permuteUnit; ++i) {
                        auto src0 = srcw2Ptr + i * permuteUnit;
                        auto dst0 = dstw2Ptr + i * quarterPermuteStride;
                        for (int b = 0; b < quarterPermuteStride; ++b) {
                            int ic_in_cell = b % 8;
                            int oc_offset = b / 8;
                            uint8_t out = 0;
                            out |= ((uint8_t)(src0[(0 + oc_offset) * 8 + ic_in_cell] & 0x3) << 6);
                            out |= ((uint8_t)(src0[(2 + oc_offset) * 8 + ic_in_cell] & 0x3) << 4);
                            out |= ((uint8_t)(src0[(4 + oc_offset) * 8 + ic_in_cell] & 0x3) << 2);
                            out |= ((uint8_t)(src0[(6 + oc_offset) * 8 + ic_in_cell] & 0x3) << 0);
                            dst0[b] = out;
                        }
                    }
                } else { // SRC_UNIT == 4 (ARMV82 sdot)
                    // 8 bytes per cell:
                    //   bytes 0..3 : bits[6:7]=oc0, [4:5]=oc1, [2:3]=oc2, [0:1]=oc3  (at IC = b)
                    //   bytes 4..7 : bits[6:7]=oc4, [4:5]=oc5, [2:3]=oc6, [0:1]=oc7  (at IC = b-4)
                    for (int i = 0; i < leng / permuteUnit; ++i) {
                        auto src0 = srcw2Ptr + i * permuteUnit;
                        auto dst0 = dstw2Ptr + i * quarterPermuteStride;
                        for (int b = 0; b < quarterPermuteStride; ++b) {
                            int ic_in_cell = b % 4;
                            int oc_offset = (b / 4) * 4;
                            uint8_t out = 0;
                            out |= ((uint8_t)(src0[(oc_offset + 0) * 4 + ic_in_cell] & 0x3) << 6);
                            out |= ((uint8_t)(src0[(oc_offset + 1) * 4 + ic_in_cell] & 0x3) << 4);
                            out |= ((uint8_t)(src0[(oc_offset + 2) * 4 + ic_in_cell] & 0x3) << 2);
                            out |= ((uint8_t)(src0[(oc_offset + 3) * 4 + ic_in_cell] & 0x3) << 0);
                            dst0[b] = out;
                        }
                    }
                }
            } else if (target->mWeightBits == 3) {
                // Loader gave us signed int8 in [-4, 3]; convert to unsigned [0, 7].
                // Reorder via int8 path, then split into 2-bit main plane (16B) + 1-bit aux plane (8B)
                // per (UNIT * SRC_UNIT) cell, contiguously laid out as [main | aux].
                // The kernel produces unsigned [0, 7]; _computeReorderQuantInfo applies
                // originOffset = -4 in the post-process bias term to re-center to signed [-4, 3].
                originOffset = -4;
                std::vector<uint8_t> tmpWeight(oc * ic * kernelCount);
                for (int idx = 0; idx < oc * ic * kernelCount; ++idx) {
                    tmpWeight[idx] = (uint8_t)((int)((int8_t*)srcPtr)[idx] + 4);
                }
                // Stage int8 buffer holds reorderWeight output (1 byte per weight).
                // Cell stride in dst depends on SRC_UNIT: i8mm packs 24B exactly per cell;
                // sdot's 12B doesn't divide UNIT, so shape[4] is rounded up to 2 -> 16B cell
                // with 4 trailing pad bytes (already zeroed by reorderWeight's memset).
                int permuteUnit = UNIT * SRC_UNIT; // 64 i8mm / 32 sdot
                int cellStride = UNIT * shape[4];  // 24 i8mm / 16 sdot
                int cellCount = weightlen / cellStride;
                int int8WeightLen = cellCount * permuteUnit;
                AutoStorage<uint8_t> packedInt8weight(int8WeightLen);
                if (packedInt8weight.get() == nullptr) {
                    MNN_ERROR("Weight reorder memory not enough!\n");
                    return -1;
                }
                reorderWeight(packedInt8weight.get(), (uint8_t*)tmpWeight.data(), info, 0, (float*)kernelsum.get(),
                              sumFunc);

                // Per cell layout (useful bytes; remainder up to cellStride is zero-pad):
                //   i8mm (8x8 cell, 64 weights): 16B main + 8B aux  = 24 useful, cellStride=24
                //   sdot (8x4 cell, 32 weights):  8B main + 4B aux  = 12 useful, cellStride=16
                int cellMain = (SRC_UNIT == 8) ? 16 : 8;
                int cellAux = (SRC_UNIT == 8) ? 8 : 4;
                auto srcStgPtr = (uint8_t*)packedInt8weight.get();
                auto dstPtr = (uint8_t*)weightReordered.get();
                MNN_ASSERT(UNIT == 8 && (SRC_UNIT == 8 || SRC_UNIT == 4));
                if (SRC_UNIT == 8) {
                    for (int i = 0; i < cellCount; ++i) {
                        auto src0 = srcStgPtr + i * permuteUnit; // 64 unsigned int8 in [0, 7]
                        auto dst0 = dstPtr + i * cellStride;     // main, aux, [pad]
                        // Main plane (2-bit), bytes 0..7 OC even, 8..15 OC odd, IC interleaved
                        for (int b = 0; b < cellMain; ++b) {
                            int ic_in_cell = b % 8;
                            int oc_offset = b / 8;
                            uint8_t out = 0;
                            out |= ((uint8_t)(src0[(0 + oc_offset) * 8 + ic_in_cell] & 0x3) << 6);
                            out |= ((uint8_t)(src0[(2 + oc_offset) * 8 + ic_in_cell] & 0x3) << 4);
                            out |= ((uint8_t)(src0[(4 + oc_offset) * 8 + ic_in_cell] & 0x3) << 2);
                            out |= ((uint8_t)(src0[(6 + oc_offset) * 8 + ic_in_cell] & 0x3) << 0);
                            dst0[b] = out;
                        }
                        // Aux plane (1-bit, IC-major): byte ic, bit (7 - oc) is the high bit.
                        // This lets the i8mm w3 kernel unpack all OC pairs with vector shifts instead of tbl/ext.
                        for (int icIdx = 0; icIdx < 8; ++icIdx) {
                            uint8_t out = 0;
                            for (int oc = 0; oc < 8; ++oc) {
                                uint8_t hi = (uint8_t)((src0[oc * 8 + icIdx] >> 2) & 1);
                                out |= (hi << (7 - oc));
                            }
                            dst0[cellMain + icIdx] = out;
                        }
                    }
                } else { // SRC_UNIT == 4 (ARMV82 sdot)
                    // Main bytes 0..3: bits[6:7]=oc0, [4:5]=oc1, [2:3]=oc2, [0:1]=oc3 at IC=b
                    // Main bytes 4..7: bits[6:7]=oc4, ..., [0:1]=oc7 at IC=b-4
                    // Aux byte 0: bits[3:0]=oc0_aux at IC[0..3], bits[7:4]=oc1_aux
                    // Aux byte 1: bits[3:0]=oc2_aux, bits[7:4]=oc3_aux
                    // Aux byte 2..3: oc4..oc7 (same pattern)
                    // Bytes 12..15: zero pad (cellStride - useful = 4 bytes).
                    for (int i = 0; i < cellCount; ++i) {
                        auto src0 = srcStgPtr + i * permuteUnit; // 32 unsigned int8 in [0, 7]
                        auto dst0 = dstPtr + i * cellStride;     // main, aux, [pad]
                        // Main plane
                        for (int b = 0; b < cellMain; ++b) {
                            int ic_in_cell = b % 4;
                            int oc_offset = (b / 4) * 4;
                            uint8_t out = 0;
                            out |= ((uint8_t)(src0[(oc_offset + 0) * 4 + ic_in_cell] & 0x3) << 6);
                            out |= ((uint8_t)(src0[(oc_offset + 1) * 4 + ic_in_cell] & 0x3) << 4);
                            out |= ((uint8_t)(src0[(oc_offset + 2) * 4 + ic_in_cell] & 0x3) << 2);
                            out |= ((uint8_t)(src0[(oc_offset + 3) * 4 + ic_in_cell] & 0x3) << 0);
                            dst0[b] = out;
                        }
                        // Aux plane: each output byte holds 2 OCs * 4 IC = 8 bits.
                        for (int b = 0; b < cellAux; ++b) {
                            int ocPair = b * 2; // OC pair at this aux byte: ocPair / ocPair+1
                            uint8_t out = 0;
                            for (int icIdx = 0; icIdx < 4; ++icIdx) {
                                uint8_t hi0 = (uint8_t)((src0[(ocPair + 0) * 4 + icIdx] >> 2) & 1);
                                uint8_t hi1 = (uint8_t)((src0[(ocPair + 1) * 4 + icIdx] >> 2) & 1);
                                out |= (hi0 << icIdx);       // bits[3:0] = OC[ocPair]
                                out |= (hi1 << (icIdx + 4)); // bits[7:4] = OC[ocPair+1]
                            }
                            dst0[cellMain + b] = out;
                        }
                    }
                }
            } else { // int8 weight
                reorderWeight((uint8_t*)weightReordered.get(), srcPtr, info, 0, (float*)kernelsum.get(), sumFunc);
            }
        }
        if (convOp->symmetricQuan() && convOp->symmetricQuan()->bias()) {
            // Compability for old model
            ::memcpy(target->mOriginBias->host<float>(), convOp->symmetricQuan()->bias()->data(), oc * sizeof(int32_t));
#ifdef MNN_USE_SSE
            if (target->mUseConvQuan) {
                for (int ks = 0; ks < oc; ++ks) {
                    target->mOriginBias->host<int32_t>()[ks] -= 128 * ((float*)kernelsum.get())[ks];
                }
            }
#endif
        }
        /* 2. compute and order dequant scale&bias */
        bool notConvertInt4ToInt8 = true;
        if (target->mWeightBits == 4 && !fastReadWeight) {
            notConvertInt4ToInt8 = false;
        }
        int32_t paramsKernelSum[2] = {blockNum, inputBlockNum * ROUND_UP(oc, UNIT)};
        float* weightKernelSum = (float*)addressPtr[2];
        /* 3. put weight and quantInfo together */
        int32_t params[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], ROUND_UP(oc, pack)};
        int8_t* weightInt8 = addressPtr[1];

        if (target->mWeightQuantInfoMode && alphaFp16Direct) {
            // fp16-disk source: pack the quant info straight from alphaHalf, no fp32 round trip.
            auto quanHalfPtr = reinterpret_cast<const int16_t*>(addressPtr[3]);
            AutoStorage<int16_t> compactInfo(2 * scaleSize);
            if (!compactInfo.get()) {
                return -1;
            }
            ::memset(compactInfo.get(), 0, 2 * scaleSize * sizeof(int16_t));
            _computeReorderKernelSumFp16(weightKernelSum, paramsKernelSum, (inputBlockQuantOption == 2),
                                         target->mWeightBits, quanHalfPtr, oc, kernelCount * ic,
                                         (float*)kernelsum.get(), UNIT, notConvertInt4ToInt8);
            const int ocUp4 = ROUND_UP(oc, pack);
            for (int b = 0; b < blockNum; ++b) {
                for (int i = 0; i < oc; ++i) {
                    compactInfo.get()[b * ocUp4 + i]             = quanHalfPtr[2 * (i * blockNum + b) + 1];
                    compactInfo.get()[scaleSize + b * ocUp4 + i] = quanHalfPtr[2 * (i * blockNum + b)];
                }
            }
            ConvInt8TiledExecutor::packWeightAndQuantInfo(weightInt8, weightReordered.get(),
                                                          reinterpret_cast<int8_t*>(compactInfo.get()), params,
                                                          (int)sizeof(int16_t));
            return 0;
        }
        float* quanScalePtr = (float*)addressPtr[3];
        _computeReorderQuantInfo(weightKernelSum, paramsKernelSum, (inputBlockQuantOption == 2), target->mWeightBits,
                                 asyWeight, quanScalePtr, oc, kernelCount * ic, pack, reorderedQuantInfo,
                                 (float*)kernelsum.get(), UNIT, notConvertInt4ToInt8);
        if (target->mWeightQuantInfoMode) {
            if (!_compactFpEnvironment()) {
                return -1;
            }
            auto sourceBias = reinterpret_cast<float*>(reorderedQuantInfo.get()) + scaleSize;
            for (int b = 0; b < blockNum; ++b) {
                for (int i = 0; i < oc; ++i) {
                    sourceBias[b * ROUND_UP(oc, pack) + i] = quanScalePtr[2 * (i * blockNum + b)];
                }
            }
            AutoStorage<int16_t> compactInfo(2 * scaleSize);
            if (!compactInfo.get()) {
                return -1;
            }
            coreFuncs->MNNFp32ToLowp(reinterpret_cast<float*>(reorderedQuantInfo.get()), compactInfo.get(),
                                     2 * scaleSize);
            ConvInt8TiledExecutor::packWeightAndQuantInfo(weightInt8, weightReordered.get(),
                                                          reinterpret_cast<int8_t*>(compactInfo.get()), params,
                                                          (int)sizeof(int16_t));
        } else {
            ConvInt8TiledExecutor::packWeightAndQuantInfo(weightInt8, weightReordered.get(),
                                                          reorderedQuantInfo.get(), params, (int)sizeof(float));
        }

        return 0;
    };

    auto function = [=]() -> int {
        bool fastReadWeight =
            (kernelCount == 1 && ROUND_UP(ocMain, UNITMain) == ocMain && ROUND_UP(ic, SRC_UNITMain) == ic);
        weightSummerFuncion sumFunc = funcsMain.MNNSumWeightInt8;
        if (needToReorderWeightOnline4Sme) {
            sumFunc = funcsMain.MNNSumWeightInt8SmeHp128;
        }

        // fp16-disk alpha in compact mode packs straight from alphaHalf; every other path
        // consumes the fp32 view, so force the lazy fill here.
        const bool quanInfoFp16 = alphaFp16Direct && target->mWeightQuantInfoMode;
        if (quanCommon && !quanInfoFp16) {
            quanCommon->getAlphaFloat();
        }
        const int quanInfoElemBytes = quanInfoFp16 ? (int)sizeof(int16_t) : (int)sizeof(float);
        int8_t* addressPtr[4];
        addressPtr[0] = quanCommon ? quanCommon->weight.get() : (int8_t*)convOp->symmetricQuan()->weight()->data();
        addressPtr[1] = target->mWeightInt8->host<int8_t>();
        addressPtr[2] = target->mWeightKernelSum->host<int8_t>();
        addressPtr[3] = quanCommon ? (quanInfoFp16 ? (int8_t*)quanCommon->alphaHalf.get() : (int8_t*)quanCommon->alpha.get())
                                   : (int8_t*)convOp->symmetricQuan()->scale()->data();

        auto reorderResult = reorderFunc(funcsMain, shapeMain, UNITMain, SRC_UNITMain, DST_XUNITMain, weightlenMain,
                                         scaleSizeMain, ocMain, 0, fastReadWeight, addressPtr, sumFunc);
        if (reorderResult != 0) {
            if (weightReorderComplete) {
                weightReorderComplete(false, target, weightReorderInfo);
            }
            return reorderResult;
        }

        if (ocBranch > 0) {
            // update the address of weight source, weight destination, weight kernel sum and weight scale
            addressPtr[0] +=
                (target->mWeightBits == 4 ? ocMain * ic * kernelCount / 2
                                          : ocMain * ic * kernelCount); // ocMain%2==0, so divides 2 directly
            addressPtr[1] += (weightlenMain + quantlenMain);
            addressPtr[2] += ROUND_UP(ocMain, UNITMain) * inputBlockNum * sizeof(float);
            addressPtr[3] += (quanCommon->asymmetric ? 2 * ocMain * blockNum * quanInfoElemBytes
                                                     : ocMain * blockNum * quanInfoElemBytes);
            sumFunc = funcsBranch.MNNSumWeightInt8;

            fastReadWeight =
                (kernelCount == 1 && ROUND_UP(ocBranch, UNITMain) == ocBranch && ROUND_UP(ic, SRC_UNITMain) == ic);
            reorderResult =
                reorderFunc(funcsBranch, shapeBranch, UNITBranch, SRC_UNITBranch, DST_XUNITBranch, weightlenBranch,
                            scaleSizeBranch, ocBranch, 1, fastReadWeight, addressPtr, sumFunc);
            if (reorderResult != 0) {
                if (weightReorderComplete) {
                    weightReorderComplete(false, target, weightReorderInfo);
                }
                return reorderResult;
            }
        }
        if (target->mWeightQuantInfoMode) {
            target->mCompactReady.store(true, std::memory_order_release);
        }
        if (weightReorderComplete) {
            weightReorderComplete(true, target, weightReorderInfo);
        }
        return 0;
    };

    static_cast<CPUBackend*>(backend)->enqueueTask(std::move(function));

    if (!isDynamicQuant) {
        mResourceInt8->mDynamicQuant = false;

        std::shared_ptr<float> scaleAndBias(new float[ocUpHpMain * 2 * mBlockNum],
                                            [](void* ptr) { delete[] (float*)ptr; });
        memset(scaleAndBias.get(), 0, ocUpHpMain * 2 * mBlockNum * sizeof(float));
        int weightSize;

        bool weightAsy = false;
        if (quanCommon && quanCommon->asymmetric) {
            weightAsy = true;
        }

        if (convOp->symmetricQuan() && convOp->symmetricQuan()->bias() && convOp->symmetricQuan()->scale()) {
            // Compability for old model
            MNN_ASSERT(convOp->symmetricQuan()->bias()->size() == oc && convOp->symmetricQuan()->scale()->size() == oc);
            ::memcpy(scaleAndBias.get(), convOp->symmetricQuan()->scale()->data(), oc * sizeof(float));
        }
        if ((convOp->quanParameter() && convOp->quanParameter()->alpha()) ||
            (quanCommon && (quanCommon->alpha.get() || quanCommon->alphaHalf.get()))) {
            int quantCount;
            if (convOp->quanParameter() && convOp->quanParameter()->alpha()) {
                quantCount = convOp->quanParameter()->alpha()->size();
            } else {
                // Forces the lazy fp32 fill when the disk form is fp16.
                quanCommon->getAlphaFloat();
                quantCount = quanCommon->alpha.size();
            }

            if (false == weightAsy) { // symmetric quant
                if (convOp->quanParameter() && convOp->quanParameter()->alpha()) {
                    ::memcpy(scaleAndBias.get(), convOp->quanParameter()->alpha()->data(), quantCount * sizeof(float));
                } else {
                    ::memcpy(scaleAndBias.get(), quanCommon->alpha.get(), quanCommon->alpha.size() * sizeof(float));
                }
            } else if (true == weightAsy) { // asymmetric
                int scaleSize = quantCount / 2;
                for (int i = 0; i < scaleSize; ++i) {
                    ((float*)scaleAndBias.get())[i] = quanCommon->alpha.get()[2 * i + 1];
                    ((float*)scaleAndBias.get())[i + ocUpHpMain] = quanCommon->alpha.get()[2 * i];
                }
            }
        }
        initializeConvInt8QuantInfo(mResourceInt8, convOp, quanCommon);
        mMutableResource.reset(new MutableResourceInt8(mResourceInt8, backend, scaleAndBias.get()));

        // gemmInt8 kernel
        mGemmKernel = mRelatedFunctions.Int8GemmKernel;
#ifdef MNN_USE_SSE
        if (convOp->symmetricQuan()) {
            int actBits = convOp->symmetricQuan()->nbits();
            if (actBits <= 7) {
                mGemmKernel = mRelatedFunctions.Int8GemmKernelFast;
            }
        }
#else
        if (convOp->symmetricQuan() && convOp->symmetricQuan()->method() == QuantizeAlgo_OVERFLOW_AWARE) {
            mGemmKernel = mRelatedFunctions.Int8GemmKernelFast;
        }
        if (mResourceInt8->mWeightBits == 4) {
            mGemmKernel = mRelatedFunctions.Int8GemmKernel_W4;
        }
#endif
    }
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Op* op,
                                                       const DenseConvInt8TiledExecutor& exe)
    : ConvInt8TiledExecutor(backend, op, exe.mResourceInt8), mGemmKernel(exe.mGemmKernel), mMixedKernel(exe.mMixedKernel) {}

DenseConvInt8TiledExecutor::~DenseConvInt8TiledExecutor() {
    // Do nothing
}

bool DenseConvInt8TiledExecutor::onSetupLinearFastPath(const std::vector<Tensor*>& inputs,
                                                       const std::vector<Tensor*>& outputs) {
    return false;
}

DenseConvInt8TiledExecutor* DenseConvInt8TiledExecutor::createClone(Backend* bn, const Op* op) const {
    return new DenseConvInt8TiledExecutor(bn, op, *this);
}

bool DenseConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (mResourceInt8->mWeightQuantInfoMode) {
        if (op->type() == OpType_GatherV2 || (dst && !_compactResourceCompatible(bn, *mResourceInt8))) {
            return false;
        }
        // Sharing transfers the compact layout contract; the destination must be rebuildable too.
        if (op->type() == OpType_Convolution && !_compactCanRebuild(op->main_as_Convolution2D())) {
            return false;
        }
    }
    if (nullptr == dst) {
        return true;
    }
    if (op->type() == OpType_GatherV2) {
        *dst = new SharedGather(bn, mResourceInt8);
        return true;
    }
    auto exe = createClone(bn, op);
    if (!exe->valid()) {
        return false;
    }
    *dst = exe;
    return true;
}

ErrorCode DenseConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs,
                                               const std::vector<Tensor*>& outputs) {
    if (mResourceInt8->mWeightQuantInfoMode && !_compactResourceCompatible(backend(), *mResourceInt8)) {
        return NOT_SUPPORT;
    }
    // Initialize.
    mUseBatchQuan = false;
    mIm2ColBasedInt8 = true;
    m4BitPtq = false;
    if (mResourceInt8->mDynamicQuant == false && mResourceInt8->mWeightBits == 4) {
        m4BitPtq = true;
    }

    // backend info
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    int threads = static_cast<CPUBackend*>(backend())->threadNumber();
    mRelatedFunctions = *(static_cast<CPUBackend*>(backend())->int8GemmFunctions());
    mArm82Functions = gcore->arm82MatmulRelatedFunctions;

    // runtime hint
    auto option = static_cast<CPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption;
    mSmeCores = gcore->smeCoreNumber;
    auto inputBlockQuantOption = option % WEIGHT_ONLINE_REORDER;
    auto weightOnlineReorderOption = WEIGHT_ONLINE_REORDER & option;

    _getProportions(static_cast<CPUBackend*>(backend())->getRuntime()->hint().divisionRatio, mRatioPrefill,
                    mRatioDecode);

    // feature map info
    int batch = inputs[0]->batch();
    int inC = inputs[0]->channel();
    auto output = outputs[0];
    int kernelCount = mCommon->kernelY() * mCommon->kernelX();
    int inputPlane = batch * inputs[0]->width() * inputs[0]->height();
    auto planeSize = output->width() * output->height() * output->batch();
    // Prefill-shaped (many-row) convolutions are compute-bound: split them
    // across performance cores only. Decode (planeSize == 1) stays
    // bandwidth-bound and keeps every thread.
    threads = static_cast<CPUBackend*>(backend())->computeThreadNumber(planeSize);
    if (mMixedKernel) {
        // The mixed SME/NEON kernel was selected at construction with threads >= 4; keep the flag
        // consistent by never capping below 4 here.
        threads = ALIMAX(threads, 4);
    }

    int UNIT, SRC_UNIT, DST_XUNIT;
    mRelatedFunctions.MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    mOnlineReorderWeightSme = (weightOnlineReorderOption > 0 && DST_XUNIT == GEMM_INT8_DST_XUNIT_SME2);
    if (mResourceInt8->mDynamicQuant == false) {
        mOnlineReorderWeightSme = false;
    }

    _updateMixedKernelFlag(mMixedKernel, mOnlineReorderWeightSme, threads, DST_XUNIT, mResourceInt8->mDynamicQuant,
                           mRatioDecode && mRatioPrefill);

    if (mOnlineReorderWeightSme && planeSize == 1) { // Decode, set runtime unit
        UNIT = GEMM_INT8_UNIT_SME2_128;
    }

    mGemmUnits[0] = UNIT;
    mGemmUnits[1] = SRC_UNIT;
    mGemmUnits[2] = DST_XUNIT;

    bool fastway = (kernelCount == 1) && (output->width() == inputs[0]->width()) &&
                   (output->height() == inputs[0]->height()) && (mCommon->strideX() * mCommon->strideY()) == 1;
    if (inputPlane > 1) {
        mUseBatchQuan = true;
    }
    if (!fastway) { // general conv
        mIm2ColBasedInt8 = false;
        if (planeSize > 1) {
            mUseBatchQuan = true;
        }
        if (inputBlockQuantOption == 1) { // lowest level.
            mIm2ColBasedInt8 = true;
            mUseBatchQuan = false;
        }
    }

    float weightBytes = 1.0f;
    if (mResourceInt8->mWeightBits == 4) {
        weightBytes = 0.5f;
    } else if (mResourceInt8->mWeightBits == 3) {
        auto packedBytesPerOc = (SRC_UNIT * 3 + 7) / 8;
        weightBytes = static_cast<float>(packedBytesPerOc) / SRC_UNIT;
    } else if (mResourceInt8->mWeightBits == 2) {
        weightBytes = 0.25f;
    }
    mBlockNum = mResourceInt8->mBlockNum;

    CPUConvolution::onResize(inputs, outputs);
    if (mResourceInt8->mDynamicQuant == false) {
        mMutableResource->updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]),
                                                 TensorUtils::getQuantInfo(outputs[0]));
        if (!mMutableResource->mResource->mUseConvQuan) {
            // In some previous quantized models, input's scale already fused with weight's scale and output's scale.
            // So there is no need to read input's scale additionally.
            mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({1, DST_XUNIT * (int)sizeof(float)}));
            auto success = backend()->onAcquireBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
            if (!success) {
                return OUT_OF_MEMORY;
            }
        }
        mIm2ColBasedInt8 = true;
        mUseBatchQuan = false;
    }
    int matmulUnits[3] = {UNIT, SRC_UNIT, DST_XUNIT};
    ConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, gcore,
                                                 core, gcore->pack, matmulUnits);

    // Im2col info
    int im2colBytes = 1;
    const int L2Size = 2048;
    int tileLimitByC = UP_DIV(L2Size, mIm2ColParamter.kernelCountUnit * SRC_UNIT);

    if (mIm2ColBasedInt8 == false) {
        im2colBytes = gcore->bytes;
        tileLimitByC = 1;
    }
    int ic = inputs[0]->channel();
    int tileLimit = 0;
    int outC = output->channel();
    int outC4 = UP_DIV(outC, gcore->pack);
    mOcMain = outC;
    mOcBranch = 0;
    const int pack = gcore->pack;
    auto kernelCountUnit = mIm2ColParamter.kernelCountUnit;
    mSplitByOc = true;
#ifdef MNN_USE_RVV
    auto extension = gcore->extension;
    const bool preferLinearPlaneSplit =
        extension != nullptr && extension->createInt8GemmExecution != nullptr && onSetupLinearFastPath(inputs, outputs);
#else
    const bool preferLinearPlaneSplit = false;
#endif

    // flop and io
    float flop = gcore->bytes * planeSize *
                 (ROUND_UP(output->channel(), gcore->pack) * kernelCountUnit * SRC_UNIT / 1024.0 / 1024.0 / 1024.0);
    float ios = (((CPUBackend*)backend())->getTensorSize(outputs[0], true) +
                 ((CPUBackend*)backend())->getTensorSize(inputs[0], true) +
                 ((CPUBackend*)backend())->getTensorSize(mResourceInt8->mWeightInt8.get()) * weightBytes) /
                (1024.0 * 1024.0 * 1024.0);

    if ((preferLinearPlaneSplit || threads < planeSize || mOnlineReorderWeightSme) &&
        !mMixedKernel) { // Thread split by output nhw.
        tileLimit = ALIMIN(tileLimitByC, UP_DIV(planeSize, threads));
        mIm2ColCount = UP_DIV(tileLimit, DST_XUNIT);
        auto DynamicDestUnit = DST_XUNIT * mIm2ColCount;
        mTileCount = UP_DIV(planeSize, DynamicDestUnit);
        if (preferLinearPlaneSplit || mTileCount > threads || (mOnlineReorderWeightSme && planeSize > 1)) {
            mSplitByOc = false;
        }
    }

    if (mSplitByOc) {
        tileLimit = ALIMIN(tileLimitByC, planeSize);
        mIm2ColCount = UP_DIV(tileLimit, DST_XUNIT);
        auto DynamicDestUnit = DST_XUNIT * mIm2ColCount;
        mTileCount = UP_DIV(planeSize, DynamicDestUnit);
        mDivides.resize(threads + 1);
        mDivides[0] = 0;
        // output channel divided by threads
        if (!mMixedKernel) {
            auto ocPerThread = UP_DIV(outC4, threads);
            auto threadNeed = UP_DIV(outC4, ocPerThread);
            int totalWork = outC4;
            int part = 1;
            if (UNIT > gcore->pack) { // AVX512:UNIT=64,pack=16
                MNN_ASSERT(UNIT % gcore->pack == 0);
                int ocDivUnit = UP_DIV(outC4 * gcore->pack, UNIT);
                ocPerThread = UP_DIV(ocDivUnit, threads);
                threadNeed = UP_DIV(ocDivUnit, ocPerThread);
                totalWork = ocDivUnit;
                part = UNIT / gcore->pack;
            }
            mThreadNums = ALIMIN(threads, threadNeed);

            if (threads >= 4 && DST_XUNIT == GEMM_INT8_DST_XUNIT_SME2 && mResourceInt8->mDynamicQuant) {
                _computeDivides4Sme(mDivides, threads, mSmeCores, totalWork);
            } else {
                mDivides.resize(threads + 1);
                mDivides[0] = 0;
                static_cast<CPUBackend*>(backend())->computeDivideSizes(totalWork, mDivides.data() + 1, flop / ios,
                                                                        threads);
            }
            for (int i = 0; i < mDivides.size(); ++i) {
                mDivides[i] *= part;
            }
        } else {
            // workload
            mOcMain = 0; // initialize for mixed kernel, before calculate
            calculateSmeNeonWorkDivision(mOcMain, mOcBranch, mDivides, outC, threads, pack, planeSize, mRatioDecode,
                                         mSmeCores, static_cast<CPUBackend*>(backend())->threadNumber());
            mThreadNums = threads;
        }
    }

    if (!mSplitByOc) {
        mThreadNums = ALIMIN(threads, mTileCount);
        if (threads >= 4 && DST_XUNIT == GEMM_INT8_DST_XUNIT_SME2 && mResourceInt8->mDynamicQuant && !mMixedKernel) {
            _computeDivides4Sme(mDivides, threads, mSmeCores, mTileCount);
        } else {
            mDivides.resize(threads + 1);
            mDivides[0] = 0;
            static_cast<CPUBackend*>(backend())->computeDivideSizes(mTileCount, mDivides.data() + 1, flop / ios,
                                                                    threads);
        }
    }
    if (mResourceInt8->mWeightQuantInfoMode) {
        // The packed layout is only valid for the configuration it was built for: SME2 online
        // reorder (mCompactNeonKernel set) or the single-kernel i8mm/sdot path (no NEON branch).
        const bool smeCompact = mResourceInt8->mCompactNeonKernel != nullptr;
        if (mOcMain != mResourceInt8->mCompactOcMain || SRC_UNIT != mResourceInt8->mLp ||
            smeCompact != mOnlineReorderWeightSme) {
            return NOT_SUPPORT;
        }
    }
    mDividesTmp.resize(threads + 1);
    if (mMixedKernel) {
        mOriginSmeWork = mDivides[mSmeCores];
    }
    int ocUp4 = ROUND_UP(outC, gcore->pack);
    int k = mThreadNums;
    int workPT = DST_XUNIT * mIm2ColCount;
    if (mSplitByOc) {
        k = 1; // Use one thread to finish im2col.
        workPT = mTileCount * DST_XUNIT * mIm2ColCount;
    }

    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    auto blitInfoSize = ConvolutionTiledExecutor::computeBlitInfoSize(
        workPT, mIm2ColParamter.ow, mIm2ColParamter.kernelX * mIm2ColParamter.kernelY, k);
    mBlitInfoStride = blitInfoSize.second;
    mBlitInfo = bufferAlloc->alloc(blitInfoSize.first);
    const int unitColBufferSize = kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    const int colBufferSize = unitColBufferSize * mIm2ColCount;

    if (!mSplitByOc) {
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({threads, colBufferSize * im2colBytes}));
        mTempSrcSum = bufferAlloc->alloc(threads * mBlockNum * DST_XUNIT * mIm2ColCount * sizeof(float));
    } else {
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mTileCount, colBufferSize * im2colBytes}));
        mTempSrcSum = bufferAlloc->alloc(mTileCount * mBlockNum * DST_XUNIT * mIm2ColCount * sizeof(float));
    }

    mAccumBuffer.reset(Tensor::createDevice<int32_t>({threads, DST_XUNIT * ALIMAX(UNIT, gcore->pack)}));

    auto success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(mAccumBuffer.get(), Backend::DYNAMIC);
    if (!success || mBlitInfo.invalid() || mTempSrcSum.invalid()) {
        return OUT_OF_MEMORY;
    }
    if (false == mResourceInt8->mDynamicQuant && false == m4BitPtq) {
        bufferAlloc->free(mBlitInfo);
        bufferAlloc->free(mTempSrcSum);
        backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        if (mBatchQuantInfo.get()) {
            backend()->onReleaseBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
        }
        backend()->onReleaseBuffer(mAccumBuffer.get(), Backend::DYNAMIC);
        return NO_ERROR;
    }

#ifdef MNN_LOW_MEMORY
    if (!mMixedKernel) { // Dynamic Quant kernels, use single gemm kernel.
        mGemmKernel = mRelatedFunctions.Int8GemmKernel;
        if (mOnlineReorderWeightSme && planeSize == 1) {
            mGemmKernel = mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP32_DecodeMax;
        }
        if (mResourceInt8->mWeightBits == 4) {
            mGemmKernel = mRelatedFunctions.Int8GemmKernel_W4;
            if (mOnlineReorderWeightSme && planeSize == 1) {
                mGemmKernel = mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP32_DecodeMax;
            }
        } else if (mResourceInt8->mWeightBits == 2 && mRelatedFunctions.Int8GemmKernel_W2 != nullptr) {
            mGemmKernel = mRelatedFunctions.Int8GemmKernel_W2;
        } else if (mResourceInt8->mWeightBits == 3 && mRelatedFunctions.Int8GemmKernel_W3 != nullptr) {
            mGemmKernel = mRelatedFunctions.Int8GemmKernel_W3;
        }
        mQuantFunc = core->MNNFloat2Int8;
        if (gcore->bytes == 2 && gcore->pack == 8) {
            mGemmKernel = mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP16;
            if (mOnlineReorderWeightSme && planeSize == 1) {
                mGemmKernel = mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP16_DecodeMax;
            }
            if (mResourceInt8->mWeightBits == 4) {
                mGemmKernel = mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16;
                if (mOnlineReorderWeightSme && planeSize == 1) {
                    mGemmKernel = mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16_DecodeMax;
                }
            } else if (mResourceInt8->mWeightBits == 2 &&
                       mRelatedFunctions.MNNGemmInt8AddBiasScale_w2_Unit_FP16 != nullptr) {
                mGemmKernel = mRelatedFunctions.MNNGemmInt8AddBiasScale_w2_Unit_FP16;
            } else if (mResourceInt8->mWeightBits == 3 &&
                       mRelatedFunctions.MNNGemmInt8AddBiasScale_w3_Unit_FP16 != nullptr) {
                mGemmKernel = mRelatedFunctions.MNNGemmInt8AddBiasScale_w3_Unit_FP16;
            }
            mQuantFunc = core->DynamicQuanInput_ARM82;
            mQuantAndReorderFunc = core->DynamicQuanInputAndReorder_ARM82;
        }
        // A axisSum kernel
    } else { // use sme and neon gemmInt8
        // Fp32
        if (planeSize == 1) { // Decode
            mGemmKernels.push_back(mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP32_DecodeMax);
            mGemmKernels.push_back(mArm82Functions.Int8GemmKernel);
            if (mResourceInt8->mWeightBits == 4) {
                mGemmKernels[0] = mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP32_DecodeMax;
                mGemmKernels[1] = mArm82Functions.Int8GemmKernel_W4;
            } else if (mResourceInt8->mWeightBits == 2 && mRelatedFunctions.Int8GemmKernel_W2 != nullptr) {
                // No SME2 DecodeMax for w2 yet; reuse plain w2 for both branches.
                mGemmKernels[0] = mRelatedFunctions.Int8GemmKernel_W2;
                mGemmKernels[1] = mArm82Functions.Int8GemmKernel_W2 != nullptr ? mArm82Functions.Int8GemmKernel_W2
                                                                               : mRelatedFunctions.Int8GemmKernel_W2;
            } else if (mResourceInt8->mWeightBits == 3 && mRelatedFunctions.Int8GemmKernel_W3 != nullptr) {
                mGemmKernels[0] = mRelatedFunctions.Int8GemmKernel_W3;
                mGemmKernels[1] = mArm82Functions.Int8GemmKernel_W3 != nullptr ? mArm82Functions.Int8GemmKernel_W3
                                                                               : mRelatedFunctions.Int8GemmKernel_W3;
            }
        } else { // Prefill
            mGemmKernels.push_back(mRelatedFunctions.Int8GemmKernel);
            mGemmKernels.push_back(mArm82Functions.Int8GemmKernel);
            if (mResourceInt8->mWeightBits == 4) {
                mGemmKernels[0] = mRelatedFunctions.Int8GemmKernel_W4;
                mGemmKernels[1] = mArm82Functions.Int8GemmKernel_W4;
            } else if (mResourceInt8->mWeightBits == 2 && mRelatedFunctions.Int8GemmKernel_W2 != nullptr) {
                mGemmKernels[0] = mRelatedFunctions.Int8GemmKernel_W2;
                mGemmKernels[1] = mArm82Functions.Int8GemmKernel_W2 != nullptr ? mArm82Functions.Int8GemmKernel_W2
                                                                               : mRelatedFunctions.Int8GemmKernel_W2;
            } else if (mResourceInt8->mWeightBits == 3 && mRelatedFunctions.Int8GemmKernel_W3 != nullptr) {
                mGemmKernels[0] = mRelatedFunctions.Int8GemmKernel_W3;
                mGemmKernels[1] = mArm82Functions.Int8GemmKernel_W3 != nullptr ? mArm82Functions.Int8GemmKernel_W3
                                                                               : mRelatedFunctions.Int8GemmKernel_W3;
            }
        }
        mQuantFunc = core->MNNFloat2Int8;

        // fp16
        if (gcore->bytes == 2 && gcore->pack == 8) {
            if (planeSize == 1) { // Decode
                mGemmKernels[0] = mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP16_DecodeMax;
                mGemmKernels[1] = mArm82Functions.MNNGemmInt8AddBiasScale_Unit_FP16;
                if (mResourceInt8->mWeightBits == 4) {
                    mGemmKernels[0] = mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16_DecodeMax;
                    mGemmKernels[1] = mArm82Functions.MNNGemmInt8AddBiasScale_w4_Unit_FP16;
                } else if (mResourceInt8->mWeightBits == 2 &&
                           mArm82Functions.MNNGemmInt8AddBiasScale_w2_Unit_FP16 != nullptr) {
                    // No SME2 DecodeMax for w2 yet; reuse plain ARMV86 for both branches.
                    mGemmKernels[0] = mArm82Functions.MNNGemmInt8AddBiasScale_w2_Unit_FP16;
                    mGemmKernels[1] = mArm82Functions.MNNGemmInt8AddBiasScale_w2_Unit_FP16;
                } else if (mResourceInt8->mWeightBits == 3 &&
                           mArm82Functions.MNNGemmInt8AddBiasScale_w3_Unit_FP16 != nullptr) {
                    mGemmKernels[0] = mArm82Functions.MNNGemmInt8AddBiasScale_w3_Unit_FP16;
                    mGemmKernels[1] = mArm82Functions.MNNGemmInt8AddBiasScale_w3_Unit_FP16;
                }
            } else { // Prefill
                mGemmKernels[0] = mRelatedFunctions.MNNGemmInt8AddBiasScale_Unit_FP16;
                mGemmKernels[1] = mArm82Functions.MNNGemmInt8AddBiasScale_Unit_FP16;
                if (mResourceInt8->mWeightBits == 4) {
                    mGemmKernels[0] = mRelatedFunctions.MNNGemmInt8AddBiasScale_w4_Unit_FP16;
                    mGemmKernels[1] = mArm82Functions.MNNGemmInt8AddBiasScale_w4_Unit_FP16;
                } else if (mResourceInt8->mWeightBits == 2 &&
                           mArm82Functions.MNNGemmInt8AddBiasScale_w2_Unit_FP16 != nullptr) {
                    mGemmKernels[0] = mRelatedFunctions.MNNGemmInt8AddBiasScale_w2_Unit_FP16;
                    mGemmKernels[1] = mArm82Functions.MNNGemmInt8AddBiasScale_w2_Unit_FP16;
                } else if (mResourceInt8->mWeightBits == 3 &&
                           mArm82Functions.MNNGemmInt8AddBiasScale_w3_Unit_FP16 != nullptr) {
                    mGemmKernels[0] = mRelatedFunctions.MNNGemmInt8AddBiasScale_w3_Unit_FP16;
                    mGemmKernels[1] = mArm82Functions.MNNGemmInt8AddBiasScale_w3_Unit_FP16;
                }
            }
            mQuantFunc = core->DynamicQuanInput_ARM82;
            mQuantAndReorderFunc = core->DynamicQuanInputAndReorder_ARM82;
        }
        // A axisSum kernel
    }

    mInputBlockNum = (inputBlockQuantOption == 2) ? mBlockNum : 1;
    bool symmetricQuant = (inputBlockQuantOption != 2 && mUseBatchQuan) ? true : false;

    int size = 0;
    if (!mUseBatchQuan) { // single quant
        if (mSplitByOc) {
            size = 2 * mInputBlockNum * ALIMIN(DST_XUNIT, planeSize) * sizeof(float);
        } else {
            size = 2 * mInputBlockNum * mIm2ColCount * DST_XUNIT * sizeof(float);
        }
    }
    if (mUseBatchQuan) {
        if (mIm2ColBasedInt8) {
            size = 2 * mInputBlockNum * inputPlane * sizeof(float);
        } else if (!mSplitByOc) { // only threads buffer needed by this case
            size = 2 * mInputBlockNum * mIm2ColCount * DST_XUNIT * sizeof(float);
        } else {
            size = 2 * mInputBlockNum * planeSize * sizeof(float);
        }
    }
    if (symmetricQuant) { // symmetric quant
        size /= 2;
    }

    if (false == m4BitPtq) {
        if (!mIm2ColBasedInt8 && !mSplitByOc) {
            mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({threads, size}));
        } else {
            mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({1, size})); // keep dimensions=2!
        }
        success &= backend()->onAcquireBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
    }

    // Dynamic quant.
    // set im2col tensor info
    if (mIm2ColBasedInt8) {
        mQuantInput.reset((
            Tensor::createDevice<int8_t>({batch, mIm2ColParamter.ih, mIm2ColParamter.iw, ROUND_UP(inC, gcore->pack)})));
    } else if (!mSplitByOc) {
        mQuantInput.reset((Tensor::createDevice<int8_t>({threads, colBufferSize * 1})));
    } else {
        mQuantInput.reset((Tensor::createDevice<int8_t>({mTileCount, colBufferSize * 1})));
    }
    success &= backend()->onAcquireBuffer(mQuantInput.get(), Backend::DYNAMIC);

    // set compute buffer
    int tempSize = threads * 2 * mInputBlockNum * inputPlane;
    if (!mIm2ColBasedInt8) {
        if (!mSplitByOc) {
            tempSize = threads * 2 * mInputBlockNum * DST_XUNIT * mIm2ColCount;
        } else {
            tempSize = threads * 2 * mInputBlockNum * ROUND_UP(planeSize, DST_XUNIT);
        }
    }
    if (symmetricQuant) { // symmetric batch quant.
        tempSize /= 2;
    }
    mSizeInputBlockQuant = tempSize / threads;
    mTempMaxMinValueBuffer = bufferAlloc->alloc(tempSize * gcore->bytes);
    mQScaleZero = bufferAlloc->alloc(tempSize * sizeof(float));

    if (mQScaleZero.invalid()) {
        return OUT_OF_MEMORY;
    }
    if (mOnlineReorderWeightSme && planeSize > 1) { // only prefill need
        int ocProcessedBySme = mOcMain;
        int ocProcessedByNeon = 0;
        if (mMixedKernel && mRatioDecode != mRatioPrefill) {
            auto workUnit = UP_DIV(outC4, mRatioPrefill * mSmeCores + 1 * (threads - mSmeCores));
            ocProcessedBySme =
                ALIMIN(ROUND_UP(workUnit * pack * mSmeCores * mRatioPrefill, GEMM_INT8_UNIT_SME2_128), outC);
            ocProcessedBySme = ALIMAX(ocProcessedBySme, mOcMain);
            ocProcessedByNeon = outC - ocProcessedBySme;
        }
        if (mResourceInt8->mWeightQuantInfoMode) {
            ocProcessedByNeon = outC - ocProcessedBySme;
        }
        int weightlenSme = ROUND_UP(ocProcessedBySme, GEMM_INT8_UNIT_SME2_128) * mBlockNum *
                           ROUND_UP(ic / mBlockNum, SRC_UNIT) * kernelCount;
        int weightlenNeon =
            ROUND_UP(ocProcessedByNeon, 8) * mBlockNum * ROUND_UP(ic / mBlockNum, SRC_UNIT) * kernelCount;
        if (mResourceInt8->mWeightBits == 4) {
            weightlenSme /= 2;
            weightlenNeon /= 2;
        }
        const int prefillQuantInfoBytes =
            mResourceInt8->mWeightQuantInfoMode ? (int)sizeof(int16_t) : (int)sizeof(float);
        int scalebiasLenSme = 2 * mBlockNum * ROUND_UP(ocProcessedBySme, GEMM_INT8_UNIT_SME2_128) * prefillQuantInfoBytes;
        int scalebiasLenNeon = 2 * mBlockNum * ROUND_UP(ocProcessedByNeon, 8) * prefillQuantInfoBytes;

        mWeight4Prefill = bufferAlloc->alloc(weightlenSme + scalebiasLenSme + weightlenNeon + scalebiasLenNeon);
        if (mWeight4Prefill.invalid()) {
            return OUT_OF_MEMORY;
        }
        if (mInputBlockNum > 1) { // only in this case, need to use weight_kernel_sum
            mWeightKernelSum4Prefill =
                bufferAlloc->alloc(ROUND_UP(outC, GEMM_INT8_UNIT_SME2_128) * mBlockNum * sizeof(float));
            if (mWeightKernelSum4Prefill.invalid()) {
                return OUT_OF_MEMORY;
            }
        }
    }
    mToFuseInputbias2Bias = (!mUseBatchQuan && inputBlockQuantOption != 2) ? true : false;
    if (mToFuseInputbias2Bias) { // input data has only one bias&scale
        if (mIm2ColBasedInt8) {
            mBiasBufferFusedInputzero = bufferAlloc->alloc(
                ROUND_UP(outC, UNIT) * sizeof(float)); // should be UP_DIV(oc, UNIT),not UP_DIV(oc, pack)
        } else {
            mBiasBufferFusedInputzero = bufferAlloc->alloc(threads * ROUND_UP(outC, UNIT) * sizeof(float));
        }
        if (mBiasBufferFusedInputzero.invalid()) {
            return OUT_OF_MEMORY;
        }
    }

    if (mBlockNum > 1 && kernelCount > 1) {
        if (mSplitByOc) {
            mReorderBuffer = bufferAlloc->alloc(UP_DIV(planeSize, DST_XUNIT) * unitColBufferSize);
        } else {
            mReorderBuffer = bufferAlloc->alloc(threads * colBufferSize);
        }
        if (mReorderBuffer.invalid()) {
            return OUT_OF_MEMORY;
        }
    }

    if (!success || mTempMaxMinValueBuffer.invalid()) {
        return OUT_OF_MEMORY;
    }
    bufferAlloc->free(mBlitInfo);
    bufferAlloc->free(mTempSrcSum);
    bufferAlloc->free(mTempMaxMinValueBuffer);
    bufferAlloc->free(mQScaleZero);
    if (mOnlineReorderWeightSme && planeSize > 1) {
        bufferAlloc->free(mWeight4Prefill);
        if (mInputBlockNum > 1) {
            bufferAlloc->free(mWeightKernelSum4Prefill);
        }
    }
    if (mBlockNum > 1 && kernelCount > 1) {
        bufferAlloc->free(mReorderBuffer);
    }
    if (mToFuseInputbias2Bias) {
        bufferAlloc->free(mBiasBufferFusedInputzero);
    }

    // Additional Adjustments
    if (m4BitPtq) {
        mTempOutput = bufferAlloc->alloc(ocUp4 * planeSize * gcore->bytes);
        if (mTempOutput.invalid()) {
            return OUT_OF_MEMORY;
        }
        bufferAlloc->free(mTempOutput);
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mQuantInput.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mAccumBuffer.get(), Backend::DYNAMIC);

    return NO_ERROR;
#else
    return NO_ERROR;
#endif
}

static void _copyPrefillQuantInfo(int8_t* dstScale, int8_t* dstOffset, const int8_t* srcScale, const int8_t* srcBias,
                                  int channels, int quantInfoMode) {
    // FP16 metadata passes through unchanged; the GEMM kernels widen (and re-center
    // int4 sourceBias by originOffset) in-register, bit-matching the loader-time fold.
    const int quantInfoBytes = quantInfoMode ? sizeof(int16_t) : sizeof(float);
    memcpy(dstScale, srcScale, channels * quantInfoBytes);
    memcpy(dstOffset, srcBias, channels * quantInfoBytes);
}

static void _onlineReorderWeightPackH128ToH32(int8_t* dst, int8_t* src, int hPSrc, int hPDst, int hU, int blockNum,
                                              int blockLu, int lp, bool int4weight, int quantInfoMode) {
    // hPSrc = 4 * hPDst

    int unitsize_ = hPDst * lp;
    if (int4weight) {
        lp /= 2;
        unitsize_ /= 2;
    }
    int unitsize4 = unitsize_ * 4;

    // Destination metadata keeps the source element size (FP16 stays FP16).
    const int srcQuantInfoBytes = quantInfoMode ? 2 : 4;
    int srcStride1 = blockLu * hPSrc * lp + 2 * hPSrc * srcQuantInfoBytes;
    int srcStride0 = blockNum * srcStride1;
    int dstStride1 = blockLu * hPDst * lp + 2 * hPDst * srcQuantInfoBytes;
    int dstStride0 = blockNum * dstStride1;

    for (int i = 0; i < hU; ++i) {
        for (int k = 0; k < blockNum; ++k) {
            auto weightsrc = (int8_t*)(src + i * srcStride0 + k * srcStride1);
            auto weightdst0 = (int8_t*)(dst + (4 * i) * dstStride0 + k * dstStride1);
            auto weightdst1 = (int8_t*)(dst + (4 * i + 1) * dstStride0 + k * dstStride1);
            auto weightdst2 = (int8_t*)(dst + (4 * i + 2) * dstStride0 + k * dstStride1);
            auto weightdst3 = (int8_t*)(dst + (4 * i + 3) * dstStride0 + k * dstStride1);
            auto lu = blockLu;

            while (lu > 7) {
                for (int j = 0; j < 8; ++j) {
                    memcpy(weightdst0 + j * unitsize_, weightsrc + j * unitsize4 + 0 * unitsize_, unitsize_);
                    memcpy(weightdst1 + j * unitsize_, weightsrc + j * unitsize4 + 1 * unitsize_, unitsize_);
                    memcpy(weightdst2 + j * unitsize_, weightsrc + j * unitsize4 + 2 * unitsize_, unitsize_);
                    memcpy(weightdst3 + j * unitsize_, weightsrc + j * unitsize4 + 3 * unitsize_, unitsize_);
                }
                weightsrc += unitsize4 * 8;
                weightdst0 += unitsize_ * 8;
                weightdst1 += unitsize_ * 8;
                weightdst2 += unitsize_ * 8;
                weightdst3 += unitsize_ * 8;
                lu -= 8;
            }

            if (lu > 3) {
                for (int j = 0; j < 4; ++j) {
                    memcpy(weightdst0 + j * unitsize_, weightsrc + j * unitsize4 + 0 * unitsize_, unitsize_);
                    memcpy(weightdst1 + j * unitsize_, weightsrc + j * unitsize4 + 1 * unitsize_, unitsize_);
                    memcpy(weightdst2 + j * unitsize_, weightsrc + j * unitsize4 + 2 * unitsize_, unitsize_);
                    memcpy(weightdst3 + j * unitsize_, weightsrc + j * unitsize4 + 3 * unitsize_, unitsize_);
                }
                weightsrc += unitsize4 * 4;
                weightdst0 += unitsize_ * 4;
                weightdst1 += unitsize_ * 4;
                weightdst2 += unitsize_ * 4;
                weightdst3 += unitsize_ * 4;
                lu -= 4;
            }

            if (lu > 1) {
                memcpy(weightdst0, weightsrc, unitsize_);
                memcpy(weightdst0 + unitsize_, weightsrc + unitsize4, unitsize_);
                memcpy(weightdst1, weightsrc + unitsize_, unitsize_);
                memcpy(weightdst1 + unitsize_, weightsrc + unitsize4 + unitsize_, unitsize_);
                memcpy(weightdst2, weightsrc + unitsize_ * 2, unitsize_);
                memcpy(weightdst2 + unitsize_, weightsrc + unitsize4 + unitsize_ * 2, unitsize_);
                memcpy(weightdst3, weightsrc + unitsize_ * 3, unitsize_);
                memcpy(weightdst3 + unitsize_, weightsrc + unitsize4 + unitsize_ * 3, unitsize_);

                weightsrc += unitsize4 * 2;
                weightdst0 += unitsize_ * 2;
                weightdst1 += unitsize_ * 2;
                weightdst2 += unitsize_ * 2;
                weightdst3 += unitsize_ * 2;
                lu -= 2;
            }

            if (lu > 0) {
                memcpy(weightdst0, weightsrc, unitsize_);
                memcpy(weightdst1, weightsrc + unitsize_, unitsize_);
                memcpy(weightdst2, weightsrc + unitsize_ * 2, unitsize_);
                memcpy(weightdst3, weightsrc + unitsize_ * 3, unitsize_);
            }

            // Reorder scale and bias
            auto scaleSrc = src + i * srcStride0 + k * srcStride1 + blockLu * hPSrc * lp;
            auto scaleDst0 = dst + (4 * i) * dstStride0 + k * dstStride1 + blockLu * hPDst * lp;
            auto scaleDst1 = dst + (4 * i + 1) * dstStride0 + k * dstStride1 + blockLu * hPDst * lp;
            auto scaleDst2 = dst + (4 * i + 2) * dstStride0 + k * dstStride1 + blockLu * hPDst * lp;
            auto scaleDst3 = dst + (4 * i + 3) * dstStride0 + k * dstStride1 + blockLu * hPDst * lp;

            const int scaleSrcSize = hPDst * srcQuantInfoBytes;
            const int scaleDstSize = hPDst * srcQuantInfoBytes;
            const auto biasSrc = scaleSrc + hPSrc * srcQuantInfoBytes;
            _copyPrefillQuantInfo(scaleDst0, scaleDst0 + scaleDstSize, scaleSrc, biasSrc, hPDst, quantInfoMode);
            _copyPrefillQuantInfo(scaleDst1, scaleDst1 + scaleDstSize, scaleSrc + scaleSrcSize,
                                  biasSrc + scaleSrcSize, hPDst, quantInfoMode);
            _copyPrefillQuantInfo(scaleDst2, scaleDst2 + scaleDstSize, scaleSrc + 2 * scaleSrcSize,
                                  biasSrc + 2 * scaleSrcSize, hPDst, quantInfoMode);
            _copyPrefillQuantInfo(scaleDst3, scaleDst3 + scaleDstSize, scaleSrc + 3 * scaleSrcSize,
                                  biasSrc + 3 * scaleSrcSize, hPDst, quantInfoMode);
        }
    }
}

static void _onlineReorderWeightPackH8ToH32(int8_t* dst, const int8_t* src, int blockLu, int lp, bool isInt4Weight,
                                            int srcH, int blockNum, int resOcBranch, int quantInfoMode) {
    constexpr int hPSrc = 8;
    constexpr int hPDst = 32;

    int srcUnitLp = isInt4Weight ? lp / 2 : lp;

    const size_t srcUnitSize = (size_t)hPSrc * srcUnitLp;
    const size_t dstUnitSize = (size_t)hPDst * srcUnitLp;

    const int srcQuantInfoBytes = quantInfoMode ? 2 : 4;
    const size_t srcStride1 = (size_t)blockLu * srcUnitSize + 2 * hPSrc * srcQuantInfoBytes;
    const size_t srcStride0 = (size_t)blockNum * srcStride1;
    const size_t dstStride1 = (size_t)blockLu * dstUnitSize + 2 * hPDst * srcQuantInfoBytes;
    const size_t dstStride0 = (size_t)blockNum * dstStride1;

    const int hUDst = srcH / 4;
    const int hTail = srcH % 4;

    for (int i = 0; i < hUDst; ++i) {
        for (int k = 0; k < blockNum; ++k) {
            auto weightSrcBase0 = src + (4 * i + 0) * srcStride0 + k * srcStride1;
            auto weightSrcBase1 = src + (4 * i + 1) * srcStride0 + k * srcStride1;
            auto weightSrcBase2 = src + (4 * i + 2) * srcStride0 + k * srcStride1;
            auto weightSrcBase3 = src + (4 * i + 3) * srcStride0 + k * srcStride1;
            auto weightDstBase = dst + i * dstStride0 + k * dstStride1;

            int lu = blockLu;

            // --- Reorder Weights ---
            if (isInt4Weight) {
                auto process_int4_block = [](uint8_t* dst_b, const uint8_t* src_b, size_t size) {
                    auto half_size = size / 2;
                    for (int s = 0; s < half_size; ++s) {
                        uint8_t p0 = src_b[2 * s];
                        uint8_t p1 = src_b[2 * s + 1];
                        dst_b[s] = (p1 & 0xF0) | (p0 >> 4);
                        dst_b[s + half_size] = (p1 << 4) | (p0 & 0x0F);
                    }
                };
                while (lu >= 4) {
                    for (int j = 0; j < 4; ++j) {
                        const auto* srcPtr0 = (const uint8_t*)(weightSrcBase0 + j * srcUnitSize);
                        const auto* srcPtr1 = (const uint8_t*)(weightSrcBase1 + j * srcUnitSize);
                        const auto* srcPtr2 = (const uint8_t*)(weightSrcBase2 + j * srcUnitSize);
                        const auto* srcPtr3 = (const uint8_t*)(weightSrcBase3 + j * srcUnitSize);
                        auto* dstPtr = (uint8_t*)(weightDstBase + j * dstUnitSize);

                        process_int4_block(dstPtr + 0 * srcUnitSize, srcPtr0, srcUnitSize);
                        process_int4_block(dstPtr + 1 * srcUnitSize, srcPtr1, srcUnitSize);
                        process_int4_block(dstPtr + 2 * srcUnitSize, srcPtr2, srcUnitSize);
                        process_int4_block(dstPtr + 3 * srcUnitSize, srcPtr3, srcUnitSize);
                    }

                    weightSrcBase0 += 4 * srcUnitSize;
                    weightSrcBase1 += 4 * srcUnitSize;
                    weightSrcBase2 += 4 * srcUnitSize;
                    weightSrcBase3 += 4 * srcUnitSize;
                    weightDstBase += 4 * dstUnitSize;
                    lu -= 4;
                }

                for (int j = 0; j < lu; ++j) {
                    const auto* srcPtr0 = (const uint8_t*)(weightSrcBase0);
                    const auto* srcPtr1 = (const uint8_t*)(weightSrcBase1);
                    const auto* srcPtr2 = (const uint8_t*)(weightSrcBase2);
                    const auto* srcPtr3 = (const uint8_t*)(weightSrcBase3);
                    auto* dstPtr = (uint8_t*)(weightDstBase);

                    process_int4_block(dstPtr + 0 * srcUnitSize, srcPtr0, srcUnitSize);
                    process_int4_block(dstPtr + 1 * srcUnitSize, srcPtr1, srcUnitSize);
                    process_int4_block(dstPtr + 2 * srcUnitSize, srcPtr2, srcUnitSize);
                    process_int4_block(dstPtr + 3 * srcUnitSize, srcPtr3, srcUnitSize);

                    weightSrcBase0 += srcUnitSize;
                    weightSrcBase1 += srcUnitSize;
                    weightSrcBase2 += srcUnitSize;
                    weightSrcBase3 += srcUnitSize;
                    weightDstBase += dstUnitSize;
                }
            } else {
                while (lu >= 4) {
                    // j = 0
                    memcpy(weightDstBase + 0 * dstUnitSize + 0 * srcUnitSize, weightSrcBase0 + 0 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 0 * dstUnitSize + 1 * srcUnitSize, weightSrcBase1 + 0 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 0 * dstUnitSize + 2 * srcUnitSize, weightSrcBase2 + 0 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 0 * dstUnitSize + 3 * srcUnitSize, weightSrcBase3 + 0 * srcUnitSize,
                           srcUnitSize);
                    // j = 1
                    memcpy(weightDstBase + 1 * dstUnitSize + 0 * srcUnitSize, weightSrcBase0 + 1 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 1 * dstUnitSize + 1 * srcUnitSize, weightSrcBase1 + 1 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 1 * dstUnitSize + 2 * srcUnitSize, weightSrcBase2 + 1 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 1 * dstUnitSize + 3 * srcUnitSize, weightSrcBase3 + 1 * srcUnitSize,
                           srcUnitSize);
                    // j = 2
                    memcpy(weightDstBase + 2 * dstUnitSize + 0 * srcUnitSize, weightSrcBase0 + 2 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 2 * dstUnitSize + 1 * srcUnitSize, weightSrcBase1 + 2 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 2 * dstUnitSize + 2 * srcUnitSize, weightSrcBase2 + 2 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 2 * dstUnitSize + 3 * srcUnitSize, weightSrcBase3 + 2 * srcUnitSize,
                           srcUnitSize);
                    // j = 3
                    memcpy(weightDstBase + 3 * dstUnitSize + 0 * srcUnitSize, weightSrcBase0 + 3 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 3 * dstUnitSize + 1 * srcUnitSize, weightSrcBase1 + 3 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 3 * dstUnitSize + 2 * srcUnitSize, weightSrcBase2 + 3 * srcUnitSize,
                           srcUnitSize);
                    memcpy(weightDstBase + 3 * dstUnitSize + 3 * srcUnitSize, weightSrcBase3 + 3 * srcUnitSize,
                           srcUnitSize);

                    weightSrcBase0 += 4 * srcUnitSize;
                    weightSrcBase1 += 4 * srcUnitSize;
                    weightSrcBase2 += 4 * srcUnitSize;
                    weightSrcBase3 += 4 * srcUnitSize;
                    weightDstBase += 4 * dstUnitSize;
                    lu -= 4;
                }

                for (int j = 0; j < lu; ++j) {
                    memcpy(weightDstBase + 0 * srcUnitSize, weightSrcBase0, srcUnitSize);
                    memcpy(weightDstBase + 1 * srcUnitSize, weightSrcBase1, srcUnitSize);
                    memcpy(weightDstBase + 2 * srcUnitSize, weightSrcBase2, srcUnitSize);
                    memcpy(weightDstBase + 3 * srcUnitSize, weightSrcBase3, srcUnitSize);

                    weightSrcBase0 += srcUnitSize;
                    weightSrcBase1 += srcUnitSize;
                    weightSrcBase2 += srcUnitSize;
                    weightSrcBase3 += srcUnitSize;
                    weightDstBase += dstUnitSize;
                }
            }

            // --- Reorder scale and bias ---
            const int scaleSrcSize = hPSrc * srcQuantInfoBytes;
            const int scaleDstSize = hPSrc * srcQuantInfoBytes;
            const int8_t* scaleSrcBase = src + (4 * i) * srcStride0 + k * srcStride1 + (size_t)blockLu * srcUnitSize;
            int8_t* scaleDstBase = dst + i * dstStride0 + k * dstStride1 + (size_t)blockLu * dstUnitSize;
            const int8_t* biasSrcBase = scaleSrcBase + scaleSrcSize;
            int8_t* biasDstBase = scaleDstBase + hPDst * srcQuantInfoBytes;

            for (int j = 0; j < 4; ++j) {
                _copyPrefillQuantInfo(scaleDstBase + j * scaleDstSize, biasDstBase + j * scaleDstSize,
                                      scaleSrcBase + j * srcStride0, biasSrcBase + j * srcStride0, hPSrc,
                                      quantInfoMode);
            }
        }
    }

    // --- 2. Process the tail ---
    if (hTail > 0) {
        // The last block starts at index hUDst.
        const int i = hUDst;
        for (int k = 0; k < blockNum; ++k) {
            const int8_t* srcBases[4] = {nullptr, nullptr, nullptr, nullptr};
            for (int j = 0; j < hTail; ++j) {
                srcBases[j] = src + (4 * i + j) * srcStride0 + k * srcStride1;
            }

            auto weightDstBase = dst + i * dstStride0 + k * dstStride1;

            int lu = blockLu;

            if (isInt4Weight) {
                auto process_int4_block = [](uint8_t* dst_b, const uint8_t* src_b, size_t size) {
                    auto half_size = size / 2;
                    for (int s = 0; s < half_size; ++s) {
                        uint8_t p0 = src_b[2 * s];
                        uint8_t p1 = src_b[2 * s + 1];
                        dst_b[s] = (p1 & 0xF0) | (p0 >> 4);
                        dst_b[s + half_size] = (p1 << 4) | (p0 & 0x0F);
                    }
                };
                while (lu-- > 0) {
                    for (int j = 0; j < hTail; ++j) {
                        process_int4_block((uint8_t*)(weightDstBase + j * srcUnitSize), (const uint8_t*)(srcBases[j]),
                                           srcUnitSize);
                    }
                    // For the remaining part of the destination block, set 0

                    if (hTail < 4) {
                        memset(weightDstBase + hTail * srcUnitSize, 0, (4 - hTail) * srcUnitSize);
                    }

                    for (int j = 0; j < hTail; ++j) {
                        srcBases[j] += srcUnitSize;
                    }
                    weightDstBase += dstUnitSize;
                }
            } else { // int8 weight
                while (lu-- > 0) {
                    for (int j = 0; j < hTail; ++j) {
                        memcpy(weightDstBase + j * srcUnitSize, srcBases[j], srcUnitSize);
                    }
                    // Zero out the rest of the destination block
                    if (hTail < 4) {
                        memset(weightDstBase + hTail * srcUnitSize, 0, (4 - hTail) * srcUnitSize);
                    }

                    for (int j = 0; j < hTail; ++j) {
                        srcBases[j] += srcUnitSize;
                    }
                    weightDstBase += dstUnitSize;
                }
            }

            // --- Reorder scale and bias for tail ---
            const int scaleSrcSize = hPSrc * srcQuantInfoBytes;
            const int scaleDstSize = hPSrc * srcQuantInfoBytes;
            const int8_t* scaleSrcBase = src + (4 * i) * srcStride0 + k * srcStride1 + (size_t)blockLu * srcUnitSize;
            int8_t* scaleDstBase = dst + i * dstStride0 + k * dstStride1 + (size_t)blockLu * dstUnitSize;
            const int8_t* biasSrcBase = scaleSrcBase + scaleSrcSize;
            int8_t* biasDstBase = scaleDstBase + hPDst * srcQuantInfoBytes;

            for (int j = 0; j < hTail; ++j) {
                _copyPrefillQuantInfo(scaleDstBase + j * scaleDstSize, biasDstBase + j * scaleDstSize,
                                      scaleSrcBase + j * srcStride0, biasSrcBase + j * srcStride0, hPSrc,
                                      quantInfoMode);
            }
            if (hTail < 4) {
                memset(scaleDstBase + hTail * scaleDstSize, 0, (4 - hTail) * scaleDstSize);
                memset(biasDstBase + hTail * scaleDstSize, 0, (4 - hTail) * scaleDstSize);
            }
        }
    }

    // --- 3. Copy the residual part ---
    if (resOcBranch > 0) {
        size_t resLp = isInt4Weight ? lp / 2 : lp;
        size_t resChannels = ROUND_UP(resOcBranch, hPSrc);
        // Residual tiles stay H8; metadata keeps the source element size, so a plain copy suffices.
        size_t resDataLen =
            (size_t)blockNum * ((size_t)blockLu * resChannels * resLp + 2 * resChannels * srcQuantInfoBytes);

        // The source for residual data starts after ALL processed srcH blocks.
        memcpy(dst + (size_t)hUDst * dstStride0 + (hTail > 0 ? dstStride0 : 0), src + (size_t)srcH * srcStride0,
               resDataLen);
    }
}

static void _onlineReorderWeightKernelSumH128ToH32(float* dst, float* src, int blockNum, int hpSrc, int hpDst, int oc) {
    // hpSrc = 4 * hpDst
    // src shape: [huSrc, blockNum, hpSrc]
    // dst shape: [huDst, blockNum, hpDst], where huDst = huSrc * 4

    auto huSrc = UP_DIV(oc, hpSrc);
    auto strideSrc = blockNum * hpSrc;
    auto strideDst = blockNum * hpDst;

    for (int i = 0; i < huSrc; ++i) {
        for (int k = 0; k < blockNum; ++k) {
            auto srcBase = src + i * strideSrc + k * hpSrc;

            auto dst0 = dst + (4 * i + 0) * strideDst + k * hpDst;
            auto dst1 = dst + (4 * i + 1) * strideDst + k * hpDst;
            auto dst2 = dst + (4 * i + 2) * strideDst + k * hpDst;
            auto dst3 = dst + (4 * i + 3) * strideDst + k * hpDst;

            memcpy(dst0, srcBase, hpDst * sizeof(float));
            memcpy(dst1, srcBase + hpDst, hpDst * sizeof(float));
            memcpy(dst2, srcBase + 2 * hpDst, hpDst * sizeof(float));
            memcpy(dst3, srcBase + 3 * hpDst, hpDst * sizeof(float));
        }
    }
}

static void _onlineReorderWeightKernelSumH8ToH32(float* dst, float* src, int blockNum, int hpSrc, int hpDst,
                                                 int ocNeedReorder, int ocPreserve) {
    // hpDst = 4 * hpSrc
    // src shape: [huSrc, blockNum, hpSrc], where huSrc = huDst * 4
    // dst shape: [huDst, blockNum, hpDst]

    auto huDst = UP_DIV(ocNeedReorder, hpDst);

    auto strideSrc = blockNum * hpSrc;
    auto strideDst = blockNum * hpDst;

    for (int i = 0; i < huDst; ++i) {
        for (int k = 0; k < blockNum; ++k) {
            auto dstBase = dst + i * strideDst + k * hpDst;

            const int sourceTiles = ALIMIN(4, UP_DIV(ocNeedReorder, hpSrc) - 4 * i);
            for (int j = 0; j < sourceTiles; ++j) {
                memcpy(dstBase + j * hpSrc, src + (4 * i + j) * strideSrc + k * hpSrc,
                       hpSrc * sizeof(float));
            }
            if (sourceTiles < 4) {
                memset(dstBase + sourceTiles * hpSrc, 0, (4 - sourceTiles) * hpSrc * sizeof(float));
            }
        }
    }

    if (ocPreserve) {
        memcpy(dst + huDst * strideDst, src + UP_DIV(ocNeedReorder, hpSrc) * strideSrc,
               ROUND_UP(ocPreserve, hpSrc) * blockNum * sizeof(float));
    }
}

ErrorCode DenseConvInt8TiledExecutor::onExecute(const std::vector<Tensor*>& inputs,
                                                const std::vector<Tensor*>& outputs) {
    if (mResourceInt8->mWeightQuantInfoMode && !mResourceInt8->mCompactReady.load(std::memory_order_acquire)) {
        return INVALID_VALUE;
    }
    if (mResourceInt8->mWeightQuantInfoMode &&
        (!_compactResourceCompatible(backend(), *mResourceInt8) || !_compactFpEnvironment())) {
        return NOT_SUPPORT;
    }
    std::atomic<bool> compactWorkerFailed{false};
    auto compactFailure = &compactWorkerFailed;
    int weightQuantInfoMode = mResourceInt8->mWeightQuantInfoMode;
    const auto input = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    auto dynamicOption = static_cast<CPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption % 8;

    int UNIT = mGemmUnits[0];
    int SRC_UNIT = mGemmUnits[1];
    int DST_XUNIT = mGemmUnits[2];
    auto blitProc = mRelatedFunctions.MNNPackC4Int8ForMatMul_A;
    const int plane = output->batch() * mIm2ColParamter.oh * mIm2ColParamter.ow;
    const int batch = input->batch();
    const int PackUnit = gcore->pack;
    const int dstZStep = plane * PackUnit;
    const int ocDiv4 = UP_DIV(output->channel(), PackUnit);
    const int ocUp4 = ROUND_UP(output->channel(), PackUnit);
    const int ocUpHp = ROUND_UP(output->channel(), UNIT);
    const auto kernelCountUnit = mIm2ColParamter.kernelCountUnit;
    const auto unitColBufferSize = kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    const auto colBufferSize = unitColBufferSize * mIm2ColCount;
    auto dstBytes = static_cast<CPUBackend*>(backend())->getBytes(backend(), output);
    const int blockL = kernelCountUnit / mBlockNum; // source depthQuad for each block.
    const int kxky = mIm2ColParamter.kernelX * mIm2ColParamter.kernelY;
    const int blocklu = blockL / kxky; // UP_DIV(ic,src_unit) per block
    const int oc = output->channel();
    const int ic = input->channel();
    float weightBytes = 1.f;
    int weightStepY = weightBytes * (UNIT * SRC_UNIT);
    int inputPlane = batch * input->width() * input->height();

    auto im2colPtr = mTempIm2ColBuffer->host<int8_t>();
    if (SRC_UNIT > PackUnit) {
        memset(im2colPtr, 0, mTempIm2ColBuffer->size());
    }
    auto weightDataPtr = mResourceInt8->mWeightInt8->host<int8_t>();
    auto srcKernelSumPtr = (int8_t*)mTempSrcSum.ptr();
    auto im2colSrc = input->host<uint8_t>();
    auto outputDataPtr = output->host<int8_t>();
    uint8_t* biasPtr = nullptr;
    int32_t inputZeroPoint = 0;
    int im2colBytes = mIm2ColBasedInt8 == true ? 1 : gcore->bytes;
    // Decode (plane==1) defers the inputzero-fused bias update into the OC-parallel
    // section: each GEMM thread updates exactly the bias slice it later reads.
    float fusedInputZeroF = 0.f;
    bool deferFusedBias = false;

    // Additional Adjustments for 4Bit Ptq model
    if (m4BitPtq) {
        outputDataPtr = (int8_t*)mTempOutput.ptr();
        dstBytes = gcore->bytes;
    }

    if (nullptr != mMutableResource.get()) {
        biasPtr = mMutableResource->mBiasFloat->host<uint8_t>();
        inputZeroPoint = mMutableResource->mInputZeroPoint;
        if (mBatchQuantInfo.get()) {
            float scalein = TensorUtils::getQuantInfo(inputs[0])[0];
            float scaleou = TensorUtils::getQuantInfo(outputs[0])[0];
            if (true == m4BitPtq) {
                scaleou = 1;
            }
            auto scaleX = scalein / scaleou;
            for (int i = 0; i < DST_XUNIT; ++i) {
                mBatchQuantInfo->host<float>()[i] = scaleX;
            }
        }
    }

    // Declare variables used in dynamic quantization
    int threads = static_cast<CPUBackend*>(backend())->computeThreadNumber(plane);
    if (mMixedKernel) {
        // Must match the clamp in onResize: division layout, temp buffers and the concurrency
        // launch below all assume the same thread count.
        threads = ALIMAX(threads, 4);
    }
    int dropBranch = 0;

#ifdef MNN_LOW_MEMORY
    auto BatchAsyDynamicQuant = [&](uint8_t* floatPtr, int32_t& inputZero, uint8_t* inputDequantScale, int LDiv4,
                                    int eCount, int innerSide, int32_t availableThreads, int8_t* dstInt8,
                                    uint8_t* inputDequantBias, int tId, bool deferBiasUpdate = false) {
        // if mIm2ColBasedInt8=false, input shape: [kernelsize,mBlockNum,blocklu,EP,LP]
        // if mIm2ColBasedInt8=true,  input shape: [ic/pack,EP,pack]
        auto scalePtr = (float*)inputDequantScale;
        auto zeroPtr = (float*)inputDequantBias;
        int scaleCount = mSizeInputBlockQuant;
        int kernelsize = 1;
        if (!mIm2ColBasedInt8) {
            kernelsize = kxky;
        }

        auto minPtr = mTempMaxMinValueBuffer.ptr() + tId * scaleCount * gcore->bytes;
        auto maxPtr = mTempMaxMinValueBuffer.ptr() + tId * scaleCount * gcore->bytes + (scaleCount / 2) * gcore->bytes;
        auto qscale = (float*)(mQScaleZero.ptr() + tId * scaleCount * sizeof(float));
        auto qbias =
            (float*)(mQScaleZero.ptr() + tId * scaleCount * sizeof(float) + (scaleCount / 2) * sizeof(float));

        size_t info[9] = {(size_t)mInputBlockNum,
                          (size_t)eCount,
                          (size_t)innerSide,
                          (size_t)DST_XUNIT,
                          (size_t)SRC_UNIT,
                          (size_t)kernelsize,
                          (size_t)blocklu,
                          0,
                          0};
        if (mIm2ColBasedInt8) {
            info[6] = LDiv4 / mInputBlockNum;
        }
        if (mToFuseInputbias2Bias) {
            info[7] = 1;
        }
        if (mIm2ColParamter.padX > 0 || mIm2ColParamter.padY > 0) {
            info[8] = 1;
        }
        // scale&bias:float32
        gcore->MNNAsyQuantInfo(scalePtr, zeroPtr, qscale, qbias, (float*)minPtr, (float*)maxPtr, (float*)floatPtr,
                               info);

        // quant: float->int8_t
        if (!mToFuseInputbias2Bias) {
            gcore->MNNAsyQuantFunc(dstInt8, (float*)floatPtr, qscale, qbias, info);
        } else {
            auto sizeDiv4 = UP_DIV(eCount * LDiv4 * innerSide, PackUnit);
            mQuantFunc((float*)floatPtr, dstInt8, sizeDiv4, qscale, -128, 127, qbias, 0);
        }

        if (mToFuseInputbias2Bias) { // Decode
            inputZero = static_cast<int32_t>(roundf(qbias[0]));
            auto updatedBiasPtr = (float*)(mBiasBufferFusedInputzero.ptr() + tId * ocUpHp * sizeof(float));
            if (deferBiasUpdate) {
                // Bias slice update runs inside the OC-parallel GEMM section instead.
                fusedInputZeroF = -qbias[0] * scalePtr[0];
                deferFusedBias = true;
            } else {
                auto matmulBiasPtr = mResourceInt8->mOriginBias->host<float>();
                auto weightKernelSum = mResourceInt8->mWeightKernelSum->host<float>();
                auto inputZeroF = -qbias[0] * scalePtr[0];
                gcore->MNNDynamicUpdateConvBiasScale(updatedBiasPtr, matmulBiasPtr, weightKernelSum, &inputZeroF,
                                                     UP_DIV(ocUpHp, 4));
            }
            biasPtr = (uint8_t*)updatedBiasPtr;
            auto unitsize = mBatchQuantInfo->length(1) / (2 * (int)sizeof(float));
            auto inputScale = scalePtr[0];
            for (int i = 0; i < unitsize; ++i) {
                ((float*)inputDequantScale)[i] = inputScale;
            }
        }
    };

    auto BatchSymDynamicQuant = [&](uint8_t* floatPtr, int32_t& inputZero, uint8_t* inputDequantScale, int LU, int EP,
                                    int LP, int32_t availableThreads, int8_t* dstInt8, int tId) {
        auto quantPtr = mQScaleZero.ptr() + tId * mSizeInputBlockQuant * sizeof(float);
        auto maxPtr = mTempMaxMinValueBuffer.ptr() + tId * mSizeInputBlockQuant * gcore->bytes;

        // compute sum and absmax
        int divlu = UP_DIV(LU, availableThreads);
        MNN_CONCURRENCY_BEGIN(tIdx, ALIMIN(availableThreads, UP_DIV(LU, divlu))) {
            auto exeLu = ALIMIN(divlu, LU - tIdx * divlu);
            auto batchMax = reinterpret_cast<float*>(maxPtr + tIdx * EP * gcore->bytes);
            auto ptr_ = reinterpret_cast<float*>(floatPtr + tIdx * divlu * gcore->bytes * EP * LP);
            gcore->MNNAbsMax((float*)ptr_, batchMax, exeLu, EP, LP);
        }
        MNN_CONCURRENCY_END();

        // Compute quant scale
        gcore->MNNQuantScale((float*)maxPtr, (float*)quantPtr, (float*)inputDequantScale, availableThreads, EP);

        // quant
        auto scale_ptr = reinterpret_cast<float*>(quantPtr);
        gcore->MNNDynamicQuant((float*)floatPtr, dstInt8, scale_ptr, LU, EP, LP, nullptr);
        inputZero = 0;
    };

    if (mResourceInt8->mDynamicQuant) {
        biasPtr = mResourceInt8->mOriginBias->host<uint8_t>();
    }
    if (mIm2ColBasedInt8 && mResourceInt8->mDynamicQuant) {
        int icDiv4 = UP_DIV(input->channel(), PackUnit);
        if (mUseBatchQuan) {
            int availthreads = (icDiv4 > mThreadNums && inputPlane > 255) ? mThreadNums : 1;
            if (dynamicOption != 2) {
                BatchSymDynamicQuant(input->host<uint8_t>(), inputZeroPoint, mBatchQuantInfo->host<uint8_t>(), icDiv4,
                                     inputPlane, PackUnit, availthreads, mQuantInput->host<int8_t>(), 0);
            } else {
                BatchAsyDynamicQuant(input->host<uint8_t>(), inputZeroPoint, mBatchQuantInfo->host<uint8_t>(), icDiv4,
                                     inputPlane, PackUnit, availthreads, mQuantInput->host<int8_t>(),
                                     mBatchQuantInfo->host<uint8_t>() + mBatchQuantInfo->stride(0) / 2, 0);
            }
        } else {
            BatchAsyDynamicQuant(input->host<uint8_t>(), inputZeroPoint, mBatchQuantInfo->host<uint8_t>(), icDiv4,
                                 inputPlane, PackUnit, 1, mQuantInput->host<int8_t>(),
                                 mBatchQuantInfo->host<uint8_t>() + mBatchQuantInfo->stride(0) / 2, 0,
                                 plane == 1 && mSplitByOc);
        }
        im2colSrc = mQuantInput->host<uint8_t>();
    }

    if (mOnlineReorderWeightSme && plane > 1) {
        const int sourceQuantInfoMode = mResourceInt8->mWeightQuantInfoMode;
        const int sourceQuantInfoBytes = sourceQuantInfoMode ? (int)sizeof(int16_t) : (int)sizeof(float);
        const size_t mainChannels = ROUND_UP(mOcMain, GEMM_INT8_UNIT_SME2_128);
        size_t mainPayloadBytes = mainChannels * mBlockNum * blockL * SRC_UNIT;
        if (mResourceInt8->mWeightBits == 4) {
            mainPayloadBytes /= 2;
        }
        const size_t sourceBranchOffset = mainPayloadBytes + mainChannels * mBlockNum * 2 * sourceQuantInfoBytes;
        const size_t destBranchOffset = sourceBranchOffset;
        _onlineReorderWeightPackH128ToH32((int8_t*)mWeight4Prefill.ptr(), weightDataPtr, GEMM_INT8_UNIT_SME2_128, UNIT,
                                          UP_DIV(mOcMain, GEMM_INT8_UNIT_SME2_128), mBlockNum, blockL, SRC_UNIT,
                                          mResourceInt8->mWeightBits == 4, sourceQuantInfoMode);

        int kernelSumMainSize = 0;
        int kernelSumBranchSize = 0;
        if (dstBytes > 1 && mInputBlockNum > 1) {
            _onlineReorderWeightKernelSumH128ToH32((float*)mWeightKernelSum4Prefill.ptr(),
                                                   mResourceInt8->mWeightKernelSum->host<float>(), mBlockNum,
                                                   GEMM_INT8_UNIT_SME2_128, UNIT, mOcMain);
            kernelSumMainSize = ROUND_UP(mOcMain, UNIT) * mBlockNum * sizeof(float);
            kernelSumBranchSize = ROUND_UP(mOcBranch, 8) * mBlockNum * sizeof(float);
        }

        // If change the workload distribution among SME and NEON cores.
        if (mMixedKernel && mRatioDecode != mRatioPrefill) {
            // Don't change mOcMain&mOcBranch here.
            int tmpMain = mOcMain;
            int tmpBranch = mOcBranch;
            calculateSmeNeonWorkDivision(tmpMain, tmpBranch, mDividesTmp, oc, threads, PackUnit, plane, mRatioPrefill,
                                         mSmeCores, threads);
            auto updatedSmeWork = mDividesTmp[mSmeCores];

            if (updatedSmeWork - mOriginSmeWork > 0 &&
                ((updatedSmeWork - mOriginSmeWork) * 4 % 8 == 0)) { // To ensure pack=4, dropBranch % 2 == 0
                dropBranch = updatedSmeWork - mOriginSmeWork;       // Ensure update "dropBranch" inner the loop.
                memcpy(mDivides.data(), mDividesTmp.data(), (threads + 1) * sizeof(float));
                dropBranch = mDivides[mSmeCores] - mOriginSmeWork;
                _onlineReorderWeightPackH8ToH32((int8_t*)(mWeight4Prefill.ptr() + destBranchOffset),
                                                weightDataPtr + sourceBranchOffset, blockL, SRC_UNIT,
                                                mResourceInt8->mWeightBits == 4, (int)(dropBranch * PackUnit / 8),
                                                mBlockNum, (mDivides[threads] - mDivides[mSmeCores]) * PackUnit,
                                                sourceQuantInfoMode);
            }

            if (dstBytes > 1 && mInputBlockNum > 1) {
                if (dropBranch > 0) {
                    // reorder
                    _onlineReorderWeightKernelSumH8ToH32(
                        (float*)(mWeightKernelSum4Prefill.ptr() + kernelSumMainSize),
                        (float*)(mResourceInt8->mWeightKernelSum->host<int8_t>() + kernelSumMainSize), mBlockNum, 8,
                        UNIT, dropBranch * PackUnit, (mDivides[threads] - mDivides[mSmeCores]) * PackUnit);
                }
            }
        }

        if (dropBranch == 0) {
            if (sourceQuantInfoMode && mOcBranch > 0) {
                _onlineReorderWeightPackH8ToH32((int8_t*)(mWeight4Prefill.ptr() + destBranchOffset),
                                                weightDataPtr + sourceBranchOffset, blockL, SRC_UNIT,
                                                mResourceInt8->mWeightBits == 4, 0,
                                                mBlockNum, mOcBranch, sourceQuantInfoMode);
            }
            if (kernelSumBranchSize > 0) {
                memcpy(mWeightKernelSum4Prefill.ptr() + kernelSumMainSize,
                       mResourceInt8->mWeightKernelSum->host<uint8_t>() + kernelSumMainSize, kernelSumBranchSize);
            }
        }

        weightDataPtr = (int8_t*)mWeight4Prefill.ptr();
    }
#endif
    if (mResourceInt8->mWeightBits == 4) {
        weightBytes = 0.5;
        weightStepY /= 2;
    } else if (mResourceInt8->mWeightBits == 3) {
        auto packedBytesPerOc = (SRC_UNIT * 3 + 7) / 8;
        weightBytes = static_cast<float>(packedBytesPerOc) / SRC_UNIT;
        weightStepY = UNIT * packedBytesPerOc;
    } else if (mResourceInt8->mWeightBits == 2) {
        weightBytes = 0.25f;
        weightStepY /= 4;
    }
    auto inputchannel = input->channel();
    SumByAxisParams sumParams;
    sumParams.oneScale = (mUseBatchQuan || dynamicOption == 2) ? 0 : 1;
    sumParams.SRC_UNIT = SRC_UNIT;
    sumParams.blockNum = mBlockNum;
    sumParams.DST_XUNIT = DST_XUNIT;
    sumParams.unitColBufferSize = unitColBufferSize;
    sumParams.kernelCountUnitDouble = kernelCountUnit;
    sumParams.valid = inputchannel % SRC_UNIT;
    sumParams.kernelxy = kxky;
    sumParams.LU = UP_DIV(inputchannel, SRC_UNIT);
    sumParams.inputBlock = (mInputBlockNum > 1) ? 1 : 0;
    std::vector<float> fakeInputScales(DST_XUNIT, 1.f);

    auto tileSplitFunction = [&](int tId, int eStartIndex, int eEndIndex, int estep) {
        if (mResourceInt8->mWeightQuantInfoMode && !_compactFpEnvironment()) {
            compactFailure->store(true, std::memory_order_relaxed);
            return;
        }
        auto ocDivThread = ocDiv4;
        float* reluPtr = mResourceInt8->mReluThreshold.data();
        float* accumbuff = nullptr;
        uint8_t* inputScale = nullptr;
        uint8_t* inputBias = nullptr;
        uint8_t* ptrInputScale = nullptr;
        uint8_t* ptrInputBias = nullptr;
        if (mBatchQuantInfo.get()) {
            if (mIm2ColBasedInt8) {
                inputScale = mBatchQuantInfo->host<uint8_t>();
                ptrInputScale = inputScale;
            }

            if (dynamicOption == 2 && mUseBatchQuan && mIm2ColBasedInt8) {
                inputBias = inputScale + mBatchQuantInfo->stride(0) / 2;
                ptrInputBias = inputBias;
            }
        } else {
            inputScale = (uint8_t*)fakeInputScales.data();
            ptrInputScale = inputScale;
        }
        if (mBlockNum > 1) {
            accumbuff = reinterpret_cast<float*>(mAccumBuffer->host<int8_t>() +
                                                 tId * mAccumBuffer->stride(0) * sizeof(int32_t));
        }
        float* ptrY = nullptr;
        if (dstBytes != 1) {
            ptrY = (mOnlineReorderWeightSme && mInputBlockNum > 1) ? (float*)mWeightKernelSum4Prefill.ptr()
                                                                   : mResourceInt8->mWeightKernelSum->host<float>();
        }
        QuanPostTreatParameters quanParam;
        quanParam.blockNum = mBlockNum;
        quanParam.weightQuantInfoMode = weightQuantInfoMode;
        int32_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        quanParam.indices = indices;
        if (dstBytes != 1) {
            quanParam.useInt8 = 0;
            quanParam.fp32minmax = reluPtr;
#ifdef MNN_USE_SSE
            if (!mBatchQuantInfo.get()) {
                quanParam.weightKernelSum = nullptr;
            }
#endif
        } else {
            quanParam.maxValue = mMutableResource->mClampMax;
            if (mResourceInt8->mRelu) {
                quanParam.minValue = mMutableResource->mOutputZeroPoint;
            } else {
                quanParam.minValue = mMutableResource->mClampMin;
            }
        }
        auto weightPtrTid = weightDataPtr;
        quanParam.weightKernelSum = ptrY;
        quanParam.biasFloat = reinterpret_cast<float*>(biasPtr);
        auto im2colDstThread = im2colPtr + tId * mTempIm2ColBuffer->stride(0);
        auto srcPtr = (int8_t const**)(mBlitInfo.ptr() + tId * mBlitInfoStride.first);
        auto el = (int32_t*)(srcPtr + mBlitInfoStride.second);
        auto xKernelSumPtrTid =
            reinterpret_cast<float*>(srcKernelSumPtr + tId * mBlockNum * DST_XUNIT * mIm2ColCount * sizeof(float));

        int32_t info[5];
        info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih * batch;
        info[2] = static_cast<int32_t>(unitColBufferSize);
        info[3] = mIm2ColParamter.strideX;
        for (int tIndex = eStartIndex; tIndex < eEndIndex; tIndex += estep) {
            const int xIndexStart = tIndex * DST_XUNIT * mIm2ColCount;
            auto outputInTilePtr = outputDataPtr + xIndexStart * PackUnit * dstBytes;
            int realDstCount = ALIMIN(plane - xIndexStart, DST_XUNIT * mIm2ColCount);
            ptrInputScale = (mUseBatchQuan && mIm2ColBasedInt8)
                                ? (inputScale + xIndexStart * mInputBlockNum * sizeof(float))
                                : inputScale;
            ptrInputBias =
                (inputBias != nullptr) ? (inputBias + xIndexStart * mInputBlockNum * sizeof(float)) : inputBias;
            // im2col
            auto im2colDst = im2colDstThread;
            auto res =
                ConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, xIndexStart, realDstCount,
                                                               mIm2ColParamter, (uint8_t*)im2colSrc, im2colBytes);
            int number = res.first;
            bool needZero = res.second;
            if (needZero && mIm2ColBasedInt8) {
#ifdef MNN_USE_SSE
                ::memset(im2colDst, inputZeroPoint + 128, colBufferSize);
#else
                ::memset(im2colDst, inputZeroPoint, colBufferSize);
#endif
            }
            info[0] = number;
            info[4] = realDstCount;
            if (mIm2ColBasedInt8 && number > 0) {
                blitProc(im2colDst, srcPtr, info, el);
            }
#ifdef MNN_LOW_MEMORY
            if (!mIm2ColBasedInt8) {
                if (needZero) {
                    ::memset(im2colDst, 0, mTempIm2ColBuffer->stride(0));
                }
                if (number > 0) {
                    if (SRC_UNIT > PackUnit && !needZero) {
                        memset(im2colDst, 0, mTempIm2ColBuffer->stride(0));
                    }
                    info[2] = realDstCount;
                    mRelatedFunctions.MNNGeneralIm2Col((float*)im2colDst, (float const**)srcPtr, info, el, SRC_UNIT,
                                                       PackUnit); // im2colDst: [lu, realDstCount, lp]
                }
                ptrInputScale = mBatchQuantInfo->host<uint8_t>() + tId * mBatchQuantInfo->stride(0);
                if (dynamicOption == 2) {
                    ptrInputBias = ptrInputScale + mBatchQuantInfo->stride(0) / 2;
                    BatchAsyDynamicQuant((uint8_t*)im2colDst, inputZeroPoint, ptrInputScale, kernelCountUnit,
                                         realDstCount, SRC_UNIT, 1,
                                         mQuantInput->host<int8_t>() + tId * mQuantInput->stride(0), ptrInputBias, tId);
                } else if (mUseBatchQuan) {
                    BatchSymDynamicQuant((uint8_t*)im2colDst, inputZeroPoint, ptrInputScale, kernelCountUnit,
                                         realDstCount, SRC_UNIT, 1,
                                         mQuantInput->host<int8_t>() + tId * mQuantInput->stride(0), tId);
                } else {
                    auto maxMinPtr = mTempMaxMinValueBuffer.ptr() + tId * 2 * gcore->bytes;
                    ptrInputBias = ptrInputScale + mBatchQuantInfo->stride(0) / 2;
                    BatchAsyDynamicQuant((uint8_t*)im2colDst, inputZeroPoint, ptrInputScale, kernelCountUnit,
                                         realDstCount, SRC_UNIT, 1,
                                         mQuantInput->host<int8_t>() + tId * mQuantInput->stride(0), ptrInputBias, tId);
                    quanParam.biasFloat = (float*)(mBiasBufferFusedInputzero.ptr() + tId * ocUpHp * sizeof(float));
                }
                im2colDst = mQuantInput->host<int8_t>() + tId * mQuantInput->stride(0);
            }
            if (mBlockNum > 1 && kxky > 1) {
                auto eU = UP_DIV(realDstCount, DST_XUNIT); // eU <= mIm2ColCount
                auto reorderBuffer = mReorderBuffer.ptr() + tId * colBufferSize;
                for (int k = 0; k < eU; ++k) {
                    int inside = blocklu * SRC_UNIT * ALIMIN(realDstCount - k * DST_XUNIT, DST_XUNIT);
                    auto dstbuffer = reorderBuffer + k * unitColBufferSize;
                    auto srcbuffer = im2colDst + k * unitColBufferSize;
                    for (int i = 0; i < mBlockNum; ++i) {
                        for (int j = 0; j < kxky; ++j) {
                            memcpy(dstbuffer + i * kxky * inside + j * inside,
                                   srcbuffer + i * inside + j * mBlockNum * inside, inside);
                        }
                    }
                }
                im2colDst = (int8_t*)reorderBuffer;
            }
#endif
            if (mResourceInt8->mWeightAsymmetricQuant) {
                MNN_ASSERT(mBatchQuantInfo.get() && mBatchQuantInfo->host<float>());
                mRelatedFunctions.MNNSumByAxisLForMatmul_A(xKernelSumPtrTid, im2colDst, (float*)ptrInputScale,
                                                           realDstCount, sumParams);
            } else {
                memset(xKernelSumPtrTid, 0, mBlockNum * DST_XUNIT * mIm2ColCount * sizeof(float));
            }
            auto ptrX = xKernelSumPtrTid;
            do {
                int step = ALIMIN(DST_XUNIT, realDstCount);
                quanParam.inputScale = (float*)ptrInputScale;
                quanParam.inputBias = (float*)ptrInputBias;
                if (mBlockNum > 1) {
                    memset(accumbuff, 0, UNIT * 4 * DST_XUNIT);
                    quanParam.accumBuffer = accumbuff;
                }
                quanParam.srcKernelSum = ptrX;
                mGemmKernel(outputInTilePtr, im2colDst, weightPtrTid, blockL, dstZStep * dstBytes, ocDivThread,
                            &quanParam, step);
                ptrX += (step * mBlockNum);
                realDstCount -= step;
                outputInTilePtr += DST_XUNIT * PackUnit * dstBytes;
                im2colDst += unitColBufferSize;
                ptrInputScale =
                    mUseBatchQuan ? (ptrInputScale + step * mInputBlockNum * sizeof(float)) : ptrInputScale;
                ptrInputBias = (ptrInputBias != nullptr) ? (ptrInputBias + step * mInputBlockNum * sizeof(float))
                                                         : ptrInputBias;
            } while (realDstCount > 0);
        }
    };
    auto ocSplitFunction = [&](int threads) { // Thread split by OC
        auto im2colDst = mTempIm2ColBuffer->host<int8_t>();
        auto srcPtr = (int8_t const**)(mBlitInfo.ptr());
        auto el = (int32_t*)(srcPtr + mBlitInfoStride.second);
        auto xKernelSumPtr = reinterpret_cast<float*>(mTempSrcSum.ptr());

        auto eU = UP_DIV(plane, DST_XUNIT);
        int32_t info[5];
        info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih * batch;
        info[2] = static_cast<int32_t>(unitColBufferSize);
        info[3] = mIm2ColParamter.strideX;

        float* reluPtr = mResourceInt8->mReluThreshold.data();
        const bool directDecodeInput = mIm2ColBasedInt8 && mMixedKernel && plane == 1 && inputPlane == 1 &&
                                       kxky == 1 && PackUnit == 8 && SRC_UNIT == 4 && ic % PackUnit == 0 &&
                                       mIm2ColParamter.padX == 0 && mIm2ColParamter.padY == 0 &&
                                       mIm2ColParamter.strideX == 1 && mIm2ColParamter.strideY == 1;
        if (directDecodeInput) {
            // At E=1, aligned NC8 input already has the compact [K/4, 1, 4] GEMM layout.
            im2colDst = reinterpret_cast<int8_t*>(im2colSrc);
        } else if (mIm2ColBasedInt8) { // im2col
            auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo(
                (const float**)srcPtr, el, 0, plane, mIm2ColParamter, (uint8_t*)im2colSrc, im2colBytes);
            int number = res.first;
            bool needZero = res.second;
            if (needZero) {
#ifdef MNN_USE_SSE
                ::memset(im2colDst, inputZeroPoint + 128, mTempIm2ColBuffer->size());
#else
                ::memset(im2colDst, inputZeroPoint, mTempIm2ColBuffer->size());
#endif
            }
            info[0] = number;
            info[4] = plane;
            if (number > 0) {
                blitProc(im2colDst, srcPtr, info, el);
            }
        }
#ifdef MNN_LOW_MEMORY
        if (false == mIm2ColBasedInt8) {
            int realDstCount = plane;
            int start = 0;
            auto ptrInputscale = mBatchQuantInfo->host<uint8_t>();
            auto ptrInputbias = ptrInputscale + mBatchQuantInfo->stride(0) / 2;
            auto int8Ptr = mQuantInput->host<int8_t>();
            int sizePacked = 0;
            auto im2colDstTmp = im2colDst;
            while (realDstCount > 0) {
                int work = std::min(realDstCount, DST_XUNIT);
                sizePacked += (work * SRC_UNIT * kernelCountUnit);
                auto res = ConvolutionTiledExecutor::turnIm2ColToBlitInfo(
                    (const float**)srcPtr, el, start, work, mIm2ColParamter, (uint8_t*)im2colSrc, im2colBytes);
                int number = res.first;
                bool needZero = res.second;
                if (needZero) {
                    ::memset(im2colDstTmp, 0, unitColBufferSize * gcore->bytes);
                }
                info[0] = number;
                info[2] = work;
                if (number > 0) { // im2col
                    mRelatedFunctions.MNNGeneralIm2Col((float*)im2colDstTmp, (float const**)srcPtr, info, el, SRC_UNIT,
                                                       PackUnit); // im2colDst: [lu, realDstCount, lp]
                }
                if (mUseBatchQuan || dynamicOption == 2) {
                    if (dynamicOption == 2) {
                        BatchAsyDynamicQuant((uint8_t*)im2colDstTmp, inputZeroPoint, ptrInputscale, kernelCountUnit,
                                             work, SRC_UNIT, 1, int8Ptr, ptrInputbias, 0);
                        ptrInputbias += (mInputBlockNum * work * sizeof(int32_t));
                    } else {
                        BatchSymDynamicQuant((uint8_t*)im2colDstTmp, inputZeroPoint, ptrInputscale, kernelCountUnit,
                                             work, SRC_UNIT, 1, int8Ptr, 0);
                    }
                    ptrInputscale += (mInputBlockNum * work * sizeof(int32_t));
                    int8Ptr += unitColBufferSize;
                }
                realDstCount -= work;
                start += work;
                im2colDstTmp += (unitColBufferSize * gcore->bytes);
            }
            if (!mUseBatchQuan && dynamicOption != 2) {
                BatchAsyDynamicQuant((uint8_t*)im2colDst, inputZeroPoint, ptrInputscale, kernelCountUnit, plane,
                                     SRC_UNIT, 1, mQuantInput->host<int8_t>(),
                                     ptrInputscale + plane * mInputBlockNum * sizeof(float), 0);
            }
            im2colDst = mQuantInput->host<int8_t>();
        }
        if (mBlockNum > 1 && kxky > 1) {
            for (int k = 0; k < eU; ++k) {
                int inside = blocklu * SRC_UNIT * ALIMIN(DST_XUNIT, plane - k * DST_XUNIT);
                auto dstbuffer = mReorderBuffer.ptr() + k * unitColBufferSize;
                auto srcbuffer = im2colDst + k * unitColBufferSize;
                for (int i = 0; i < mBlockNum; ++i) {
                    for (int j = 0; j < kxky; ++j) {
                        memcpy(dstbuffer + i * kxky * inside + j * inside,
                               srcbuffer + i * inside + j * mBlockNum * inside, inside);
                    }
                }
            }
            im2colDst = (int8_t*)mReorderBuffer.ptr();
        }
#endif
        if (mResourceInt8->mWeightAsymmetricQuant) {
            MNN_ASSERT(mBatchQuantInfo.get() && mBatchQuantInfo->host<float>());
            mRelatedFunctions.MNNSumByAxisLForMatmul_A(xKernelSumPtr, im2colDst, mBatchQuantInfo->host<float>(), plane,
                                                       sumParams);
        } else {
            memset(xKernelSumPtr, 0, mTileCount * mBlockNum * DST_XUNIT * mIm2ColCount * sizeof(float));
        }

        // Decode GEMV on heterogeneous core sets: a static equal OC split strands work on
        // slow cores and the barrier waits for them. Claim the NEON-packed region
        // [mDivides[mSmeCores], neonClaimEnd) dynamically (guided chunks) so fast cores
        // absorb it. SME-packed channels stay exclusive to SME kernels; every thread can
        // run the NEON kernel, so SME threads join the claiming once their share is done.
        // Only worthwhile with many threads on a saturated core set: with few threads the
        // static split is already balanced (or the OS has slack cores), and claiming only
        // adds atomic/call overhead and breaks contiguous per-thread weight streaming.
        const bool dynamicNeonClaim = mMixedKernel && plane == 1 && !m4BitPtq && threads >= 8 &&
                                      threads > gcore->perfCoreNumber;
        const int neonClaimEnd = ALIMIN(mDivides[threads], ocDiv4);
        const int payloadBytesPerUnit = ALIMAX(1, static_cast<int>(PackUnit * kernelCountUnit * weightBytes));
        const int minClaimUnits = ALIMAX(1, ALIMIN(4, 16384 / payloadBytesPerUnit));
        // The NEON branch kernel consumes weights in UNITBranch-channel packed tiles
        // ([W tile][quant tile]); weightPtrTid assumes every range starts on that tile
        // grid. Keep all claim boundaries multiples of UNITBranch/PackUnit units,
        // otherwise every channel of a misaligned chunk reads quant bytes as weights.
        const int claimAlignUnits = ALIMAX(1, GEMM_INT8_UNIT_ARM82 / PackUnit);
        std::atomic<int> neonClaim{dynamicNeonClaim ? mDivides[mSmeCores] : 0};

        auto ocGemmRange = [&](int tId, int rangeBegin, int rangeEnd, bool forceNeonKernel) {
            int ocIndex = PackUnit * rangeBegin;
            auto ocDivThread = ALIMIN(rangeEnd - rangeBegin, ocDiv4 - rangeBegin);
            if (ocIndex >= ocUp4 || ocDivThread <= 0) {
                return;
            }
            decltype(mGemmKernel) gemmInt8;
            if (mMixedKernel) {
                gemmInt8 = (tId < mSmeCores && !forceNeonKernel) ? mGemmKernels[0] : mGemmKernels[1];
            } else {
                gemmInt8 = mGemmKernel;
            }
            auto im2colDstThread = im2colDst;
            float* ptrY = nullptr;
            if (dstBytes != 1) {
                float* wkernelSum = (mOnlineReorderWeightSme && mInputBlockNum > 1 && plane > 1)
                                        ? (float*)mWeightKernelSum4Prefill.ptr()
                                        : mResourceInt8->mWeightKernelSum->host<float>();
                ptrY = wkernelSum + ocIndex * mInputBlockNum;
            }
            QuanPostTreatParameters quanParam;
            quanParam.blockNum = mBlockNum;
            quanParam.weightKernelSum = ptrY;
            quanParam.biasFloat = reinterpret_cast<float*>(biasPtr + ocIndex * 4);
            if (deferFusedBias) {
                // Deferred from the serial quant prologue: update exactly the bias
                // slice this range's GEMM reads; element-wise op keeps bits identical.
                gcore->MNNDynamicUpdateConvBiasScale(
                    reinterpret_cast<float*>(mBiasBufferFusedInputzero.ptr()) + ocIndex,
                    mResourceInt8->mOriginBias->host<float>() + ocIndex,
                    mResourceInt8->mWeightKernelSum->host<float>() + ocIndex, &fusedInputZeroF,
                    UP_DIV(static_cast<int>(ocDivThread) * PackUnit, 4));
            }
            int32_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
            quanParam.indices = indices;
            if (dstBytes != 1) {
                quanParam.useInt8 = 0;
                quanParam.fp32minmax = reluPtr;
#ifdef MNN_USE_SSE
                if (!mBatchQuantInfo.get()) {
                    quanParam.weightKernelSum = nullptr;
                }
#endif
            } else {
                quanParam.maxValue = mMutableResource->mClampMax;
                if (mResourceInt8->mRelu) {
                    quanParam.minValue = mMutableResource->mOutputZeroPoint;
                } else {
                    quanParam.minValue = mMutableResource->mClampMin;
                }
            }
            uint8_t* inputScale = nullptr; // input scale for batch dynamic quant.
            uint8_t* inputBias = nullptr;
            float* accumbuff = nullptr;
            if (mBatchQuantInfo.get()) {
                inputScale = mBatchQuantInfo->host<uint8_t>();
                if (dynamicOption == 2) {
                    inputBias = inputScale + mInputBlockNum * plane * sizeof(float);
                }
            } else {
                inputScale = (uint8_t*)fakeInputScales.data();
            }
            if (mBlockNum > 1) {
                accumbuff = reinterpret_cast<float*>(mAccumBuffer->host<int8_t>() +
                                                     tId * mAccumBuffer->stride(0) * sizeof(int32_t));
            }

            auto outputInTilePtr = outputDataPtr + ocIndex * plane * dstBytes;

            auto weightSrc = weightDataPtr;
            int activeQuantInfoMode = weightQuantInfoMode;
            if ((tId >= mSmeCores || forceNeonKernel) && dropBranch == 0 && mMixedKernel &&
                !(mResourceInt8->mWeightQuantInfoMode && plane > 1)) {
                weightSrc = mResourceInt8->mWeightInt8->host<int8_t>();
                activeQuantInfoMode = mResourceInt8->mWeightQuantInfoMode;
            }
            quanParam.weightQuantInfoMode = activeQuantInfoMode;
            const int activeQuantInfoBytes = activeQuantInfoMode ? (int)sizeof(int16_t) : (int)sizeof(float);
            auto weightPtrTid =
                weightSrc + static_cast<int32_t>(ocIndex * mBlockNum * blockL * SRC_UNIT * weightBytes +
                                                 ocIndex * 2 * mBlockNum * activeQuantInfoBytes);

            int realDstCount = plane;
            auto ptrX = xKernelSumPtr;
            do {
                int step = ALIMIN(DST_XUNIT, realDstCount);
                quanParam.inputScale = (float*)inputScale;
                quanParam.inputBias = (float*)inputBias;
                quanParam.srcKernelSum = ptrX;
                if (mBlockNum > 1) {
                    memset(accumbuff, 0, UNIT * 4 * DST_XUNIT);
                    quanParam.accumBuffer = accumbuff;
                }
                gemmInt8(outputInTilePtr, im2colDstThread, weightPtrTid, blockL, dstZStep * dstBytes, ocDivThread,
                         &quanParam, step);
                ptrX += (step * mBlockNum);
                realDstCount -= step;
                outputInTilePtr += DST_XUNIT * PackUnit * dstBytes;
                im2colDstThread += unitColBufferSize;
                inputScale = mUseBatchQuan ? (inputScale + mInputBlockNum * step * sizeof(float)) : inputScale;
                inputBias =
                    (inputBias != nullptr) ? (inputBias + mInputBlockNum * step * sizeof(float)) : inputBias;
            } while (realDstCount > 0);
        };

        MNN_CONCURRENCY_BEGIN(tId, threads) {
            const bool compatibleFp = !mResourceInt8->mWeightQuantInfoMode || _compactFpEnvironment();
            if (!compatibleFp) {
                compactFailure->store(true, std::memory_order_relaxed);
            } else if (dynamicNeonClaim) {
                if (tId < mSmeCores) {
                    ocGemmRange((int)tId, mDivides[tId], mDivides[tId + 1], false);
                }
                int cur = neonClaim.load(std::memory_order_relaxed);
                while (cur < neonClaimEnd) {
                    int chunk = ALIMAX(minClaimUnits, (neonClaimEnd - cur) / (2 * threads));
                    chunk = ROUND_UP(chunk, claimAlignUnits);
                    chunk = ALIMIN(chunk, neonClaimEnd - cur);
                    if (neonClaim.compare_exchange_weak(cur, cur + chunk, std::memory_order_acq_rel,
                                                        std::memory_order_relaxed)) {
                        ocGemmRange((int)tId, cur, cur + chunk, true);
                        cur = neonClaim.load(std::memory_order_relaxed);
                    }
                }
            } else {
                ocGemmRange((int)tId, mDivides[tId], mDivides[tId + 1], false);
            }
        }
        MNN_CONCURRENCY_END();

    };
    if (!mSplitByOc) {
        MNN_CONCURRENCY_BEGIN(tId, threads) {
            if (mDivides[tId + 1] - mDivides[tId] > 0) {
                tileSplitFunction((int)tId, mDivides[tId], mDivides[tId + 1], 1);
            }
        }
        MNN_CONCURRENCY_END();
    } else {
        ocSplitFunction(threads);
    }
    if (compactWorkerFailed.load(std::memory_order_relaxed)) {
        return NOT_SUPPORT;
    }
    if (m4BitPtq) {
        std::vector<float> outputQuantScale(PackUnit);
        float s = TensorUtils::getQuantInfo(outputs[0])[0] == 0 ? 0 : 1.f / TensorUtils::getQuantInfo(outputs[0])[0];
        for (int i = 0; i < PackUnit; ++i) {
            outputQuantScale[i] = s;
        }
        float zero_ = TensorUtils::getQuantInfo(outputs[0])[1];
        mQuantFunc((float*)mTempOutput.ptr(), output->host<int8_t>(), plane * ocDiv4, outputQuantScale.data(),
                   mResourceInt8->mClampMin, mResourceInt8->mClampMax, &zero_, 0);
    }
    return NO_ERROR;
}
} // namespace MNN
