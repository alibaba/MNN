#include "ConvInt83x3.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "ConvOpt.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "Int8FunctionsOpt.h"
#include "CommonOptFunction.h"
#include "WinogradHelper.hpp"
#include "MNN/AutoTime.hpp"
#include <map>
#include <string>
#include <memory>
#include <vector>
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

static const int BLOCK_UNIT = MNN::WinogradHelper::L2K3::blockUnit();
static const int SRC_UNIT = MNN::WinogradHelper::L2K3::srcUnit();
static const int DST_UNIT = MNN::WinogradHelper::L2K3::dstUnit();
static const int BLOCK_UNIT2 = BLOCK_UNIT * BLOCK_UNIT;
static const int GEMM_TILE_UNIT = DST_XUNIT;

using namespace MNN::WinogradHelper::L2K3;

namespace MNN {

/*
 w0: original int8 weight (3x3), 1x
 w1_{int8, int16}: winograd transformed (2-dim) weight (4x4, two dim), (4x4)/(3x3) = 1.77x for int8, 3.54x for int16
 w2_{int8, int16}: winograd transformed (1-dim, horizontal and vertical) weight (2x3x4, one dim), (2x3x4)/(3x3) = 2.66x, 5.32x for int16
 w3: original int16 weight (3x3), (int16 / int8) = 2x
 
 1. Memory usage Analysis
 [[static memory], [dynamic memory]]
 |-- int8
      |---- combine1D2D
      |         |--- Online:      [[w0], [w1_int8, w2_int8]]        1x
      |         |--- ExtraOnline: [[w0, w1_int8], [w2_int8]]        2.77x
      |         |--- Offline:     [[w0, w1_int8, w2_int8], []]      5.44x
      |
      |------ only2D
                |--- Online:      [[w0], [w1_int8]]                 1x
                |--- Offline:     [[w0, w1_int8], []]               2.77x
 Note:
   N1: when w{1,2}_{int8, int16} in [dynamic memory], we do corresponding transform on runtime,
       so extra calculation needed.
   N2: combine1D2D for 3x3 winograd F(2,3), bottom-right leftover gemm only need (2x2, upper-left) kernel,
       so w0 is (2x2)/(3x3) = 0.44x. (int8, combine1D2D, Offline) is 4.4x + 0.44x = 4.84x
 
 2. Calculation load analysis
   2.1 weight transformed load
        3x3 origin -> 4x4 transformed 2d:   oc * ic * (4 * 3 * 3 + 4 * 3 * 4) = oc * ic * 84
        3x3 origin -> 2x3x4 transformed 1d: oc * ic * 2 * (4 * 3 * 3) = oc * ic * 72
   2.2 combine1D2D saved (when odd height and width)
        bottom-right corner: (oc * ic * 4 * 4) - (oc * ic * 2 * 2) = oc * ic * 12
        bottom and right 1d unit: N * ((oc * ic * 4 * 4) - (oc * ic * 2 * 4)) = oc * ic * 8 * N, N = H / 2 + W / 2 ~= feature map size
   2.3 result
        In theory, combine1D2D can speed up execution when N (feature map size) >= 8.
        But combine1D2D algorithm is more significant for small feature map.
        Now we will not do combine1D2D transform on runtime for simplify (but we may try this in the future).
 3. Conclusion
        See Fig1 (above figure)
 */
ConvInt83x3::ComputeStrategy ConvInt83x3::getComputeStrategy(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) const {
    /* TODO: Decide better strategy according to bit-num, max compute on per-cpu core, memory mode in backend config and overral model param.
     */
    ComputeStrategy strategy;
    
    int outputCount = outputs[0]->channel(), srcCount = inputs[0]->channel();
    int oh = outputs[0]->height(), ow = outputs[0]->width();
    int threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    
    auto maxMacPerCoreFunc = [=](int unitNum, int blockSize) {
        int tileNum = UP_DIV(unitNum, GEMM_TILE_UNIT), tileRemain = tileNum % threadNumber;
        int unitRemain = unitNum % GEMM_TILE_UNIT, lastGemmTileUnit = (unitRemain == 0 ? GEMM_TILE_UNIT : unitRemain);
        size_t maxMacPerCore = 0;
        if (threadNumber == 1) {
            maxMacPerCore = ALIMAX(tileNum - 1, 0) * blockSize * GEMM_TILE_UNIT * outputCount * srcCount;
            maxMacPerCore += blockSize * lastGemmTileUnit * outputCount * srcCount;
            return maxMacPerCore;
        }
        maxMacPerCore = tileNum / threadNumber * blockSize * GEMM_TILE_UNIT * outputCount * srcCount;
        if (tileRemain != 0) {
            maxMacPerCore += (tileRemain - 1) * UP_DIV(blockSize, threadNumber) * GEMM_TILE_UNIT * outputCount * srcCount;
            maxMacPerCore += UP_DIV(blockSize, threadNumber) * lastGemmTileUnit * outputCount * srcCount;
        }
        return maxMacPerCore;
    };
    size_t macOnly2DPerCore = maxMacPerCoreFunc(UP_DIV(oh, DST_UNIT) * UP_DIV(ow, DST_UNIT), BLOCK_UNIT2);
    size_t mac1D2DPerCore = 0;
    {
        int hUnit2D = oh / DST_UNIT, wUnit2D = ow / DST_UNIT;
        int hRemain = oh % DST_UNIT, wRemain = ow % DST_UNIT, leftOverPoint = hRemain * wRemain;
        mac1D2DPerCore = maxMacPerCoreFunc(hUnit2D * wUnit2D, BLOCK_UNIT2);
        mac1D2DPerCore += maxMacPerCoreFunc(hUnit2D * wRemain, BLOCK_UNIT * 3);
        mac1D2DPerCore += maxMacPerCoreFunc(wUnit2D * hRemain, BLOCK_UNIT * 3);
        mac1D2DPerCore += 9 * leftOverPoint * UP_DIV(outputCount, threadNumber) * srcCount;
    }
    //MNN_PRINT("macOnly2DPerCore: %lu, mac1D2DPerCore: %lu\n", macOnly2DPerCore, mac1D2DPerCore);
    
    auto memoryMode = static_cast<CPUBackend*>(backend())->memoryMode();
    if (mac1D2DPerCore < macOnly2DPerCore && memoryMode == BackendConfig::Memory_High) {
        strategy.unitType = ComputeStrategy::D2_D1;
    } else {
        strategy.unitType = ComputeStrategy::D2;
    }
    strategy.transPhase = ComputeStrategy::Offline;
    
    return strategy;
}

void ConvInt83x3::weightContent(bool trans2d, bool trans1d) {
    const int threadNumber = ((CPUBackend*)backend())->threadNumber();
    // winograd weight transform 2d
    auto transformWeightFunc = [=](std::shared_ptr<const Tensor> weightOrigin, std::shared_ptr<Tensor> weight) {
        const int totalCount = weight->length(1) * weight->length(2); // outputCountUnit * srcCountUnit
        const int unitSize = weight->length(3); // unitO(4) * unitI(4 or 8)
        const int weightOffset = weight->stride(0);
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            int step = UP_DIV(totalCount, threadNumber), start = (int)tId * step, end = ALIMIN(start + step, totalCount);
            for (int z = start; z < end; ++z) {
                auto src = weightOrigin->host<int8_t>() + unitSize * 9 * z;
                auto dst = weight->host<int8_t>() + unitSize * z;
                weightTransform2D<int8_t, 16>(src, dst, unitSize, weightOffset, unitSize / 16);
            }
        } MNN_CONCURRENCY_END();
    };
    // winograd weight transform 1d
    auto transformWeightExtraFunc = [=](std::shared_ptr<Tensor> weightOrigin, std::shared_ptr<Tensor> weight) {
        const int totalCount = weight->length(2) * weight->length(3); // outputCountUnit * srcCountUnit
        const int unitSize = weight->length(4); // unitO(4) * unitI(4 or 8)
        const int weightOffset = weight->stride(1);
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            int step = UP_DIV(totalCount, threadNumber), start = (int)tId * step, end = ALIMIN(start + step, totalCount);
            for (int z = start; z < end; ++z) {
                auto src = weightOrigin->host<int8_t>() + unitSize * 9 * z;
                auto dst = weight->host<int8_t>() + unitSize * z;
                for (int i = 0; i < 2 /* instead of 3 */; ++i) { // special case for F(2,3) winograd 1D, only left 3x2 kernel be used.
                    weightTransform1D<int8_t, 16>(src + i * unitSize, dst + i * BLOCK_UNIT * weightOffset, unitSize * 3, weightOffset, unitSize / 16);
                }
                dst += weight->stride(0);
                for (int i = 0; i < 2 /* instead of 3 */; ++i) { // special case for F(2,3) winograd 1D, only upper 2x3 kernel be used.
                    weightTransform1D<int8_t, 16>(src + i * 3 * unitSize, dst + i * BLOCK_UNIT * weightOffset, unitSize, weightOffset, unitSize / 16);
                }
            }
        } MNN_CONCURRENCY_END();
    };
        
    if (trans2d) {
        transformWeightFunc(mWeightInt8, mWeight);
    }
    if (trans1d) {
        transformWeightExtraFunc(mWeightInt8, mWeightExtra);
    }
}

ErrorCode ConvInt83x3::tensorMemoryOnStrategyChange(ComputeStrategy* oldStrategy, ComputeStrategy* newStrategy,
                                                    const std::vector<Tensor *> &inputs,
                                                    const std::vector<Tensor *> &outputs,
                                                    std::vector<Tensor *> *dynamicAllocTensors) {
    if (oldStrategy == nullptr && newStrategy == nullptr) {
        return INVALID_VALUE;
    }
    if (newStrategy == nullptr) {
        backend()->onReleaseBuffer(mWeightInt8.get(), Backend::STATIC);
        if (oldStrategy->transPhase != ComputeStrategy::Online) {
            backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
            if (oldStrategy->unitType == ComputeStrategy::D2_D1 && oldStrategy->transPhase == ComputeStrategy::Offline) {
                backend()->onReleaseBuffer(mWeightExtra.get(), Backend::STATIC);
            }
        }
        return NO_ERROR;
    }
    
    bool trans2d = false, trans1d = false;
// -1 if x is null, 1 if attr of x is equal to ComputeStrategy::value, 0 otherwise.
#define ATTR_CHECK(x, attr, value) (x == nullptr ? -1 : (x->attr == ComputeStrategy::value ? 1 : 0))
    int oldTransPhaseIsOnline = ATTR_CHECK(oldStrategy, transPhase, Online);
    bool newTransPhaseIsOnline = (newStrategy->transPhase == ComputeStrategy::Online);

#define ALLOC_CHECK(res) if(!(res)) { return OUT_OF_MEMORY; }
    if (newTransPhaseIsOnline) {
        if (oldTransPhaseIsOnline == 0) {
            backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
        }
        dynamicAllocTensors->push_back(mWeight.get());
    } else if (!newTransPhaseIsOnline && oldTransPhaseIsOnline != 0) {
        ALLOC_CHECK(backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC));
        trans2d = true;
    }
    
    int oldIsCombine1D2D = ATTR_CHECK(oldStrategy, unitType, D2_D1);
    int oldTransPhaseIsOffline = ATTR_CHECK(oldStrategy, transPhase, Offline);
    bool newIsCombine1D2D = (newStrategy->unitType == ComputeStrategy::D2_D1);
    bool newTransPhaseIsOffline = (newStrategy->transPhase == ComputeStrategy::Offline);
    
    if ((oldIsCombine1D2D == 1 && oldTransPhaseIsOffline == 1) && (!newIsCombine1D2D || !newTransPhaseIsOffline)) {
        backend()->onReleaseBuffer(mWeightExtra.get(), Backend::STATIC);
    }
    if (newIsCombine1D2D) {
        // compute shape of mWeightExtra and mWeightLeftOver based on feature map size
        auto input = inputs[0], output = outputs[0];
        const int kernel = 3, unitI = 8;
        int outputCountUnit = UP_DIV(output->channel(), 4), srcCountUnit = UP_DIV(input->channel(), unitI);
        int usedKernelX = input->width() - (output->width() / DST_UNIT * DST_UNIT - mPadX);
        int usedKernelY = input->height() - (output->height() / DST_UNIT * DST_UNIT - mPadY);
        usedKernelX = (usedKernelX == 0 ? kernel : usedKernelX);
        usedKernelY = (usedKernelY == 0 ? kernel : usedKernelY);
        
        mWeightExtra.reset(Tensor::createDevice<int8_t>({2, ALIMAX(usedKernelX, usedKernelY) * BLOCK_UNIT, outputCountUnit, srcCountUnit, unitI * 4}));
        mWeightLeftOver.reset(Tensor::createDevice<int8_t>({usedKernelX * usedKernelY, outputCountUnit, srcCountUnit, unitI * 4}));
        // do memory alloc
        if (newTransPhaseIsOffline == 1) {
            ALLOC_CHECK(backend()->onAcquireBuffer(mWeightExtra.get(), Backend::STATIC));
            trans1d = true;
        } else {
            dynamicAllocTensors->push_back(mWeightExtra.get());
        }
        dynamicAllocTensors->push_back(mWeightLeftOver.get());
    }
    
    if (trans2d || trans1d) {
        weightContent(trans2d, trans1d);
    }
    
    return NO_ERROR;
}

ConvInt83x3::ConvInt83x3(Backend *backend, const MNN::Convolution2D *convParam, const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs) : CPUConvolution(convParam->common(), backend) {
    mActBits = convParam->symmetricQuan()->nbits();
    
    if (((CPUBackend*)backend)->memoryMode() == BackendConfig::Memory_High) {
        mFixedSimpleStrategy = false;
    }
    if (mFixedSimpleStrategy) {
        mStrategy.unitType = ComputeStrategy::D2;
        mStrategy.transPhase = ComputeStrategy::Offline;
    }
    const auto convCommon  = convParam->common();
    const auto outputCount = convCommon->outputCount(), srcCount = convCommon->inputCount();
    const int unitI = 8, srcCountUnit = UP_DIV(srcCount, unitI), outputCountUnit = UP_DIV(outputCount, 4);
    
    // mWeightInt8 is used to store untransformed reordered weight
    mWeightInt8.reset(Tensor::createDevice<int8_t>({UP_DIV(outputCount, 4), UP_DIV(srcCount, unitI), 9, unitI * 4}));
    bool res = backend->onAcquireBuffer(mWeightInt8.get(), Backend::STATIC);
    if (!res) {
        return;
    }
    const int outputChannleUp4 = ALIGN_UP4(outputCount);
    mBiasFloat.reset(Tensor::createDevice<float>({outputChannleUp4}));
    res = backend->onAcquireBuffer(mBiasFloat.get(), Backend::STATIC);
    if (!res) {
        mValid = false;
        return;
    }
    mScaleFloat.reset(Tensor::createDevice<float>({outputChannleUp4}));
    res = backend->onAcquireBuffer(mScaleFloat.get(), Backend::STATIC);
    if (!res) {
        mValid = false;
        return;
    }
    auto biasPtr = mBiasFloat->host<int32_t>();
    memset(biasPtr, 0, outputChannleUp4 * sizeof(int32_t));
    auto scalePtr = mScaleFloat->host<float>();
    memset(scalePtr, 0, outputChannleUp4 * sizeof(float));
    const int8_t *weightSrc = nullptr;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    float inputScale = TensorUtils::getDescribe(inputs[0])->quantAttr ?
                       TensorUtils::getDescribe(inputs[0])->quantAttr->scale : 0.f;
    float outputScale = TensorUtils::getDescribe(outputs[0])->quantAttr ?
                       TensorUtils::getDescribe(outputs[0])->quantAttr->scale : 0.f;
    if (!ConvolutionCommon::getConvInt8Parameters(convParam, quanCommon, weightSrc, scalePtr, biasPtr, inputScale, outputScale)) {
        return;
    }

    auto weightDst = mWeightInt8->host<int8_t>();
    CPUConvolution::reorderWeightSlow<int8_t>(weightDst, weightSrc, srcCount, outputCount, 9, unitI, 4, true);
    // mWeight is used to store 2d-transformed weight
    mWeight.reset(Tensor::createDevice<int8_t>({BLOCK_UNIT2, outputCountUnit, srcCountUnit, unitI * 4}));
    if (mFixedSimpleStrategy) {
        auto code = tensorMemoryOnStrategyChange(nullptr, &mStrategy, inputs, outputs, nullptr);
        if (code != NO_ERROR) {
            mValid = false;
            return;
        }
    }

    mRelu    = convCommon->relu() || convCommon->relu6();
}
ConvInt83x3::~ConvInt83x3() {
    tensorMemoryOnStrategyChange(&mStrategy, nullptr, {}, {}, nullptr);
    backend()->onReleaseBuffer(mBiasFloat.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mScaleFloat.get(), Backend::STATIC);
}
ErrorCode ConvInt83x3::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    
    std::vector<Tensor*> dynamicAllocTensors;
    // update mStrategy when tensor size be changed.
    if (!mFixedSimpleStrategy) {
        auto strategy = getComputeStrategy(inputs, outputs);
        //MNN_PRINT("unitType: %d, transPhase: %d\n", strategy.unitType, strategy.transPhase);
        auto code = tensorMemoryOnStrategyChange((mStrategyCompleted ? &mStrategy : nullptr), &strategy,
                                                 inputs, outputs, &dynamicAllocTensors);
        if (code != NO_ERROR) {
            mValid = false;
            return code;
        }
        mStrategyCompleted = true;
        mStrategy = strategy;
    }
    
    const auto srcCount = inputs[0]->channel();
    const auto outputCountUnit = UP_DIV(outputs[0]->channel(), 4);
    auto threadNums = ((CPUBackend*)backend())->threadNumber();
    const bool combine1D2D = (mStrategy.unitType == ComputeStrategy::D2_D1);
    
    int unitI = 8, srcCountUnit = UP_DIV(srcCount, unitI);
    const auto height = inputs[0]->height(), width = inputs[0]->width(), batch = inputs[0]->batch();
    mTempInput.reset(Tensor::createDevice<int8_t>({batch, srcCountUnit, height, width, unitI}));
    mTempSrcBuffer.reset(Tensor::createDevice<int8_t>({threadNums, BLOCK_UNIT2, srcCountUnit, GEMM_TILE_UNIT * unitI}));
    mTempDstBuffer.reset(Tensor::createDevice<float>({threadNums, BLOCK_UNIT2, outputCountUnit, GEMM_TILE_UNIT * 4}));
    mTempTransformBuffer.reset(Tensor::createDevice<int32_t>({threadNums, 2, SRC_UNIT * SRC_UNIT, 4}));
    
    dynamicAllocTensors.push_back(mTempSrcBuffer.get());
    dynamicAllocTensors.push_back(mTempDstBuffer.get());
    dynamicAllocTensors.push_back(mTempTransformBuffer.get());
    dynamicAllocTensors.push_back(mTempInput.get());

    if (combine1D2D) {
        mTempOutBuffer.reset(Tensor::createDevice<int32_t>({threadNums, 2, DST_UNIT, outputCountUnit, GEMM_TILE_UNIT * 4}));
        dynamicAllocTensors.push_back(mTempOutBuffer.get());
    }
    // dynamic alloc tensor memory
    bool res = true;
    for (int i = 0; i < dynamicAllocTensors.size(); ++i) {
        res = res && backend()->onAcquireBuffer(dynamicAllocTensors[i], Backend::DYNAMIC);
    }
    if (!res) {
        return OUT_OF_MEMORY;
    }
    for (int i = 0; i < dynamicAllocTensors.size(); ++i) {
        backend()->onReleaseBuffer(dynamicAllocTensors[i], Backend::DYNAMIC);
    }

    return NO_ERROR;
}
ErrorCode ConvInt83x3::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    
    // Clip into [min, max]
    const int minVal = -(1<<(mActBits-1)), maxVal = (1<<(mActBits-1))-1, size = inputs[0]->elementSize();
    const int threadNumber = ((CPUBackend*)backend())->threadNumber();
    auto data = input->host<int8_t>();
    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        int step = UP_DIV(size, 8 * threadNumber), start = (int)tId * step, num = ALIMIN(start + step, size) - start;
        if (num > 0) {
            MNNInt8ClipInplace(data + start, num, minVal, maxVal);
        }
    } MNN_CONCURRENCY_END();
    
    const int ow = output->width(), oh = output->height();
    const int iw = input->width(), ih = input->height();
    const int dc_4 = UP_DIV(output->channel(), 4);
    const int padX = mPadX, padY = mPadY;
    
    const bool combine1D2D = (mStrategy.unitType == ComputeStrategy::D2_D1);
    const bool offline = (mStrategy.transPhase == ComputeStrategy::Offline);
    const bool online = (mStrategy.transPhase == ComputeStrategy::Online);
    
    const int wUnit = (combine1D2D ? ow / DST_UNIT: UP_DIV(ow, DST_UNIT));
    const int hUnit = (combine1D2D ? oh / DST_UNIT: UP_DIV(oh, DST_UNIT));

    const int unitI = 8, ic_unit = UP_DIV(input->channel(), unitI), weightOffset = mWeight->stride(0);
    // weight transform if needed.
    bool trans2d = online, trans1d = (combine1D2D && !offline);
    if (trans2d || trans1d) {
        weightContent(trans2d, trans1d);
    }
    // C4 to C16 for int8 multype
    for (int b = 0; b < input->batch(); ++b) {
        auto src = input->host<int8_t>() + b * input->stride(0);
        auto dst = mTempInput->host<int8_t>() + b * mTempInput->stride(0);
        const int ic8 = UP_DIV(input->channel(), 8), ic4 = UP_DIV(input->channel(), 4);
        // C4 to C8
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            int step = UP_DIV(ic8, threadNumber) * 2, start = (int)tId * step, num = ALIMIN(start + step, ic4) - start;
            if (num > 0) {
                MNNInt8C4ToC8(dst + start * iw * ih * 8, src + start * iw * ih * 8, iw * ih, num);
            }
        } MNN_CONCURRENCY_END();
    }

    auto sourceTransform2DFunc = [=](int xIndex, int xC, const int8_t* srcOrigin, int8_t* srcBlockInt8, int8_t* dstOrigin) {
        for (int xi = 0; xi < xC; ++xi) {
            auto index   = xIndex + xi;
            auto dstUnit = dstOrigin + unitI * xi;

            int wIndex = index % wUnit, hIndex = index / wUnit;
            int srcX = wIndex * DST_UNIT - padX, srcY = hIndex * DST_UNIT - padY;
            int sy = ALIMAX(0, srcY) - srcY, ey = ALIMIN(srcY + SRC_UNIT, ih) - srcY;
            int sx = ALIMAX(0, srcX) - srcX, ex = ALIMIN(srcX + SRC_UNIT, iw) - srcX;
            int xL = ex - sx;
            auto srcStart = srcOrigin + (srcX + srcY * iw) * unitI;
            
            for (int z = 0; z < ic_unit; ++z) {
                ::memset(srcBlockInt8, 0, unitI * SRC_UNIT * SRC_UNIT * sizeof(int8_t));
                auto dstStart = dstUnit + z * unitI * xC;
                auto src_z = srcStart + z * unitI * iw * ih;
                // Extract One Block
                if (xL > 0) {
                    for (int yy = sy; yy < ey; ++yy) {
                        auto dst_yy = srcBlockInt8 + yy * unitI * SRC_UNIT;
                        auto src_yy = src_z + unitI * iw * yy;
                        ::memcpy(dst_yy + sx * unitI, src_yy + sx * unitI , xL * unitI * sizeof(int8_t));
                    }
                }
                // Source Transform
                sourceTransformUnit2D<int8_t, 8>(srcBlockInt8, dstStart, unitI, unitI * xC * ic_unit, 1);
            }
        }
    };
    
    // input tensor right and bottom leftover points
    auto sourceTransform1DFunc = [=](int xIndex, int xC, const int8_t* srcOrigin, int8_t* srcBlockInt8,
                                     int8_t* dstOrigin, bool hDirection) -> int {
        const int wRemain = ow % DST_UNIT, hBlock = oh / DST_UNIT, wBlock = ow / DST_UNIT, kernel = 3;
        // summary used kernel tile (1x3 or 3x1). for example, right and bottom riegon in F(2,3) is 2 instead of 3
        int usableKernel = 0;
        for (int xi = 0; xi < xC; ++xi) {
            auto index = xIndex + xi;
            if (hDirection) { // H direction
                int srcX = wBlock * DST_UNIT + index % wRemain - padX;
                usableKernel = ALIMAX(ALIMIN(srcX + kernel, iw) - srcX, usableKernel);
            } else { // W direction
                int srcY = hBlock * DST_UNIT + index / wBlock - padY;
                usableKernel = ALIMAX(ALIMIN(srcY + kernel, ih) - srcY, usableKernel);
            }
        }
        // do source 1d-winograd transform
        for (int xi = 0; xi < xC; ++xi) {
            auto dstUnit = dstOrigin + unitI * xi;
            auto index   = xIndex + xi;
            
            int srcX, srcY, unitX, unitY;
            if (hDirection) { // H direction
                srcX  = wBlock * DST_UNIT + index % wRemain - padX;
                srcY  = index / wRemain * DST_UNIT - padY;
                unitX = kernel;
                unitY = SRC_UNIT;
            } else { // W direction
                srcX = index % wBlock * DST_UNIT - padX;
                srcY = hBlock * DST_UNIT + index / wBlock - padY;
                unitX = SRC_UNIT;
                unitY = kernel;
            }
            int sx = ALIMAX(0, srcX) - srcX, ex = ALIMIN(srcX + unitX, iw) - srcX;
            int sy = ALIMAX(0, srcY) - srcY, ey = ALIMIN(srcY + unitY, ih) - srcY;
            int xL = ex - sx;

            auto srcStart = srcOrigin + (srcX + srcY * iw) * unitI;
            
            for (int z = 0; z < ic_unit; ++z) {
                ::memset(srcBlockInt8, 0, unitI * SRC_UNIT * kernel * sizeof(int8_t));
                auto dstStart = dstUnit + z * unitI * xC;
                auto src_z = srcStart + z * unitI * iw * ih;
                // Extract One Block
                if (xL > 0) {
                    for (int yy = sy; yy < ey; ++yy) {
                        auto dst_yy = srcBlockInt8 + yy * unitI * unitX;
                        auto src_yy = src_z + unitI * iw * yy;
                        ::memcpy(dst_yy + sx * unitI, src_yy + sx * unitI , xL * unitI * sizeof(int8_t));
                    }
                }
                // Source Transform
                for (int k = 0; k < usableKernel; ++k) {
                    int8_t* dst = dstStart + k * unitI * xC * ic_unit * SRC_UNIT;
                    if (hDirection) {
                        sourceTransformUnit1D<int8_t, 8>(srcBlockInt8 + unitI * k, dst, kernel * unitI, unitI * xC * ic_unit, 1);
                    } else {
                        sourceTransformUnit1D<int8_t, 8>(srcBlockInt8 + unitI * SRC_UNIT * k, dst, unitI, unitI * xC * ic_unit, 1);
                    }
                }
            }
        }
        return usableKernel;
    };
    
    auto addBiasAndQuantize = [=](const float* srcOrigin, const float* bias, const float* scale, float* tmpBuffer, int8_t* dstOrigin,
                                  size_t srcStep, size_t count, size_t numBit, bool uint) {
#ifdef MNN_USE_NEON
        auto biasV = vld1q_f32(bias);
#endif
        for (int j=0; j < count; ++j) {
            auto src = srcOrigin + srcStep * j;
            auto dst = tmpBuffer + 4 * j;
#ifdef MNN_USE_NEON
            vst1q_f32(dst, vld1q_f32(src) + biasV);
#else
            for (int k=0; k<4; ++k) {
                dst[k] = src[k] + bias[k];
            }
#endif
        }
        ssize_t minValue, maxValue;
        if (uint) {
            minValue = 0;
            maxValue = (1 << numBit) - 1;
        } else {
            minValue = -(1 << (numBit - 1));
            maxValue = (1 << (numBit - 1)) - 1;
        }
        MNNFloat2Int8(tmpBuffer, dstOrigin, count, scale, minValue, maxValue, 0);
    };
    
    auto destTransform2DFunc =
        [=, &addBiasAndQuantize](int xIndex, int xC, const float* srcOrigin, const float* bias, const float* scale,
                                 float* dstBlock, int8_t* midBlock, int8_t* dstOrigin) {
        // Dest Transform
        for (int xi = 0; xi < xC; ++xi) {
            auto index   = xIndex + xi;
            auto srcUnit = srcOrigin + 4 * xi;
            
            int wIndex = index % wUnit, hIndex = index / wUnit;
            int dstX = wIndex * DST_UNIT, dstY = hIndex * DST_UNIT;
            int dstValidX = ALIMIN(ow - dstX, DST_UNIT), dstValidY = ALIMIN(oh - dstY, DST_UNIT);

            auto dstStart = dstOrigin + 4 * (dstX + dstY * ow);
            for (int z = 0; z < dc_4; ++z) {
                auto srcZ = srcUnit + z * xC * 4;
                auto dstZ = dstStart + z * ow * oh * 4;
                destTransform2D<WinogradHelper::FractionsInA>(srcZ, dstBlock, dc_4 * 4 * xC, 4, 1);
                addBiasAndQuantize(dstBlock, bias + 4 * z, scale + 4 * z, dstBlock, midBlock, 4, DST_UNIT * DST_UNIT, 8, false);
                for (int j = 0; j < dstValidY; ++j) {
                    ::memcpy(dstZ + ow * 4 * j, midBlock + (DST_UNIT * j) * 4, 4 * dstValidX * sizeof(int8_t));
                }
            }
        }
    };
    
    auto destTransform1DFunc = [=](int xC, const float* srcOrigin, float* dstOrigin) {
        // Dest Transform
        for (int xi = 0; xi < xC; ++xi) {
            auto srcUnit = srcOrigin + 4 * xi;
            auto dstStart = dstOrigin + 4 * xi;
            for (int z = 0; z < dc_4; ++z) {
                auto srcZ = srcUnit + z * xC * 4;
                auto dstZ = dstStart + z * xC * 4;
                destTransform1D<WinogradHelper::FractionsInA>(srcZ, dstZ, dc_4 * 4 * xC, dc_4 * 4 * xC, 1);
            }
        }
    };
    
    auto outAddBiasQuantizeStore =
        [=, &addBiasAndQuantize](int xIndex, int xC, const float* bias, const float* scale, const float* srcOrigin,
                                 int8_t* midBlock, int8_t* dstOrigin, float* tempBuffer, bool hDirection) {
        const int wRemain = ow % DST_UNIT, hBlock = oh / DST_UNIT, wBlock = ow / DST_UNIT;
        for (int xi = 0; xi < xC; ++xi) {
            int index = xIndex + xi;
            int dstX, dstY;
            if (hDirection) {
                dstX = wBlock * DST_UNIT + index % wRemain;
                dstY = index / wRemain * DST_UNIT;
            } else {
                dstX = index % wBlock * DST_UNIT;
                dstY = hBlock * DST_UNIT + index / wBlock;
            }
            auto src_i = srcOrigin + xi * 4;
            auto dst_i = dstOrigin + (dstY * ow + dstX) * 4;
            for (int z = 0; z < dc_4; ++z) {
                auto src_z = src_i + z * xC * 4;
                auto dst_z = dst_i + z * oh * ow * 4;
                addBiasAndQuantize(src_z, bias + 4 * z, scale + 4 * z, tempBuffer, midBlock, dc_4 * xC * 4, DST_UNIT, 8, false);
                if (hDirection) {
                    for (int i = 0; i < DST_UNIT; ++i) {
                        ::memcpy(dst_z + i * ow * 4, midBlock + i * 4, sizeof(int8_t) * 4);
                    }
                } else {
                    ::memcpy(dst_z, midBlock, DST_UNIT * sizeof(int8_t) * 4);
                }
            }
        }
    };
    
    auto gemmFunc = [=](int xC, int start, int end, const int8_t* srcOrigin, const int8_t* weight, float* dstOrigin) {
        if (xC == GEMM_TILE_UNIT) {
            for (int i = start; i < end; ++i) {
                MNNGemmInt8toFloat32_8x4_Unit(dstOrigin + i * dc_4 * 4 * xC, srcOrigin + i * ic_unit * unitI * xC,
                                              weight + i * weightOffset, ic_unit, xC * 4, dc_4);
            }
        } else {
            for (int i = start; i < end; ++i) {
                MNNGemmInt8toFloat32_8x4_Common(dstOrigin + i * dc_4 * 4 * xC, srcOrigin + i * ic_unit * unitI * xC,
                                                weight + i * weightOffset, ic_unit, xC, xC * 4, dc_4);
            }
        }
    };
    
    auto gemmConcurrencyFunc = [=, &gemmFunc](int xC, int gemmNum, const int8_t* srcOrigin, const int8_t* weight, float* dstOrigin) {
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            const int step = UP_DIV(gemmNum, threadNumber);
            gemmFunc(xC, (int)tId * step, ALIMIN((tId + 1) * step, gemmNum), srcOrigin, weight, dstOrigin);
        }
        MNN_CONCURRENCY_END()
    };
    
    auto tFunction2D = [&](int tId, int tileStart, int tileStep, int tileEnd, int totalCount,
                           const int8_t* srcOrigin, int8_t* dstOrigin) {
        //MNN_PRINT("tId: %d, tileStart: %d, tileStep: %d, tileEnd: %d, totalCount: %d\n",
        //          tId, tileStart, tileStep, tileEnd, totalCount);
        auto srcBlock = (int8_t*)(mTempTransformBuffer->host<int32_t>() + mTempTransformBuffer->stride(0) * tId);
        auto midBlock = (float*)(srcBlock) + mTempTransformBuffer->stride(1);
        auto _srcOrigin = mTempSrcBuffer->host<int8_t>() + mTempSrcBuffer->stride(0) * tId;
        auto _dstOrigin = mTempDstBuffer->host<float>() + mTempDstBuffer->stride(0) * tId;
        
        for (int tIndex = tileStart; tIndex < tileEnd; tIndex += tileStep) {
            int xIndex  = (int)tIndex * GEMM_TILE_UNIT;
            int xReamin = totalCount - xIndex;
            int xC      = xReamin > GEMM_TILE_UNIT ? GEMM_TILE_UNIT : xReamin;
            
            sourceTransform2DFunc(xIndex, xC, srcOrigin, srcBlock, _srcOrigin);
            if (threadNumber != tileStep) {
                gemmConcurrencyFunc(xC, BLOCK_UNIT2, _srcOrigin, mWeight->host<int8_t>(), _dstOrigin);
            } else {
                gemmFunc(xC, 0, BLOCK_UNIT2, _srcOrigin, mWeight->host<int8_t>(), _dstOrigin);
            }
            destTransform2DFunc(xIndex, xC, _dstOrigin, mBiasFloat->host<float>(), mScaleFloat->host<float>(),
                                midBlock, srcBlock, dstOrigin);
        }
    };
    
    auto tFunction1D = [&](int tId, int tileStart, int tileStep, int tileEnd, int totalCount,
                           const int8_t* srcOrigin, int8_t* dstOrigin, bool hDirection) {
        auto srcBlock = (int8_t*)(mTempTransformBuffer->host<int32_t>() + mTempTransformBuffer->stride(0) * tId);
        auto midBlock = (int8_t*)(((int32_t*)srcBlock) + mTempTransformBuffer->stride(1));
        auto srcBlockInt8 = midBlock;
        auto _srcOrigin = mTempSrcBuffer->host<int8_t>() + mTempSrcBuffer->stride(0) * tId;
        auto _dstOriginTemp = mTempDstBuffer->host<float>() + mTempDstBuffer->stride(0) * tId;
        float* _dstOrigin[2] = {mTempOutBuffer->host<float>() + mTempOutBuffer->stride(0) * tId, nullptr};
        _dstOrigin[1] = _dstOrigin[0] + mTempOutBuffer->stride(1);
        
        const int8_t* weightOrigin = mWeightExtra->host<int8_t>();
        if (!hDirection) {
            weightOrigin = weightOrigin + mWeightExtra->stride(0);
        }
        for (int tIndex = tileStart; tIndex < tileEnd; tIndex += tileStep) {
            int xIndex  = (int)tIndex * GEMM_TILE_UNIT;
            int xReamin = totalCount - xIndex;
            int xC      = xReamin > GEMM_TILE_UNIT ? GEMM_TILE_UNIT : xReamin;
            int stride = DST_UNIT * xC * 4;
            
            int usableKernel = sourceTransform1DFunc(xIndex, xC, srcOrigin, srcBlockInt8, _srcOrigin, hDirection);
            for (int kIndex = 0; kIndex < usableKernel; ++kIndex) {
                const int8_t* weight = weightOrigin + kIndex * BLOCK_UNIT * mWeightExtra->stride(1);
                const int8_t* src    = _srcOrigin + kIndex * BLOCK_UNIT * ic_unit * xC * unitI;
                if (threadNumber != tileStep) {
                    gemmConcurrencyFunc(xC, BLOCK_UNIT, src, weight, _dstOriginTemp);
                } else {
                    gemmFunc(xC, 0, BLOCK_UNIT, src, weight, _dstOriginTemp);
                }
                if (kIndex == 0) {
                    destTransform1DFunc(xC, _dstOriginTemp, _dstOrigin[0]);
                } else {
                    destTransform1DFunc(xC, _dstOriginTemp, _dstOrigin[1]);
                    MNNMatrixAdd(_dstOrigin[0], _dstOrigin[0], _dstOrigin[1], xC * DST_UNIT, stride, stride, stride, dc_4);
                }
            }
            outAddBiasQuantizeStore(xIndex, xC, mBiasFloat->host<float>(), mScaleFloat->host<float>(),
                                    _dstOrigin[0], midBlock, dstOrigin, (float*)srcBlock, hDirection);
        }
    };
    
    auto tFunctionLeftOverGemm = [&](const int8_t* srcOrigin, int8_t* dstOrigin) {
        auto weightSrc = mWeightInt8->host<int8_t>();
        auto weight = mWeightLeftOver->host<int8_t>();
        auto src = mTempSrcBuffer->host<int8_t>();
        auto dst = mTempDstBuffer->host<float>();
        auto tempBuffer = (float*)(dst + dc_4 * 4);
        auto dstInt8 = (int8_t*)(dst + 2 * dc_4 * 4);
        const int wRemain = ow % DST_UNIT, hRemain = oh % DST_UNIT, kernel = 3;
        const int hOffset = oh / DST_UNIT * DST_UNIT, wOffset = ow / DST_UNIT * DST_UNIT;
        for (int xi = 0; xi < wRemain * hRemain; ++xi) {
            int wIndex = wOffset + xi % wRemain, hIndex = hOffset + xi / wRemain;
            int total = 0, cur = 0;
            for (int k = 0; k < kernel * kernel; ++k) {
                int srcX = wIndex + (k % kernel) - padX, srcY = hIndex + (k / kernel) - padY;
                if (srcX >= 0 && srcX < iw && srcY >= 0 && srcY < ih) {
                    ++total;
                }
            }
            for (int k = 0; k < kernel * kernel; ++k) {
                int srcX = wIndex + (k % kernel) - padX, srcY = hIndex + (k / kernel) - padY;
                if (srcX < 0 || srcX >= iw || srcY < 0 || srcY >= ih) {
                    continue;
                }
                for (int zo = 0; zo < dc_4; ++zo) {
                    for (int zi = 0; zi < ic_unit; ++zi) {
                        auto weightSrc_ = weightSrc + ((zo * ic_unit + zi) * kernel * kernel + k) * unitI * 4;
                        auto weight_ = weight + ((zo * total + cur) * ic_unit + zi) * unitI * 4;
                        ::memcpy(weight_, weightSrc_, unitI * 4 * sizeof(int8_t));
                    }
                }
                for (int zi = 0; zi < ic_unit; ++zi) {
                    auto srcOrigin_ = srcOrigin + ((zi * ih + srcY) * iw + srcX) * unitI;
                    auto src_ = src + (cur * ic_unit + zi) * unitI;
                    ::memcpy(src_, srcOrigin_, unitI * sizeof(int8_t));
                }
                ++cur;
            }
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                const int ocStep = UP_DIV(dc_4, threadNumber);
                const int ocStart = ALIMIN(tId * ocStep, dc_4), ocEnd = ALIMIN(ocStart + ocStep, dc_4);
                if (ocStart < ocEnd) {
                    MNNGemmInt8toFloat32_8x4_Common(dst + ocStart * 4, src, weight + ocStart * ic_unit * total * unitI * 4,
                                                    ic_unit * total, 1, 4, ocEnd - ocStart);
                }
            }
            MNN_CONCURRENCY_END()
            addBiasAndQuantize(dst, mBiasFloat->host<float>(), mScaleFloat->host<float>(), tempBuffer, dstInt8, 4, dc_4, 8, false);
            auto dstOrigin_ = dstOrigin + (hIndex * ow + wIndex) * 4;
            for (int z = 0; z < dc_4; ++z) {
                ::memcpy(dstOrigin_ + z * oh * ow * 4, dstInt8 + z * 4, 4 * sizeof(int8_t));
            }
        }
    };
    
    int totalCount, tileCount;
    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        auto srcOrigin = mTempInput->host<int8_t>() + batchIndex * mTempInput->stride(0);
        auto dstOrigin = output->host<int8_t>() + batchIndex * output->stride(0);
        // MNN_PRINT("%d, %d, %d, %d\n", wUnit, hUnit, layer->aMin, layer->aMax);

        // 2D tile
        totalCount = hUnit * wUnit;
        tileCount = UP_DIV(totalCount, GEMM_TILE_UNIT);
        if (tileCount >= threadNumber) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                tFunction2D((int)tId, (int)tId, threadNumber, tileCount / threadNumber * threadNumber, totalCount, srcOrigin, dstOrigin);
            }
            MNN_CONCURRENCY_END();
        }
        if (tileCount % threadNumber != 0) {
            tFunction2D(0, tileCount / threadNumber * threadNumber, 1, tileCount, totalCount, srcOrigin, dstOrigin);
        }
        
        if (!combine1D2D) {
            continue;
        }
        
        // 1D tile (H direction)
        totalCount = (ow % DST_UNIT) * (oh / DST_UNIT);
        tileCount = UP_DIV(totalCount, GEMM_TILE_UNIT);
        if (tileCount >= threadNumber) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                tFunction1D((int)tId, (int)tId, threadNumber, tileCount / threadNumber * threadNumber, totalCount, srcOrigin, dstOrigin, true);
            }
            MNN_CONCURRENCY_END();
        }
        if (tileCount % threadNumber != 0) {
            tFunction1D(0, tileCount / threadNumber * threadNumber, 1, tileCount, totalCount, srcOrigin, dstOrigin, true);
        }
        
        // 1D tile (W direction)
        totalCount = (oh % DST_UNIT) * (ow / DST_UNIT);
        tileCount = UP_DIV(totalCount, GEMM_TILE_UNIT);
        if (tileCount >= threadNumber) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                tFunction1D((int)tId, (int)tId, threadNumber, tileCount / threadNumber * threadNumber, totalCount, srcOrigin, dstOrigin, false);
            }
            MNN_CONCURRENCY_END();
        }
        if (tileCount % threadNumber != 0) {
            tFunction1D(0, tileCount / threadNumber * threadNumber, 1, tileCount, totalCount, srcOrigin, dstOrigin, false);
        }
        
        // leftover gemm
        tFunctionLeftOverGemm(srcOrigin, dstOrigin);
    }
    if (mRelu) {
        const int dstZStep = ow * oh * 4;
        for (int batchIndex = 0; batchIndex < output->batch(); ++batchIndex) {
            auto dstOrigin = output->host<int8_t>() + batchIndex * output->stride(0);
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                for (int z = (int)tId; z < dc_4; z += threadNumber) {
                    MNNReluInt8(dstOrigin + z * dstZStep, dstOrigin + z * dstZStep, dstZStep);
                }
            }
            MNN_CONCURRENCY_END();
        }
    }
    return NO_ERROR;
}

} /* MNN */
