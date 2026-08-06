#include "HexagonLoop.hpp"

#include <cstring>

#include "HexagonBackend.hpp"
#include "HexagonBinary.hpp"
#include "HexagonRaster.hpp"
#include "HexagonRuntime.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "htp_command.h"
#include "dsprpc_interface.h"

namespace MNN {

typedef struct {
  int32_t loopNumber;
  int32_t sizeZYX[3];
  int32_t dstStrideZYX[3];
  int32_t src0StrideZYX[3];
  int32_t src1StrideZYX[3];
  int32_t cmdSteps[3];
  int32_t cmdViewOffset[3];
  int64_t outputElementSize;
  int64_t input0Size;
  int64_t input1Size;
} __attribute__((packed)) HtpOpsLoopParam;

static void _copyZYXStrideBytes(const int zyxStride[3], int bytes, int zyxStrideBytes[3]) {
    zyxStrideBytes[0] = zyxStride[0] * bytes;
    zyxStrideBytes[1] = zyxStride[1] * bytes;
    zyxStrideBytes[2] = zyxStride[2] * bytes;
}

static bool _mapBinaryOp(int mnnOpType, int* dspOpType) {
    switch (mnnOpType) {
        case BinaryOpOperation_ADD:
            *dspOpType = 1;
            return true;
        case BinaryOpOperation_SUB:
            *dspOpType = 2;
            return true;
        case BinaryOpOperation_MUL:
            *dspOpType = 3;
            return true;
        case BinaryOpOperation_DIV:
        case BinaryOpOperation_REALDIV:
            *dspOpType = 4;
            return true;
        case BinaryOpOperation_MAXIMUM:
            *dspOpType = 5;
            return true;
        case BinaryOpOperation_MINIMUM:
            *dspOpType = 6;
            return true;
        case BinaryOpOperation_MUL_SILU:
            *dspOpType = 7;
            return true;
        case BinaryOpOperation_SquaredDifference:
            *dspOpType = 11;
            return true;
        default:
            return false;
    }
}

HexagonLoop::HexagonLoop(Backend* backend, const LoopParam* loop) : HexagonExecution(backend), mLoop(loop) {
    mAllocator = static_cast<HexagonBackend*>(backend)->getAllocator(0);
    auto runtime = static_cast<const HexagonRuntime*>(backend->getRuntime());
    mPack = runtime->info().vectorSize;
    if (mPack <= 0) {
        mPack = 4;
    }
    mStack.resize(loop->tensorNumber());
}

HexagonLoop::~HexagonLoop() {
    if (mZeroChunk.first != nullptr) {
        mAllocator->free(mZeroChunk);
    }

    if (mInitZeroParamChunk.first != nullptr) {
        mAllocator->free(mInitZeroParamChunk);
    }
    if (mInitRegionChunk.first != nullptr) {
        mAllocator->free(mInitRegionChunk);
    }
}

ErrorCode HexagonLoop::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  std::vector<HexagonCommand>& dst) {
    mLoopNumber = mLoop->loopNumber();
    mBytes = HexagonBackend::getBytes(outputs[0]);
    if (mBytes != 1 && mBytes != 2 && mBytes != 4) {
        MNN_PRINT("HexagonLoop Error at line 113: NOT_SUPPORT\n");
        return NOT_SUPPORT;
    }
    if (mLoop == nullptr || mLoop->commands() == nullptr || mLoop->commands()->size() != 1) {
        MNN_PRINT("HexagonLoop Error at line 116: NOT_SUPPORT\n");
        return NOT_SUPPORT;
    }

    auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
    if (cmd == nullptr || cmd->op() == nullptr || cmd->fuse() >= 0) {
        MNN_PRINT("HexagonLoop Error at line 121: NOT_SUPPORT\n");
        return NOT_SUPPORT;
    }

    auto inputIndexes = mLoop->inputIndexes();
    if (!inputs.empty()) {
        if (inputIndexes == nullptr || inputIndexes->size() != inputs.size()) {
            MNN_PRINT("HexagonLoop Error at line 127: INPUT_DATA_ERROR\n");
            return INPUT_DATA_ERROR;
        }
        for (int i = 0; i < inputs.size(); ++i) {
            mStack[inputIndexes->data()[i]] = inputs[i];
        }
    }
    auto outputIndexes = mLoop->outputIndexes();
    if (outputIndexes == nullptr || outputIndexes->size() != outputs.size()) {
        MNN_PRINT("HexagonLoop Error at line 135: INPUT_DATA_ERROR\n");
        return INPUT_DATA_ERROR;
    }
    for (int i = 0; i < outputs.size(); ++i) {
        mStack[outputIndexes->data()[i]] = outputs[i];
    }

    for (int i = 0; i < 3; ++i) {
        mCmdSizeZYX[i] = cmd->size()->data()[i];
    }

    // clear cached resources
    if (mZeroChunk.first != nullptr) {
        mAllocator->free(mZeroChunk);
        mZeroChunk = MemChunk();
    }

    if (mInitZeroParamChunk.first != nullptr) {
        mAllocator->free(mInitZeroParamChunk);
        mInitZeroParamChunk = MemChunk();
    }
    if (mInitRegionChunk.first != nullptr) {
        mAllocator->free(mInitRegionChunk);
        mInitRegionChunk = MemChunk();
    }

    // clear commands
    mInitZeroCmds.clear();
    mInitCopyCmds.clear();

    // shared 0
    mZeroChunk = mAllocator->alloc((size_t)mBytes);
    if (mZeroChunk.first == nullptr) {
        MNN_PRINT("HexagonLoop Error at line 166: OUT_OF_MEMORY\n");
        return OUT_OF_MEMORY;
    }
    ::memset(HexagonBackend::getPtr(mZeroChunk), 0, (size_t)mBytes);
    static_cast<HexagonBackend*>(backend())->markHostInput(mZeroChunk, mBytes);



    // initCommand
    mInitZeroTensorIndexes.clear();
    mInitCopyCommands.clear();

    // command
    auto op = cmd->op();
    if (op->type() == OpType_UnaryOp && op->main() == nullptr) {
        if (cmd->indexes() == nullptr || cmd->indexes()->size() != 2) {
            MNN_PRINT("HexagonLoop Error at line 232: NOT_SUPPORT\n");
            return NOT_SUPPORT;
        }
        if (cmd->view() == nullptr || cmd->view()->size() != 2) {
            MNN_PRINT("HexagonLoop Error at line 235: NOT_SUPPORT\n");
            return NOT_SUPPORT;
        }
        if (cmd->steps() == nullptr || cmd->steps()->size() != 2) {
            MNN_PRINT("HexagonLoop Error at line 238: NOT_SUPPORT\n");
            return NOT_SUPPORT;
        }
        if (cmd->iterIndexes() == nullptr || cmd->iterIndexes()->size() != 2) {
            MNN_PRINT("HexagonLoop Error at line 241: NOT_SUPPORT\n");
            return NOT_SUPPORT;
        }
        mCmdIndexes[0] = cmd->indexes()->data()[0];
        mCmdIndexes[1] = cmd->indexes()->data()[1];

        mCmdIterIndexes[0] = cmd->iterIndexes()->data()[0];
        mCmdIterIndexes[1] = cmd->iterIndexes()->data()[1];

        mCmdSteps[0] = cmd->steps()->data()[0];
        mCmdSteps[1] = cmd->steps()->data()[1];

        auto dstView = cmd->view()->GetAs<View>(0);
        auto srcView = cmd->view()->GetAs<View>(1);
        mCmdViewOffset[0] = dstView->offset();
        mCmdViewOffset[1] = srcView->offset();
        for (int d = 0; d < 3; ++d) {
            mCmdViewStride[0][d] = dstView->stride()->data()[d];
            mCmdViewStride[1][d] = srcView->stride()->data()[d];
        }
        struct MergedLoopParam {
            int32_t cmdKind;
            int32_t opType;
            int32_t bytes;
            HtpOpsLoopParam loopParam;
        } __attribute__((packed));

        if (mLoopNumber > 0) {
            MergedLoopParam params;
            params.cmdKind = 0;
            params.opType = 0;
            params.bytes = mBytes;

            auto& lp = params.loopParam;
            lp.loopNumber = mLoopNumber;
            ::memcpy(lp.sizeZYX, mCmdSizeZYX, sizeof(lp.sizeZYX));
            _copyZYXStrideBytes(mCmdViewStride[0], mBytes, lp.dstStrideZYX);
            _copyZYXStrideBytes(mCmdViewStride[1], mBytes, lp.src0StrideZYX);
            memset(lp.src1StrideZYX, 0, sizeof(lp.src1StrideZYX));
            lp.cmdSteps[0] = mCmdSteps[0];
            lp.cmdSteps[1] = mCmdSteps[1];
            lp.cmdSteps[2] = 0;
            lp.cmdViewOffset[0] = mCmdViewOffset[0];
            lp.cmdViewOffset[1] = mCmdViewOffset[1];
            lp.cmdViewOffset[2] = 0;
            lp.outputElementSize = HexagonBackend::getElementSize(mStack[mCmdIndexes[0]], mPack);
            lp.input0Size = HexagonBackend::getElementSize(mStack[mCmdIndexes[1]], mPack);
            lp.input1Size = 0;

            auto outputTensor = mStack[mCmdIndexes[0]];
            auto input0 = mStack[mCmdIndexes[1]];

            auto dstDev = HexagonBackend::getDevicePtr(outputTensor);
            auto src0Dev = HexagonBackend::getDevicePtr(input0);
            auto iter0Dev = mCmdIterIndexes[0] >= 0 ? HexagonBackend::getDevicePtr(mStack[mCmdIterIndexes[0]]) : std::make_pair(-1, 0);
            auto iter1Dev = mCmdIterIndexes[1] >= 0 ? HexagonBackend::getDevicePtr(mStack[mCmdIterIndexes[1]]) : std::make_pair(-1, 0);

            std::vector<std::pair<int, int>> inputFds = {src0Dev, {-1, 0}, iter0Dev, iter1Dev, {-1, 0}};
            std::vector<std::pair<int, int>> outputFds = {dstDev};

            dst.emplace_back();
            dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_LOOP_BLIT, &params, sizeof(params),
                             inputFds,  outputFds,  inputs, outputs);
        }
        return NO_ERROR;
    }

    if (op->type() == OpType_BinaryOp) {
        if (cmd->indexes() == nullptr || cmd->indexes()->size() != 3) {
            MNN_PRINT("HexagonLoop Error at line 315: NOT_SUPPORT\n");
            return NOT_SUPPORT;
        }
        if (cmd->view() == nullptr || cmd->view()->size() != 3) {
            MNN_PRINT("HexagonLoop Error at line 319: NOT_SUPPORT\n");
            return NOT_SUPPORT;
        }

        int dspOpType = 0;
        auto binary = op->main_as_BinaryOp();
        if (binary == nullptr || binary->activationType() != 0 || !_mapBinaryOp(binary->opType(), &dspOpType)) {
            MNN_PRINT("HexagonLoop Error at line 325: NOT_SUPPORT\n");
            return NOT_SUPPORT;
        }

        auto outputTensor = mStack[cmd->indexes()->data()[0]];
        auto input0 = mStack[cmd->indexes()->data()[1]];
        auto input1 = mStack[cmd->indexes()->data()[2]];
        if (outputTensor == nullptr || input0 == nullptr || input1 == nullptr) {
            return INPUT_DATA_ERROR;
        }

        auto dstView = cmd->view()->GetAs<View>(0);
        auto src0View = cmd->view()->GetAs<View>(1);
        auto src1View = cmd->view()->GetAs<View>(2);

        if (mLoopNumber > 1) {
            struct MergedLoopParam {
                int32_t cmdKind;
                int32_t opType;
                int32_t bytes;
                HtpOpsLoopParam loopParam;
            } __attribute__((packed));

            MergedLoopParam params;
            params.cmdKind = 1;
            params.opType = dspOpType;
            params.bytes = mBytes;

            auto& lp = params.loopParam;
            lp.loopNumber = mLoopNumber;
            ::memcpy(lp.sizeZYX, mCmdSizeZYX, sizeof(lp.sizeZYX));
            _copyZYXStrideBytes(dstView->stride()->data(), mBytes, lp.dstStrideZYX);
            _copyZYXStrideBytes(src0View->stride()->data(), mBytes, lp.src0StrideZYX);
            _copyZYXStrideBytes(src1View->stride()->data(), mBytes, lp.src1StrideZYX);
            lp.cmdSteps[0] = cmd->steps() && cmd->steps()->size() > 0 ? cmd->steps()->data()[0] : 0;
            lp.cmdSteps[1] = cmd->steps() && cmd->steps()->size() > 1 ? cmd->steps()->data()[1] : 0;
            lp.cmdSteps[2] = cmd->steps() && cmd->steps()->size() > 2 ? cmd->steps()->data()[2] : 0;
            lp.cmdViewOffset[0] = dstView->offset();
            lp.cmdViewOffset[1] = src0View->offset();
            lp.cmdViewOffset[2] = src1View->offset();
            lp.outputElementSize = HexagonBackend::getElementSize(outputTensor, mPack);
            lp.input0Size = HexagonBackend::getElementSize(input0, mPack);
            lp.input1Size = HexagonBackend::getElementSize(input1, mPack);

            auto dstDev = HexagonBackend::getDevicePtr(outputTensor);
            auto src0Dev = HexagonBackend::getDevicePtr(input0);
            auto src1Dev = HexagonBackend::getDevicePtr(input1);
            auto iter0Dev = (cmd->iterIndexes() && cmd->iterIndexes()->size() > 0 && cmd->iterIndexes()->data()[0] >= 0)
                                ? HexagonBackend::getDevicePtr(mStack[cmd->iterIndexes()->data()[0]])
                                : std::make_pair(-1, 0);
            auto iter1Dev = (cmd->iterIndexes() && cmd->iterIndexes()->size() > 1 && cmd->iterIndexes()->data()[1] >= 0)
                                ? HexagonBackend::getDevicePtr(mStack[cmd->iterIndexes()->data()[1]])
                                : std::make_pair(-1, 0);
            auto iter2Dev = (cmd->iterIndexes() && cmd->iterIndexes()->size() > 2 && cmd->iterIndexes()->data()[2] >= 0)
                                ? HexagonBackend::getDevicePtr(mStack[cmd->iterIndexes()->data()[2]])
                                : std::make_pair(-1, 0);

            std::vector<std::pair<int, int>> inputFds = {src0Dev, src1Dev, iter0Dev, iter1Dev, iter2Dev};
            std::vector<std::pair<int, int>> outputFds = {dstDev};
            dst.emplace_back();
            dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_LOOP_BLIT, &params, sizeof(params),
                             inputFds, outputFds, inputs, outputs);
            return NO_ERROR;
        }

        struct MergedBinaryRegionParam {
            int32_t regionCount;
            int32_t bytes;
            int32_t opType;
            HexagonBinary::BinaryRegion region;
        } __attribute__((packed));

        MergedBinaryRegionParam params;
        params.regionCount = 1;
        params.bytes = mBytes;
        params.opType = dspOpType;
        params.region.src0Offset = src0View->offset() * mBytes;
        params.region.src1Offset = src1View->offset() * mBytes;
        params.region.dstOffset = dstView->offset() * mBytes;
        for (int d = 0; d < 3; ++d) {
            params.region.size[d] = mCmdSizeZYX[d];
            params.region.src0Stride[d] = src0View->stride()->data()[d] * mBytes;
            params.region.src1Stride[d] = src1View->stride()->data()[d] * mBytes;
            params.region.dstStride[d] = dstView->stride()->data()[d] * mBytes;
        }
        auto dstDev = HexagonBackend::getDevicePtr(outputTensor);
        auto src0Dev = HexagonBackend::getDevicePtr(input0);
        auto src1Dev = HexagonBackend::getDevicePtr(input1);
        std::vector<std::pair<int, int>> inputFds = {src0Dev, src1Dev};
        std::vector<std::pair<int, int>> outputFds = {dstDev};

        dst.emplace_back();
        dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_BINARY_BLIT, &params, sizeof(params),
                         inputFds,  outputFds,  inputs, outputs);
        return NO_ERROR;
    }

    if (op->type() == OpType_MatMul) {
        if (cmd->indexes() == nullptr || cmd->indexes()->size() != 3) {
            MNN_PRINT("HexagonLoop Error at line 331: NOT_SUPPORT\n");
            return NOT_SUPPORT;
        }
        if (cmd->view() == nullptr || cmd->view()->size() != 3) {
            MNN_PRINT("HexagonLoop Error at line 335: NOT_SUPPORT\n");
            return NOT_SUPPORT;
        }
        auto outputTensor = mStack[cmd->indexes()->data()[0]];
        auto input0 = mStack[cmd->indexes()->data()[1]];
        auto input1 = mStack[cmd->indexes()->data()[2]];
        if (outputTensor == nullptr || input0 == nullptr || input1 == nullptr) {
            return INPUT_DATA_ERROR;
        }
        if (mBytes != 2) {
            MNN_PRINT("HexagonLoop Error at line 345: NOT_SUPPORT\n");
            return NOT_SUPPORT;
        }

        auto dstView = cmd->view()->GetAs<View>(0);
        auto src0View = cmd->view()->GetAs<View>(1);
        auto src1View = cmd->view()->GetAs<View>(2);

        struct BatchMatmulParam {
            int32_t bytes;
            HtpOpsLoopParam loopParam;
        } __attribute__((packed));

        BatchMatmulParam params;
        params.bytes = mBytes;

        auto& lp = params.loopParam;
        lp.loopNumber = mLoopNumber;
        for (int d = 0; d < 3; ++d) {
            lp.sizeZYX[d] = mCmdSizeZYX[d];
            lp.dstStrideZYX[d] = dstView->stride()->data()[d] * mBytes;
            lp.src0StrideZYX[d] = src0View->stride()->data()[d] * mBytes;
            lp.src1StrideZYX[d] = src1View->stride()->data()[d] * mBytes;
        }
        lp.cmdSteps[0] = cmd->steps() && cmd->steps()->size() > 0 ? cmd->steps()->data()[0] : 0;
        lp.cmdSteps[1] = cmd->steps() && cmd->steps()->size() > 1 ? cmd->steps()->data()[1] : 0;
        lp.cmdSteps[2] = cmd->steps() && cmd->steps()->size() > 2 ? cmd->steps()->data()[2] : 0;
        lp.cmdViewOffset[0] = dstView->offset();
        lp.cmdViewOffset[1] = src0View->offset();
        lp.cmdViewOffset[2] = src1View->offset();
        lp.outputElementSize = HexagonBackend::getElementSize(outputTensor, mPack);
        lp.input0Size = HexagonBackend::getElementSize(input0, mPack);
        lp.input1Size = HexagonBackend::getElementSize(input1, mPack);

        auto dstDev = HexagonBackend::getDevicePtr(outputTensor);
        auto src0Dev = HexagonBackend::getDevicePtr(input0);
        auto src1Dev = HexagonBackend::getDevicePtr(input1);
        auto iter0Dev = (cmd->iterIndexes() && cmd->iterIndexes()->size() > 0 && cmd->iterIndexes()->data()[0] >= 0)
                            ? HexagonBackend::getDevicePtr(mStack[cmd->iterIndexes()->data()[0]])
                            : std::make_pair(-1, 0);
        auto iter1Dev = (cmd->iterIndexes() && cmd->iterIndexes()->size() > 1 && cmd->iterIndexes()->data()[1] >= 0)
                            ? HexagonBackend::getDevicePtr(mStack[cmd->iterIndexes()->data()[1]])
                            : std::make_pair(-1, 0);
        auto iter2Dev = (cmd->iterIndexes() && cmd->iterIndexes()->size() > 2 && cmd->iterIndexes()->data()[2] >= 0)
                            ? HexagonBackend::getDevicePtr(mStack[cmd->iterIndexes()->data()[2]])
                            : std::make_pair(-1, 0);

        std::vector<std::pair<int, int>> inputFds = {src0Dev, src1Dev, iter0Dev, iter1Dev, iter2Dev};
        std::vector<std::pair<int, int>> outputFds = {dstDev};
        dst.emplace_back();
        dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_BATCH_MATMUL, &params, sizeof(params),
                         inputFds, outputFds, inputs, outputs);
        return NO_ERROR;
    }

    MNN_PRINT("HexagonLoop Error at line 328: NOT_SUPPORT\n");
    return NOT_SUPPORT;

}

HexagonLoop* HexagonLoop::create(Backend* backend, const Op* op) {
    if (op == nullptr || op->type() != OpType_While || op->main_type() != OpParameter_LoopParam) {
        return nullptr;
    }
    auto loop = op->main_as_LoopParam();
    if (loop == nullptr || loop->commands() == nullptr || loop->commands()->size() != 1) {
        MNN_PRINT("HexagonLoop Error at line 584: create failed, return nullptr\n");
        return nullptr;
    }
    auto cmd = loop->commands()->GetAs<RegionCommand>(0);
    if (cmd == nullptr || cmd->op() == nullptr || cmd->fuse() >= 0) {
        MNN_PRINT("HexagonLoop Error at line 588: create failed, return nullptr\n");
        return nullptr;
    }

    auto functions = HexagonRuntime::getDstFunctions();
    if (functions == nullptr ) {
        MNN_PRINT("HexagonLoop Error at line 593: create failed, return nullptr\n");
        return nullptr;
    }

    auto opT = cmd->op();
    if (opT->type() == OpType_UnaryOp && opT->main() == nullptr) {
        return new HexagonLoop(backend, loop);
    }
    if (opT->type() == OpType_BinaryOp && cmd->fuse() < 0 && loop->initCommand() == nullptr) {
        return new HexagonLoop(backend, loop);
    }
    if (opT->type() == OpType_MatMul && cmd->fuse() < 0 && loop->initCommand() == nullptr &&
        cmd->indexes() != nullptr && cmd->indexes()->size() == 3 &&
        cmd->view() != nullptr && cmd->view()->size() == 3) {
        return new HexagonLoop(backend, loop);
    }

    MNN_PRINT("HexagonLoop Error at line 630: create failed, return nullptr\n");
    return nullptr;
}

} // namespace MNN
