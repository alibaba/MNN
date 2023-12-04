//
//  CUDALoop.cpp
//  MNN
//
//  Created by MNN on b'2021/04/20'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/core/CUDABackend.hpp"
#include "Raster.cuh"

#include "MatMulExecution.hpp"

namespace MNN {
namespace CUDA {
class CUDALoop : public Execution {
public:
    struct Unit {
        std::vector<Tensor*> inputs;
        std::vector<Tensor*> outputs;
        std::shared_ptr<Execution> exe;
    };
    CUDALoop(Backend* bn, const LoopParam* loop) : Execution(bn) {
        // The LoopParam is created by geometry, won't be released
        mLoop = loop;
        mStack.resize(loop->tensorNumber());
        mExecutions.resize(loop->commands()->size());
        mStackPtr.resize(loop->tensorNumber());
    }
    virtual ~ CUDALoop() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto bytes = static_cast<CUDABackend*>(backend())->getBytes(outputs[0]);
        auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
        if (1 == mLoop->commands()->size()) {
            auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
            auto op = cmd->op();
            if (OpType_MatMul == op->type() && mLoop->parallel() && mLoop->loopNumber() > 1) {
                auto step = cmd->steps()->data();
                if (inputs.size() <= 3) {
                    auto& unit = mExecutions[0];
                    int as = 1, bs = 1, cs = 1;
                    if (step[1] == 0) {
                        as = 0;
                    }
                    if (step[2] == 0) {
                        bs = 0;
                    }
                    unit.exe.reset(new MatMulExecution(op->main_as_MatMul()->transposeA(),  op->main_as_MatMul()->transposeB(), backend(), as, bs, cs));
                    if (nullptr == unit.exe) {
                        return OUT_OF_MEMORY;
                    }
                    unit.inputs = inputs;
                    unit.outputs = outputs;
                    auto code = unit.exe->onResize(unit.inputs, unit.outputs);
                    if (NO_ERROR != code) {
                        return code;
                    }
                    mSingleMatMul = true;
                    return NO_ERROR;
                }
            }
        }

        mMidTensors.clear();
        mIndiceCopy.clear();
        int inputIndexSize = mLoop->inputIndexes()->size();
        MNN_ASSERT(inputIndexSize == inputs.size());
        for (int i=0; i<inputIndexSize; ++i) {
            mStack[mLoop->inputIndexes()->data()[i]] = inputs[i];
        }
        int outputIndexSize = mLoop->outputIndexes()->size();
        MNN_ASSERT(outputIndexSize == outputs.size());
        for (int i=0; i<outputIndexSize; ++i) {
            mStack[mLoop->outputIndexes()->data()[i]] = outputs[i];
        }
        if (1 == mLoop->commands()->size()) {
            auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
            auto op = cmd->op();
            if (OpType_UnaryOp == op->type() && nullptr == op->main()) {
                return NO_ERROR;
            }
        }
        for (int i=0; i<mLoop->commands()->size(); ++i) {
            auto cmd = mLoop->commands()->GetAs<RegionCommand>(i);
            auto op = cmd->op();
            auto& unit = mExecutions[i];
            // Find indice and copy to cpu
            int size = cmd->iterIndexes()->size();
            for (int v=0; v<size; ++v) {
                auto tensorIndex = cmd->indexes()->data()[v];
                auto tensor = mStack[tensorIndex];
                auto iterIndex = cmd->iterIndexes()->data()[v];
                if (iterIndex >= 0 && mStack[iterIndex]->host<void>() == nullptr) {
                    std::shared_ptr<Tensor> tensorHost(new Tensor(mStack[iterIndex], mStack[iterIndex]->getDimensionType()));
                    mIndiceCopy.insert(std::make_pair(mStack[iterIndex], tensorHost.get()));
                    mStack[iterIndex] = tensorHost.get();
                    mMidTensors.emplace_back(std::move(tensorHost));
                }
            }
            // Prepare for MatMul
            if (OpType_MatMul == op->type()) {
                bool transposeC = true;
                int e = cmd->size()->data()[0];
                int l = cmd->size()->data()[1];
                int h = cmd->size()->data()[2];
                std::shared_ptr<Tensor> A, B, C, Bias;
                C.reset(Tensor::createDevice<float>({e, h}));
                if (op->main_as_MatMul()->transposeA()) {
                    A.reset(Tensor::createDevice<float>({l, e}));
                } else {
                    A.reset(Tensor::createDevice<float>({e, l}));
                }
                if (op->main_as_MatMul()->transposeB()) {
                    B.reset(Tensor::createDevice<float>({h, l}));
                } else {
                    B.reset(Tensor::createDevice<float>({l, h}));
                }
                auto view = cmd->view()->GetAs<View>(0);
                if (view->stride()->data()[0] == 1) {
                    transposeC = false;
                }
                if (cmd->indexes()->size() > 3) {
                    Bias.reset(Tensor::createDevice<float>({h}));
                    unit.inputs = {A.get(), B.get(), Bias.get()};
                } else {
                    unit.inputs = {A.get(), B.get()};
                }
                unit.outputs = {C.get()};
                unit.exe.reset(new MatMulExecution(op->main_as_MatMul()->transposeA(),  op->main_as_MatMul()->transposeB(), backend()));
                if (nullptr == unit.exe) {
                    return OUT_OF_MEMORY;
                }
                auto code = unit.exe->onResize(unit.inputs, unit.outputs);
                if (NO_ERROR != code) {
                    return code;
                }
                mMidTensors.emplace_back(A);
                mMidTensors.emplace_back(B);
                mMidTensors.emplace_back(C);
                mMidTensors.emplace_back(Bias);
                continue;
            }
        }
        return NO_ERROR;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor *> &originInputs, const std::vector<Tensor *> &originOutputs) override {
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        if (mSingleMatMul) {
            auto& unit = mExecutions[0];
            unit.inputs = originInputs;
            unit.outputs = originOutputs;

            auto code = unit.exe->onExecute(unit.inputs, unit.outputs);
            if (NO_ERROR != code) {
                return code;
            }
            return NO_ERROR;
        }
        if (nullptr != mLoop->initCommand()) {
            for (int i=0; i<mLoop->initCommand()->size(); ++i) {
                auto cmd = mLoop->initCommand()->GetAs<RegionCommand>(i);
                auto index = cmd->indexes()->data()[0];
                auto tensor = mStack[index];
                auto size = static_cast<CUDABackend*>(backend())->realSize(tensor) * sizeof(float);
                runtime->memset((void*)tensor->deviceId(), 0, size);
            }
        }
        if (1 == mLoop->commands()->size()) {
            auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
            auto op = cmd->op();

            if (OpType_UnaryOp == op->type() && nullptr == op->main() && cmd->fuse() < 0) {
                Tensor::InsideDescribe::Region reg;
                auto srcView = cmd->view()->GetAs<View>(1);
                auto dstView = cmd->view()->GetAs<View>(0);
                ::memcpy(reg.size, cmd->size()->data(), 3 * sizeof(int32_t));
                ::memcpy(reg.src.stride, srcView->stride()->data(), 3 * sizeof(int32_t));
                ::memcpy(reg.dst.stride, dstView->stride()->data(), 3 * sizeof(int32_t));
                auto input = mStack[cmd->indexes()->data()[1]];
                auto inputSize = input->elementSize();
                auto output = mStack[cmd->indexes()->data()[0]];
                auto bytes = static_cast<CUDABackend*>(backend())->getBytes(input);
                auto step0 = cmd->steps()->data()[0];
                auto step1 = cmd->steps()->data()[1];
                auto loopNumber = mLoop->loopNumber();
                auto index0 = cmd->iterIndexes()->data()[0];
                const int32_t* dstIndice = nullptr;
                if (index0 >= 0) {
                    dstIndice = (int32_t*)originInputs[index0]->deviceId();
                }
                auto index1 = cmd->iterIndexes()->data()[1];
                const int32_t* srcIndice = nullptr;
                if (index1 >= 0) {
                    srcIndice = (int32_t*)originInputs[index1]->deviceId();
                }
                auto src = (uint8_t*)(input->deviceId()) + srcView->offset() * bytes;
                auto dstOrigin = (output->deviceId()) + dstView->offset() * bytes;
                auto dst = dstOrigin;
                BlitWithIndice(
                        (uint8_t*)dst,
                        (uint8_t*)src,
                        dstIndice, srcIndice, index0, index1,
                        loopNumber, step0, step1, input->elementSize(),
                        reg, bytes, runtime);


                if(cmd->fuse() >= 0) {
                    auto opType = cmd->fuse();
                    auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
                    auto srcStride0 = dstStride;
                    auto srcStride1 = dstStride;
                    int32_t tmpSize[3];
                    ::memcpy(tmpSize, cmd->size()->data(), 3 * sizeof(int32_t));
                    tmpSize[0] *=  loopNumber;
                    auto type = halide_type_of<float>();
                    if (static_cast<CUDABackend*>(backend())->useFp16()) {
                        type.bits = 16;
                    }
                    // MNN_PRINT("Binary Loop in optype:%d\n", opType);
                    BinaryBlit((uint8_t*)dstOrigin, (uint8_t*)dstOrigin, (const uint8_t*)dst,
                        tmpSize, srcStride0, srcStride1, dstStride, type, runtime, opType);
                }
                return NO_ERROR;
            }
        }
        // Copy Index
        for (auto& iter : mIndiceCopy) {
            backend()->onCopyBuffer(iter.first, iter.second);
        }
        auto bytes = static_cast<CUDABackend*>(backend())->getBytes(originOutputs[0]);
        for (int iter=0; iter < mLoop->loopNumber(); ++iter) {
            for (int index=0; index<mLoop->commands()->size(); ++index) {
                auto cmd = mLoop->commands()->GetAs<RegionCommand>(index);
                auto op = cmd->op();
                int size = cmd->iterIndexes()->size();
                for (int v=0; v<size; ++v) {
                    auto tensorIndex = cmd->indexes()->data()[v];
                    auto tensor = mStack[tensorIndex];
                    auto iterIndex = cmd->iterIndexes()->data()[v];
                    auto offset = iter;
                    if (iterIndex >= 0) {
                        offset = mStack[iterIndex]->host<int32_t>()[iter];
                    }
                    auto view = cmd->view()->GetAs<View>(v);
                    offset = offset * cmd->steps()->data()[v] + view->offset();
                    mStackPtr[tensorIndex] = tensor->deviceId() + offset * static_cast<CUDABackend*>(backend())->getBytes(tensor);
                }

                auto dstOrigin = mStackPtr[cmd->indexes()->data()[0]];
                auto dst = dstOrigin;
                auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();

                if (OpType_UnaryOp == op->type()) {

                    auto src = (float*)mStackPtr[cmd->indexes()->data()[1]];
                    int unaryType = op->main_as_UnaryOp()->opType();

                    auto srcStride = cmd->view()->GetAs<View>(1)->stride()->data();
                    UnaryBlit((uint8_t*)dst, (const uint8_t*)src, cmd->size()->data(), srcStride, dstStride, bytes, runtime, unaryType);
                    continue;
                }
                if (OpType_MatMul == op->type()) {
                    auto& unit = mExecutions[index];
                    if (3 == size) {
                        unit.inputs[0]->buffer().device = mStackPtr[cmd->indexes()->data()[1]];
                        unit.inputs[1]->buffer().device = mStackPtr[cmd->indexes()->data()[2]];
                        unit.outputs[0]->buffer().device = dst;
                    } else {
                        MNN_ASSERT(4 == size);
                        unit.inputs[0]->buffer().device = mStackPtr[cmd->indexes()->data()[1]];
                        unit.inputs[1]->buffer().device = mStackPtr[cmd->indexes()->data()[2]];
                        unit.inputs[2]->buffer().device = mStackPtr[cmd->indexes()->data()[3]];
                        unit.outputs[0]->buffer().device = dst;
                    }
                    unit.exe->onExecute(unit.inputs, unit.outputs);
                    continue;
                }
                if (OpType_BinaryOp == op->type()) {
                    auto type = halide_type_of<float>();
                    if (static_cast<CUDABackend*>(backend())->useFp16()) {
                        type.bits = 16;
                    }
                    auto src0 = mStackPtr[cmd->indexes()->data()[1]];
                    auto src1 = mStackPtr[cmd->indexes()->data()[2]];
                    auto opType = op->main_as_BinaryOp()->opType();
                    auto srcStride0 = cmd->view()->GetAs<View>(1)->stride()->data();
                    auto srcStride1 = cmd->view()->GetAs<View>(2)->stride()->data();
                    if (cmd->fuse() == 0) {
                        BinaryBlitFuse((uint8_t*)dst, (const uint8_t*)src0, (const uint8_t*)src1,
                            cmd->size()->data(), srcStride0, srcStride1, dstStride, type, runtime, opType);
                    } else {
                        // MNN_PRINT("Binary Loop in optype:%d\n", opType);
                        BinaryBlit((uint8_t*)dst, (const uint8_t*)src0, (const uint8_t*)src1,
                            cmd->size()->data(), srcStride0, srcStride1, dstStride, type, runtime, opType);
                    }
                }
            }
        }
        return NO_ERROR;
    }
private:
    const LoopParam* mLoop;
    std::vector<Tensor*> mStack;
    std::vector<std::shared_ptr<Tensor>> mMidTensors;
    std::vector<Unit> mExecutions;
    std::vector<uint64_t> mStackPtr;
    std::map<Tensor*, Tensor*> mIndiceCopy;
    bool mSingleMatMul = false;
};

class LoopCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->main_type() != OpParameter_LoopParam) {
            return nullptr;
        }
        auto mLoop = op->main_as_LoopParam();
        for (int i=0; i<mLoop->commands()->size(); ++i) {
            auto cmd = mLoop->commands()->GetAs<RegionCommand>(i);

            if(cmd->fuse() > 0) {
                // Currently don't need not add fuse
                return nullptr;//
            }
            if(cmd->fuse() == 0) {
                if (cmd->op()->type() != OpType_BinaryOp) {
                    // TODO: support afterwards
                    return nullptr;
                }
                auto bytes = static_cast<CUDABackend*>(backend)->getBytes(outputs[0]);
                if (2 == bytes) {
                    return nullptr;
                }
            }
        }
        if (nullptr != mLoop->initCommand()) {
            for (int i=0; i<mLoop->initCommand()->size(); ++i) {
                auto cmd = mLoop->initCommand()->GetAs<RegionCommand>(i);
                if (nullptr != cmd->op()) {
                    // Currently don't support other init
                    return nullptr;
                }
            }

        }

        return new CUDALoop(backend, op->main_as_LoopParam());
    }
};

static CUDACreatorRegister<LoopCreator> __init(OpType_While);

};
};
