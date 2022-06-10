//
//  GeometryComputerUtils.cpp
//  MNN
//
//  Created by MNN on 2020/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/RuntimeFactory.hpp"
#include "shape/SizeComputer.hpp"
#include "core/AutoStorage.h"

#ifdef MNN_BUILD_CODEGEN
#include "OpFuse.hpp"
#endif
#define DEFAULT_ALLOCATE_SIZE 32
namespace MNN {
static bool _hasZeroShapeOutput(const Schedule::PipelineInfo& info) {
    for (auto t : info.outputs) {
        for (int v = 0; v < t->dimensions(); ++v) {
            if (t->length(v) <= 0) {
                return true;
            }
        }
    }
    return false;
}
flatbuffers::Offset<Op> GeometryComputerUtils::makePool(flatbuffers::FlatBufferBuilder& builder, std::pair<int, int> kernel, std::pair<int, int> stride, PoolType type, MNN::PoolPadType pad, std::pair<int, int> pads,  bool isglobal, AvgPoolCountType countType) {
    PoolBuilder poolB(builder);
    poolB.add_type(type);
    poolB.add_padType(pad);
    poolB.add_padX(pads.first);
    poolB.add_padY(pads.second);
    poolB.add_kernelX(kernel.first);
    poolB.add_kernelY(kernel.second);
    poolB.add_strideX(stride.first);
    poolB.add_strideY(stride.second);
    poolB.add_isGlobal(isglobal);
    if (AvgPoolCountType_DEFAULT != countType) {
        poolB.add_countType(countType);
    }
    auto poolOffset = poolB.Finish();
    OpBuilder opB(builder);
    opB.add_type(OpType_Pooling);
    opB.add_main(poolOffset.Union());
    opB.add_main_type(OpParameter_Pool);
    return opB.Finish();
}

int GeometryComputerUtils::buildConstantTensors(std::vector<Schedule::PipelineInfo>& infos) {
    // Check Middle Const
    for (auto& info : infos) {
        if (info.op->type() == OpType_Const) {
            continue;
        }
        bool isConst = true;
        for (int i = 0; i < info.inputs.size(); ++i) {
            if (TensorUtils::getDescribe(info.inputs[i])->usage == Tensor::InsideDescribe::CONSTANT) {
                continue;
            }
            if (OpCommonUtils::opNeedContent(info.op->type(), i)) {
                isConst = false;
                break;
            }
        }
        if (isConst) {
            for (auto t : info.outputs) {
                TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::CONSTANT;
            }
            info.type = Schedule::CONSTANT;
        }
    }
    // Check force size compute op
    int breakIndex = -1;
    for (int infoIndex=0; infoIndex < infos.size(); ++infoIndex) {
        auto& info = infos[infoIndex];
        if (info.op->type() == OpType_Const) {
            continue;
        }
        if (info.op->type() == OpType_Where && info.op->main_type() != OpParameter_Extra) {
            // For compability old model
            continue;
        }
        auto dims = SizeComputer::needInputContent(info.op, info.inputs.size());
        for (auto index : dims) {
            if (index < info.inputs.size()) {
                if (TensorUtils::getDescribe(info.inputs[index])->usage != Tensor::InsideDescribe::CONSTANT) {
                    breakIndex = infoIndex;
                    TensorUtils::getDescribe(info.inputs[index])->usage = Tensor::InsideDescribe::CONSTANT;
                }
            }
        }
    }
    if (breakIndex >= 0) {
        bool hasConst = true;
        while (hasConst) {
            hasConst = false;
            for (auto& info : infos) {
                if (info.type == Schedule::CONSTANT) {
                    continue;
                }
                bool turnConst = false;
                for (auto t : info.outputs) {
                    if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::CONSTANT) {
                        turnConst = true;
                        break;
                    }
                }
                if (turnConst) {
                    for (auto t : info.outputs) {
                        TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::CONSTANT;
                    }
                    for (auto t : info.inputs) {
                        TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::CONSTANT;
                    }
                    info.type = Schedule::CONSTANT;
                    hasConst  = true;
                }
            }
        }
    }
    for (auto& info : infos) {
        if (info.type == Schedule::CONSTANT) {
            for (auto t : info.outputs) {
                TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::CONSTANT;
            }
        }
    }
    return breakIndex;
}

ErrorCode GeometryComputerUtils::shapeComputeAndGeometryTransform(
    std::vector<Schedule::PipelineInfo>& infos,
    GeometryComputer::Context& geoContext,
    std::shared_ptr<Backend> backupBackend,
    Runtime::CompilerType compileType) {
    /** Size Compute and compute Const Begin */
    GeometryComputer::Context ctx(backupBackend, false);
    // Size Compute and compute Const
    for (int i=0; i<infos.size(); ++i) {
        auto& info = infos[i];
        auto& cmdBufferVir = info.executeBuffer;
        auto& tempBuffer = info.cacheBuffer;
        // TODO: Optimize
        cmdBufferVir.command.clear();
        cmdBufferVir.extras.clear();
        for (auto t : info.outputs) {
            if (!TensorUtils::getDescribe(t)->isMutable) {
                continue;
            }
            auto usage = TensorUtils::getDescribe(t)->usage;
            auto type = TensorUtils::getDescribe(t)->memoryType;
            MNN_ASSERT(type != Tensor::InsideDescribe::MEMORY_OUTSIDE);
            MNN_ASSERT(type != Tensor::InsideDescribe::MEMORY_HOST);
            if (TensorUtils::getDescribeOrigin(t)->mContent->count() > 1) {
                TensorUtils::getDescribeOrigin(t)->mContent = new Tensor::InsideDescribe::NativeInsideDescribe;
                t->buffer().dim = TensorUtils::getDescribe(t)->dims;
                TensorUtils::getDescribe(t)->usage = usage;
            } else {
                TensorUtils::getDescribeOrigin(t)->mContent->backend = nullptr;
                if (info.type != Schedule::CONSTANT) {
                    TensorUtils::getDescribeOrigin(t)->mContent->mem.reset(nullptr);
                }
            }
        }
        auto res = SizeComputer::computeOutputSize(info.op, info.inputs, info.outputs);
        if (!res) {
            MNN_ERROR("Compute Shape Error for %s\n", info.op->name()->c_str());
            return COMPUTE_SIZE_ERROR;
        }
        // FIXME: Find better way to may compability for old model
        /**
         For Convolution of 2D / 3D Tensor(Dense / 1D Convolution)
         Because of old code, we will acces dim[2] / dim[3] to get width and height
         Set the lenght to 1 for compability
         */
        for (auto t : info.outputs) {
            TensorUtils::adjustTensorForCompability(t);
        }


        if (info.type == Schedule::CONSTANT) {
            if (_hasZeroShapeOutput(info)) {
                continue;
            }
            ctx.clear();
            auto geo = GeometryComputer::search(info.op->type(), Runtime::Compiler_Loop);
            {
                auto res = geo->onRecompute(info.op, info.inputs, info.outputs, geoContext, tempBuffer);
                if (!res) {
                    tempBuffer.command.clear();
                    tempBuffer.extras.clear();
                    res = geo->onCompute(info.op, info.inputs, info.outputs, geoContext, tempBuffer);
                }
                if (!res) {
                    MNN_ERROR("Const Folder Error in geometry for %s\n", info.op->name()->c_str());
                    return NOT_SUPPORT;
                }
            }
            GeometryComputerUtils::makeRaster(tempBuffer, cmdBufferVir, ctx);
            for (auto t : info.outputs) {
                ctx.getRasterCacheCreateRecursive(t, cmdBufferVir);
            }
            for (auto& cp : cmdBufferVir.command) {
                auto& c = *cp;
                if (nullptr == c.execution) {
                    c.execution.reset(backupBackend->onCreate(c.inputs, c.outputs, c.op));
                }
                auto exe = c.execution;
                if (nullptr == exe.get()) {
                    MNN_ERROR("Const Folder Error for %s\n", info.op->name()->c_str());
                    return NO_EXECUTION;
                }
                for (auto t : c.outputs) {
                    auto des = TensorUtils::getDescribe(t);
                    if (des->backend == nullptr) {
                        TensorUtils::setLinearLayout(t);
                        res = backupBackend->onAcquireBuffer(t, Backend::STATIC);
                        if (!res) {
                            return OUT_OF_MEMORY;
                        }
                        des->backend = backupBackend.get();
                    }
                }
                auto code = exe->onResize(c.inputs, c.outputs);
                if (NO_ERROR != code) {
                    return NOT_SUPPORT;
                }
                code = exe->onExecute(c.inputs, c.outputs);
                if (NO_ERROR != code) {
                    return NOT_SUPPORT;
                }

            }
            // Clear const command
            ctx.pushCache(cmdBufferVir);
            cmdBufferVir.command.clear();
            cmdBufferVir.extras.clear();
        }


    }
    /** Size Compute and compute Const End */

    /** Geometry Transform */
    for (int i=0; i<infos.size(); ++i) {
        auto& info = infos[i];
        auto& cmdBufferReal = info.executeBuffer;
        auto& tempBuffer = info.cacheBuffer;
        // TODO: Optimize
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        if (_hasZeroShapeOutput(info)) {
            continue;
        }
        auto geo = GeometryComputer::search(info.op->type(), compileType);
        {
            auto res = geo->onRecompute(info.op, info.inputs, info.outputs, geoContext, tempBuffer);
            if (!res) {
                tempBuffer.command.clear();
                tempBuffer.extras.clear();
                res = geo->onCompute(info.op, info.inputs, info.outputs, geoContext, tempBuffer);
            }
            if (!res) {
                return NOT_SUPPORT;
            }
            GeometryComputerUtils::makeRaster(tempBuffer, cmdBufferReal, geoContext);
            for (auto t : info.outputs) {
                auto des = TensorUtils::getDescribe(t);
                if (des->usage == Tensor::InsideDescribe::OUTPUT) {
                    // TODO: If output is static and lenght larger than new size, don't clear mem
                    des->mem.reset(nullptr);
                    geoContext.getRasterCacheCreateRecursive(t, cmdBufferReal);
                }
            }
        }
    }
#ifdef MNN_BUILD_CODEGEN
    // fuse op and codegen
    // TODO: Merge CommandBuffer and then Fuse
    {
        for (auto& buf : buffer) {
            opFuse(buf);
        }
    }
#endif
    return NO_ERROR;
}

void GeometryComputerUtils::makeRaster(const CommandBuffer& srcBuffer, CommandBuffer& dstBuffer,
                                       GeometryComputer::Context& ctx) {
    dstBuffer.extras = srcBuffer.extras;
    for (int index = 0; index < srcBuffer.command.size(); ++index) {
        auto& iter = *srcBuffer.command[index];
        const Op* op = iter.op;
        auto& cmd     = iter;
        auto type = op->type();
        MNN_ASSERT(OpType_Raster != type);
        for (int i = 0; i < iter.inputs.size(); ++i) {
            if (!OpCommonUtils::opNeedContent(type, i)) {
                continue;
            }
            auto des = TensorUtils::getDescribe(cmd.inputs[i]);
            //MNN_ASSERT(des->tensorArrayAttr == nullptr);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                ctx.getRasterCacheCreateRecursive(cmd.inputs[i], dstBuffer);
            }
        }
        dstBuffer.command.emplace_back(srcBuffer.command[index]);
    }
}
SharedPtr<Command> GeometryComputerUtils::makeBinary(int type, Tensor* input0, Tensor* input1, Tensor* output) {
    flatbuffers::FlatBufferBuilder builder(DEFAULT_ALLOCATE_SIZE);
    BinaryOpBuilder builder_(builder);
    builder_.add_opType(type);
    auto mainOffset = builder_.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_BinaryOp);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_BinaryOp);
    builder.Finish(opB.Finish());
    SharedPtr<Command> cmdP = new Command;
    auto& cmd = *cmdP;
    cmd.buffer.reset(new BufferStorage);
    cmd.buffer->storage = builder.ReleaseRaw(cmd.buffer->allocated_size, cmd.buffer->offset);
    cmd.inputs  = {input0, input1};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
    return cmdP;
}

SharedPtr<Command> GeometryComputerUtils::makeReduce(ReductionType type, Tensor* input0, Tensor* output) {
    flatbuffers::FlatBufferBuilder builder(DEFAULT_ALLOCATE_SIZE);
    auto vec = builder.CreateVector(std::vector<int>{1});
    ReductionParamBuilder builder_(builder);
    builder_.add_operation(type);
    builder_.add_keepDims(true);
    builder_.add_dim(vec);
    auto mainOffset = builder_.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_Reduction);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_ReductionParam);
    builder.Finish(opB.Finish());
    SharedPtr<Command> cmdP = new Command;
    auto& cmd = *cmdP;
    cmd.buffer.reset(new BufferStorage);
    cmd.buffer->storage = builder.ReleaseRaw(cmd.buffer->allocated_size, cmd.buffer->offset);
    cmd.inputs  = {input0};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
    return cmdP;
}
SharedPtr<Command> GeometryComputerUtils::makeUnary(UnaryOpOperation type, Tensor* input0, Tensor* output) {
    flatbuffers::FlatBufferBuilder builder(DEFAULT_ALLOCATE_SIZE);
    UnaryOpBuilder builder_(builder);
    builder_.add_opType(type);
    auto mainOffset = builder_.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_UnaryOp);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_UnaryOp);
    builder.Finish(opB.Finish());
    SharedPtr<Command> cmdP = new Command;
    auto& cmd = *cmdP;
    cmd.buffer.reset(new BufferStorage);
    cmd.buffer->storage = builder.ReleaseRaw(cmd.buffer->allocated_size, cmd.buffer->offset);
    cmd.inputs  = {input0};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
    return cmdP;
}
SharedPtr<Command> GeometryComputerUtils::makeCommand(flatbuffers::FlatBufferBuilder& builder, const std::vector<Tensor*>& inputs,
                                           const std::vector<Tensor*>& outputs) {
    SharedPtr<Command> cmdP = new Command;
    auto& cmd = *cmdP;
    cmd.buffer.reset(new BufferStorage);
    cmd.buffer->storage = builder.ReleaseRaw(cmd.buffer->allocated_size, cmd.buffer->offset);
    cmd.outputs = outputs;
    cmd.inputs  = inputs;
    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
    return cmdP;
}

SharedPtr<Command> GeometryComputerUtils::makeMatMul(Tensor* input0, Tensor* input1, Tensor* output, Tensor* Bias, bool transposeA,
                                          bool transposeB) {
    SharedPtr<Command> cmdP = new Command;
    auto& cmd = *cmdP;
    flatbuffers::FlatBufferBuilder builder(DEFAULT_ALLOCATE_SIZE);
    MatMulBuilder builder_(builder);
    builder_.add_transposeA(transposeA);
    builder_.add_transposeB(transposeB);
    auto mainOffset = builder_.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_MatMul);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_MatMul);
    builder.Finish(opB.Finish());
    cmd.buffer.reset(new BufferStorage);
    cmd.buffer->storage = builder.ReleaseRaw(cmd.buffer->allocated_size, cmd.buffer->offset);
    if (nullptr == Bias) {
        cmd.inputs = {input0, input1};
    } else {
        cmd.inputs = {input0, input1, Bias};
    }
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
    return cmdP;
}

Tensor::InsideDescribe::Region GeometryComputerUtils::makeRawAddressRef(Tensor* src, int srcOffset, int size,
                                                                        int dstOffset) {
    Tensor::InsideDescribe::Region reg;
    // Default is 1, 1, 1
    reg.size[2] = size;

    // Default is 0, 1, 1, 1
    reg.src.offset = srcOffset;
    reg.dst.offset = dstOffset;
    reg.origin     = src;
    return reg;
}

void GeometryComputerUtils::makeRawAddressRef(Tensor* dst, Tensor* src, int srcOffset, int size, int dstOffset) {
    auto describe        = TensorUtils::getDescribe(dst);
    describe->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
    describe->regions    = {makeRawAddressRef(src, srcOffset, size, dstOffset)};
}

}; // namespace MNN
