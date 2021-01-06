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
#ifdef MNN_BUILD_CODEGEN
#include "OpFuse.hpp"
#endif
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
void GeometryComputerUtils::buildConstantTensors(std::vector<Schedule::PipelineInfo>& infos,
                                                 std::shared_ptr<Backend> backupBackend, bool netBufferHold,
                                                 std::vector<Tensor*>& constTensors,
                                                 std::vector<Tensor*>& midConstTensors) {
    // Create Const Tensors
    for (auto& info : infos) {
        if (info.op->type() != OpType_Const) {
            continue;
        }
        SizeComputer::computeOutputSize(info.op, info.inputs, info.outputs);
        for (auto t : info.outputs) {
            TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::CONSTANT;
        }
        info.type                                        = Schedule::CONSTANT;
        TensorUtils::getDescribe(info.outputs[0])->usage = Tensor::InsideDescribe::CONSTANT;
        TensorUtils::setLinearLayout(info.outputs[0]);
        if (_hasZeroShapeOutput(info)) {
            continue;
        }
        auto parameter                                     = info.op->main_as_Blob();
        TensorUtils::getDescribe(info.outputs[0])->backend = backupBackend.get();
        if (netBufferHold && (parameter->dataType() != DataType_DT_HALF)) {
            // The net buffer will be hold by user, we can directly use it
            info.outputs[0]->buffer().host = (uint8_t*)OpCommonUtils::blobData(info.op);
        } else {
            // The net buffer may be released later, or we can't directly use it (for half we need cast to float)
            auto res = backupBackend->onAcquireBuffer(info.outputs[0], Backend::STATIC);
            if (!res) {
                MNN_ERROR("Error for alloc const in pipeline\n");
                return;
            }
            TensorUtils::getDescribe(info.outputs[0])->backend = backupBackend.get();
            std::shared_ptr<Execution> exe(backupBackend->onCreate(info.inputs, info.outputs, info.op));
            exe->onResize(info.inputs, info.outputs);
            exe->onExecute(info.inputs, info.outputs);
            constTensors.emplace_back(info.outputs[0]);
        }
    }
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
            if (SizeComputer::opNeedContent(info.op->type(), i)) {
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
    bool hasSizeComputeOp = false;
    for (auto& info : infos) {
        if (info.op->type() == OpType_Const) {
            continue;
        }
        auto dims = SizeComputer::needInputContent(info.op);
        for (auto index : dims) {
            if (index < info.inputs.size()) {
                if (TensorUtils::getDescribe(info.inputs[index])->usage != Tensor::InsideDescribe::CONSTANT) {
                    hasSizeComputeOp                                    = true;
                    TensorUtils::getDescribe(info.inputs[index])->usage = Tensor::InsideDescribe::CONSTANT;
                }
            }
        }
    }
    if (hasSizeComputeOp) {
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
        if (info.op->type() == OpType_Const) {
            continue;
        }
        if (info.type == Schedule::CONSTANT) {
            for (auto t : info.outputs) {
                TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::CONSTANT;
                midConstTensors.emplace_back(t);
            }
        }
    }
}

ErrorCode GeometryComputerUtils::shapeComputeAndGeometryTransform(
    std::vector<Schedule::PipelineInfo>& infos,
    CommandBuffer& buffer,
    GeometryComputer::Context& geoContext,
    std::shared_ptr<Backend> backupBackend,
    bool geometry) {
    /** Size Compute and compute Const Begin */
    GeometryComputer::Context ctx(backupBackend, false);

    // Size Compute and compute Const
    for (auto& info : infos) {
        if (info.op->type() == OpType_Const) {
            continue;
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
            CommandBuffer tempSrcbuffer;
            CommandBuffer tempDstBuffer;
            auto geo = GeometryComputer::search(info.op->type());
            {
                res = geo->compute(info.op, info.inputs, info.outputs, ctx, tempSrcbuffer);
                if (!res) {
                    MNN_ERROR("Const Folder Error in geometry for %s\n", info.op->name()->c_str());
                    return NOT_SUPPORT;
                }
            }
            GeometryComputerUtils::makeRaster(tempSrcbuffer, tempDstBuffer, ctx);
            for (auto& c : tempDstBuffer.command) {
                std::shared_ptr<Execution> exe(backupBackend->onCreate(c.inputs, c.outputs, c.op));
                if (nullptr == exe) {
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
            for (auto& c : tempDstBuffer.command) {
                for (auto t : c.outputs) {
                    if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                        backupBackend->onReleaseBuffer(t, Backend::STATIC);
                    }
                }
            }
        }
    }
    /** Size Compute and compute Const End */

    /** Geometry Transform */
    if (geometry) {
        CommandBuffer tmpBuffer;
        for (auto& info : infos) {
            if (info.type == Schedule::CONSTANT) {
                continue;
            }
            if (_hasZeroShapeOutput(info)) {
                continue;
            }
            auto geo = GeometryComputer::search(info.op->type());
            {
                bool res = geo->compute(info.op, info.inputs, info.outputs, geoContext, tmpBuffer);
                if (!res) {
                    return NOT_SUPPORT;
                }
            }
        }
        GeometryComputerUtils::makeRaster(tmpBuffer, buffer, geoContext);
    } else {
        for (auto& info : infos) {
            if (info.type == Schedule::CONSTANT) {
                continue;
            }
            if (_hasZeroShapeOutput(info)) {
                continue;
            }
            Command command;
            command.op = info.op;
            command.inputs = info.inputs;
            command.outputs = info.outputs;
            buffer.command.emplace_back(std::move(command));
        }
    }
#ifdef MNN_BUILD_CODEGEN
    // fuse op and codegen
    {
        opFuse(buffer);
    }
#endif
    return NO_ERROR;
}

void GeometryComputerUtils::makeRaster(const CommandBuffer& srcBuffer, CommandBuffer& dstBuffer,
                                       GeometryComputer::Context& ctx) {
    dstBuffer.extras = std::move(srcBuffer.extras);
    for (auto& iter : srcBuffer.command) {
        const Op* op = iter.op;
        auto cmd     = iter;
        if (!iter.buffer.empty()) {
            op = flatbuffers::GetRoot<Op>((void*)iter.buffer.data());
        }
        auto type = op->type();
        if (OpType_Raster == type) {
            bool exist = false;
            for (int i = 0; i < dstBuffer.command.size() && !exist; i++) {
                exist |= (dstBuffer.command[i].outputs[0] == cmd.outputs[0]);
            }
            if (!exist) {
                dstBuffer.command.emplace_back(std::move(cmd));
            }
            continue;
        }
        for (int i = 0; i < iter.inputs.size(); ++i) {
            if (!SizeComputer::opNeedContent(type, i)) {
                continue;
            }
            auto des = TensorUtils::getDescribe(cmd.inputs[i]);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                cmd.inputs[i] = ctx.getRasterCacheCreateRecurrse(cmd.inputs[i], dstBuffer);
            }
        }
        dstBuffer.command.emplace_back(std::move(cmd));
    }
}
Command GeometryComputerUtils::makeBinary(int type, Tensor* input0, Tensor* input1, Tensor* output) {
    std::unique_ptr<OpT> mul(new OpT);
    mul->type                      = OpType_BinaryOp;
    mul->main.type                 = OpParameter_BinaryOp;
    mul->main.value                = new BinaryOpT;
    mul->main.AsBinaryOp()->opType = type;
    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = Op::Pack(builder, mul.get());
    builder.Finish(lastOffset);
    Command cmd;
    cmd.buffer.resize(builder.GetSize());
    ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
    cmd.inputs  = {input0, input1};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
    return cmd;
}

Command GeometryComputerUtils::makeReduce(ReductionType type, Tensor* input0, Tensor* output) {
    std::unique_ptr<OpT> sum(new OpT);
    sum->type                               = OpType_Reduction;
    sum->main.type                          = OpParameter_ReductionParam;
    sum->main.value                         = new ReductionParamT;
    sum->main.AsReductionParam()->dim       = {1};
    sum->main.AsReductionParam()->keepDims  = true;
    sum->main.AsReductionParam()->operation = type;
    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = Op::Pack(builder, sum.get());
    builder.Finish(lastOffset);
    Command cmd;
    cmd.buffer.resize(builder.GetSize());
    ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
    cmd.inputs  = {input0};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
    return cmd;
}
Command GeometryComputerUtils::makeUnary(UnaryOpOperation type, Tensor* input0, Tensor* output) {
    std::unique_ptr<OpT> sum(new OpT);
    sum->type                     = OpType_UnaryOp;
    sum->main.type                = OpParameter_UnaryOp;
    sum->main.value               = new UnaryOpT;
    sum->main.AsUnaryOp()->opType = type;
    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = Op::Pack(builder, sum.get());
    builder.Finish(lastOffset);
    Command cmd;
    cmd.buffer.resize(builder.GetSize());
    ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
    cmd.inputs  = {input0};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
    return cmd;
}
Command GeometryComputerUtils::makeCommand(const OpT* op, const std::vector<Tensor*>& inputs,
                                           const std::vector<Tensor*>& outputs) {
    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = Op::Pack(builder, op);
    builder.Finish(lastOffset);
    Command cmd;
    cmd.buffer.resize(builder.GetSize());
    ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
    cmd.outputs = outputs;
    cmd.inputs  = inputs;
    cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
    return cmd;
}

Command GeometryComputerUtils::makeMatMul(Tensor* input0, Tensor* input1, Tensor* output, Tensor* Bias, bool transposeA,
                                          bool transposeB) {
    std::unique_ptr<OpT> matmul(new OpT);
    matmul->type                        = OpType_MatMul;
    matmul->main.type                   = OpParameter_MatMul;
    matmul->main.value                  = new MatMulT;
    matmul->main.AsMatMul()->transposeA = transposeA;
    matmul->main.AsMatMul()->transposeB = transposeB;
    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = Op::Pack(builder, matmul.get());
    builder.Finish(lastOffset);
    Command cmd;
    cmd.buffer.resize(builder.GetSize());
    ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
    if (nullptr == Bias) {
        cmd.inputs = {input0, input1};
    } else {
        cmd.inputs = {input0, input1, Bias};
    }
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
    return cmd;
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

void GeometryComputerUtils::makeSliceRef(Tensor* dst, Tensor* src, const std::vector<int>& originSize,
                                         const std::vector<int>& offset, const std::vector<int>& dstSize) {
    auto describe        = TensorUtils::getDescribe(dst);
    describe->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
    Tensor::InsideDescribe::Region reg;
    reg.origin  = src;
    reg.size[0] = dstSize[0];
    reg.size[1] = dstSize[1];
    reg.size[2] = dstSize[2];

    reg.src.offset    = offset[0] * originSize[1] * originSize[2] + offset[1] * originSize[2] + offset[2];
    reg.src.stride[0] = originSize[1] * originSize[2];
    reg.src.stride[1] = originSize[2];
    reg.src.stride[2] = 1;

    reg.dst.offset    = 0;
    reg.dst.stride[0] = dstSize[1] * dstSize[2];
    reg.dst.stride[1] = dstSize[2];
    reg.dst.stride[2] = 1;
    describe->regions = {reg};
}
}; // namespace MNN
