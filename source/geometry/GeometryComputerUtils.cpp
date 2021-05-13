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
#include <unordered_map>
#include "core/AutoStorage.h"

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
            AutoRelease<Execution> exe(backupBackend->onCreate(info.inputs, info.outputs, info.op));
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
    bool hasSizeComputeOp = false;
    for (auto& info : infos) {
        if (info.op->type() == OpType_Const) {
            continue;
        }
        auto dims = SizeComputer::needInputContent(info.op, info.inputs.size());
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
    std::unordered_map<const Tensor*, std::string> outputTensorName;
    // Size Compute and compute Const
    for (auto& info : infos) {
        if (info.op->type() == OpType_Const) {
            continue;
        }
        if (info.op->name() != nullptr) {
            for (auto t : info.outputs) {
                outputTensorName.insert(std::make_pair(t, info.op->name()->str()));
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
                AutoRelease<Execution> exe(backupBackend->onCreate(c.inputs, c.outputs, c.op));
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
#ifdef MNN_ADD_NAME
        std::unordered_map<std::string, int> nameIdx;
        auto getName = [&nameIdx](const std::string& name) {
            auto iter = nameIdx.find(name);
            if (iter == nameIdx.end()) {
                nameIdx[name] = 0;
                return name;
            }
            iter->second += 1;
            char str[200];
            sprintf(str, "%s#%d", name.c_str(), iter->second);
            return std::string(str);
        };
        // set name for geometry op
        for (int i = buffer.command.size() - 1; i >= 0; i--) {
            auto& cmd = buffer.command[i];
            // has name, dont deal
            if (!cmd.name.empty()) {
                continue;
            }
            std::vector<Tensor*> inputs, outputs;
            if (!cmd.inputs.empty() &&
                TensorUtils::getDescribe(cmd.inputs[0])->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                outputs = { cmd.inputs[0], cmd.outputs[0] };
                for (const auto& r : TensorUtils::getDescribe(cmd.inputs[0])->regions) {
                    inputs.push_back(r.origin);
                }
            } else {
                inputs = cmd.inputs;
                outputs = cmd.outputs;
            }

            std::string name;
            // set name by output tensor
            for (const Tensor* t : outputs) {
                auto iter = outputTensorName.find(t);
                if (iter != outputTensorName.end()) {
                    name = getName(iter->second);
                    cmd.name = name;
                    break;
                }
            }
            // if input tensor is hidden tensor, set same name
            if (!name.empty()) {
                for (const Tensor* t : inputs) {
                    if (outputTensorName.find(t) == outputTensorName.end()) {
                        outputTensorName.insert(std::make_pair(t, name));
                    }
                }
            }
        }
#endif
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
    for (int index = 0; index < srcBuffer.command.size(); ++index) {
        auto& iter = srcBuffer.command[index];
        const Op* op = iter.op;
        auto cmd     = iter;
        if (!iter.buffer.empty()) {
            op = flatbuffers::GetRoot<Op>((void*)iter.buffer.data());
        }
        auto type = op->type();
        MNN_ASSERT(OpType_Raster != type);
        for (int i = 0; i < iter.inputs.size(); ++i) {
            if (!OpCommonUtils::opNeedContent(type, i)) {
                continue;
            }
            auto des = TensorUtils::getDescribe(cmd.inputs[i]);
            MNN_ASSERT(des->tensorArrayAttr == nullptr);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                ctx.getRasterCacheCreateRecurrse(cmd.inputs[i], dstBuffer);
            }
        }
        dstBuffer.command.emplace_back(std::move(cmd));
    }
    auto& outputs = ctx.pOutputs;
    for (auto& o : ctx.pOutputs) {
        ctx.getRasterCacheCreateRecurrse(o, dstBuffer);
    }
}
Command GeometryComputerUtils::makeBinary(int type, Tensor* input0, Tensor* input1, Tensor* output) {
    flatbuffers::FlatBufferBuilder builder;
    BinaryOpBuilder builder_(builder);
    builder_.add_opType(type);
    auto mainOffset = builder_.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_BinaryOp);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_BinaryOp);
    builder.Finish(opB.Finish());
    Command cmd;
    cmd.buffer.resize(builder.GetSize());
    ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
    cmd.inputs  = {input0, input1};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
    return cmd;
}

Command GeometryComputerUtils::makeReduce(ReductionType type, Tensor* input0, Tensor* output) {
    flatbuffers::FlatBufferBuilder builder;
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
    Command cmd;
    cmd.buffer.resize(builder.GetSize());
    ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
    cmd.inputs  = {input0};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
    return cmd;
}
Command GeometryComputerUtils::makeUnary(UnaryOpOperation type, Tensor* input0, Tensor* output) {
    flatbuffers::FlatBufferBuilder builder;
    UnaryOpBuilder builder_(builder);
    builder_.add_opType(type);
    auto mainOffset = builder_.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_UnaryOp);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_UnaryOp);
    builder.Finish(opB.Finish());
    Command cmd;
    cmd.buffer.resize(builder.GetSize());
    ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
    cmd.inputs  = {input0};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
    return cmd;
}
Command GeometryComputerUtils::makeCommand(flatbuffers::FlatBufferBuilder& builder, const std::vector<Tensor*>& inputs,
                                           const std::vector<Tensor*>& outputs) {
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
    flatbuffers::FlatBufferBuilder builder;
    MatMulBuilder builder_(builder);
    builder_.add_transposeA(transposeA);
    builder_.add_transposeB(transposeB);
    auto mainOffset = builder_.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_MatMul);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_MatMul);
    builder.Finish(opB.Finish());
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
