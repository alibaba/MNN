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
#include "core/FileLoader.hpp"
#ifdef MNN_BUILD_CODEGEN
#include "OpFuse.hpp"
#endif
#define DEFAULT_ALLOCATE_SIZE 32
namespace MNN {
static bool _hasZeroShapeOutput(const Schedule::OpCacheInfo& info) {
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

int GeometryComputerUtils::buildConstantTensors(std::vector<Schedule::OpCacheInfo>& infos) {
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
            if (OpCommonUtils::opNeedContent(info.op, i)) {
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
                auto des = TensorUtils::getDescribe(info.inputs[index]);
                des->stageMask |= MNN::Tensor::InsideDescribe::StageInfo::GEOMETRY_STAGE;
                if (des->usage != Tensor::InsideDescribe::CONSTANT) {
                    breakIndex = infoIndex;
                    TensorUtils::getDescribe(info.inputs[index])->usage = Tensor::InsideDescribe::CONSTANT;
                }
                if (des->isMutable) {
                    info.computeCache.addContentIndex(index);
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
            for (auto t : info.inputs) {
                TensorUtils::getDescribe(t)->stageMask |= MNN::Tensor::InsideDescribe::StageInfo::GEOMETRY_STAGE;
            }
            for (auto t : info.outputs) {
                TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::CONSTANT;
            }
        }
    }
    return breakIndex;
}

ErrorCode GeometryComputerUtils::shapeComputeAndGeometryTransform(
    FileLoader* external,
    std::vector<Schedule::OpCacheInfo>& infos,
    GeometryComputer::Context& geoContext,
    std::shared_ptr<Backend> backupBackend,
    Runtime::CompilerType compileType, 
    bool skipShapeCompute,
    bool permitCodegen) {
    bool openCache = geoContext.support(Interpreter::GeometryComputeMask::GEOMETRCOMPUTEMASK_OPENCACHE);
    /** Size Compute and compute Const Begin */
    GeometryComputer::Context ctx(Interpreter::GeometryComputeMask::GEOMETRCOMPUTEMASK_ALL, backupBackend);
    // Size Compute and compute Const
    for (int i=0; i<infos.size(); ++i) {
        auto& info = infos[i];
        auto& cmdBufferVir = info.executeBuffer;
        auto& tempBuffer = info.cacheBuffer;
        // TODO: Optimize
        for (auto t : info.outputs) {
            if (!TensorUtils::getDescribe(t)->isMutable) {
                continue;
            }
            auto des = TensorUtils::getDescribe(t);
            auto usage = des->usage;
            auto type = des->memoryType;
            MNN_ASSERT(type != Tensor::InsideDescribe::MEMORY_OUTSIDE);
            MNN_ASSERT(type != Tensor::InsideDescribe::MEMORY_HOST);
            if (TensorUtils::getDescribeOrigin(t)->mContent.use_count() > 1) {
                TensorUtils::getDescribeOrigin(t)->mContent.reset(new  Tensor::InsideDescribe::NativeInsideDescribe);
                t->buffer().dim = TensorUtils::getDescribe(t)->dims;
                TensorUtils::getDescribeOrigin(t)->setBackend(nullptr);
                TensorUtils::getDescribeOrigin(t)->mem = nullptr;
                TensorUtils::getDescribe(t)->usage = usage;
                info.computeCache.close();
            } else if (des->group == 0) {
                if (info.type != Schedule::CONSTANT && usage != Tensor::InsideDescribe::TRAINABLE) {
                    TensorUtils::getDescribeOrigin(t)->setBackend(nullptr);
                    // TODO: If output is static and length larger than new size, don't clear mem
                    TensorUtils::getDescribeOrigin(t)->mem = nullptr;
                }
            }
        }
        for (auto t : info.outputs) {
            TensorUtils::getDescribe(t)->stageMask &= (~Tensor::InsideDescribe::StageInfo::COMPUTE_SHAPE_STAGE);
        }
        bool compared = false;
        bool needCompute = !info.computeCache.match(info.inputs, compared);
        if (needCompute && compared) {
            // If not match, means the op's shape is mutable, close cache and don't compare
            info.computeCache.close(false);
        }
        if ((!skipShapeCompute) && needCompute) {
            auto res = SizeComputer::computeOutputSize(info.op, info.inputs, info.outputs);
            if (!res) {
                if (info.op->name() != nullptr) {
                    MNN_ERROR("Compute Shape Error for %s\n", info.op->name()->c_str());
                } else {
                    MNN_ERROR("Compute Shape Error for %d\n", info.op->type());
                }
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
            for (auto t: info.inputs) {
                TensorUtils::adjustTensorForCompability(t);
            }
            info.computeCache.insert(info.inputs);
            for (auto t : info.outputs) {
                TensorUtils::getDescribe(t)->rasterCommand.reset();
                TensorUtils::getDescribe(t)->stageMask |= Tensor::InsideDescribe::StageInfo::COMPUTE_SHAPE_STAGE;
                // The content may be computed by geometry computer, which will not make execution
                TensorUtils::getDescribe(t)->stageMask &= (~Tensor::InsideDescribe::StageInfo::CONTENT_NOT_CHANGE);
            }
        }
        info.computeCache.needComputeShape = needCompute;
        if (info.type != Schedule::CONSTANT) {
            continue;
        }
        if (!needCompute) {
            for (auto t : info.outputs) {
                TensorUtils::getDescribe(t)->stageMask |= Tensor::InsideDescribe::StageInfo::CONTENT_NOT_CHANGE;
            }
        }
        if (_hasZeroShapeOutput(info)) {
            continue;
        }
        // Skip geometry compute if no-needCompute
        if (needCompute) {
            cmdBufferVir.command.clear();
            cmdBufferVir.extras.clear();
            
            ctx.clear();
            auto geo = GeometryComputer::search(info.op->type(), Runtime::Compiler_Loop);
            {
                bool res = false;
                if (openCache) {
                    res = geo->onRecompute(info.op, info.inputs, info.outputs, geoContext, tempBuffer);
                }
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
                if (Tensor::InsideDescribe::MEMORY_VIRTUAL == TensorUtils::getDescribe(t)->memoryType) {
                    TensorUtils::getDescribe(t)->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
                }
            }
            for (auto& cp : cmdBufferVir.command) {
                auto& c = *cp;
                std::shared_ptr<BufferStorage> tmpStorge;
                if (nullptr == c.execution) {
                    auto exe = OpCommonUtils::createExecutionWithExternal(backupBackend.get(), c.inputs, c.outputs, c.op, external, tmpStorge);
                    c.execution.reset(exe);
                }
                auto exe = c.execution;
                if (nullptr == exe.get()) {
                    MNN_ERROR("Const Folder Error for %s\n", info.op->name()->c_str());
                    return NO_EXECUTION;
                }
                for (auto t : c.outputs) {
                    auto des = TensorUtils::getDescribeOrigin(t);
                    TensorUtils::setLinearLayout(t);
                    auto res = backupBackend->onAcquireBuffer(t, Backend::STATIC);
                    if (!res) {
                        return OUT_OF_MEMORY;
                    }
                    des->setBackend(backupBackend.get());
                }
                backupBackend->onResizeBegin();
                auto code = exe->onResize(c.inputs, c.outputs);
                if (NO_ERROR != code) {
                    return NOT_SUPPORT;
                }
                code = backupBackend->onResizeEnd();
                if (NO_ERROR != code) {
                    return NOT_SUPPORT;
                }
            }
        }
        for (auto& cp : cmdBufferVir.command) {
            auto& c = *cp;
            bool dirty = needCompute || c.op->type() == OpType_RandomNormal || c.op->type() == OpType_RandomUniform;
            if (!dirty) {
                for (auto t : c.inputs) {
                    auto des = TensorUtils::getDescribe(t);
                    if (!des->isMutable) {
                        continue;
                    }
                    if (des->group < 0) {
                        // From User Input, group = -1
                        dirty = true;
                        break;
                    }
                    if ((des->stageMask &                Tensor::InsideDescribe::StageInfo::CONTENT_NOT_CHANGE) == 0) {
                        dirty = true;
                        break;
                    }
                }
            }
            info.computeCache.needExecuteConst = dirty;
            if (dirty) {
                backupBackend->onExecuteBegin();
                auto code = cp->execution->onExecute(c.inputs, c.outputs);
                if (NO_ERROR != code) {
                    return NOT_SUPPORT;
                }
                backupBackend->onExecuteEnd();

                for (auto t : c.outputs) {
                    TensorUtils::getDescribe(t)->stageMask &= (~Tensor::InsideDescribe::StageInfo::CONTENT_NOT_CHANGE);
                }
            } else {
                for (auto t : c.outputs) {
                    TensorUtils::getDescribe(t)->stageMask |= Tensor::InsideDescribe::StageInfo::CONTENT_NOT_CHANGE;
                }
            }
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
        if ((!info.computeCache.needComputeShape) && (!tempBuffer.hasWrap)) {
            continue;
        }
        cmdBufferReal.command.clear();
        cmdBufferReal.extras.clear();
        if (_hasZeroShapeOutput(info)) {
            continue;
        }
        auto geo = GeometryComputer::search(info.op->type(), compileType);
        {
            bool res = false;
            if ((!tempBuffer.hasWrap) && openCache) {
                res = geo->onRecompute(info.op, info.inputs, info.outputs, geoContext, tempBuffer);
            }
            if (!res) {
                tempBuffer.command.clear();
                tempBuffer.extras.clear();
                res = geo->onCompute(info.op, info.inputs, info.outputs, geoContext, tempBuffer);
            }
            if (!res) {
                return NOT_SUPPORT;
            }
            tempBuffer.hasWrap = false;
            GeometryComputerUtils::makeRaster(tempBuffer, cmdBufferReal, geoContext);
            for (int v=0; v<info.outputs.size(); ++v) {
                auto t = info.outputs[v];
                auto des = TensorUtils::getDescribe(t);
                if (des->usage == Tensor::InsideDescribe::OUTPUT || des->usage == Tensor::InsideDescribe::TRAINABLE) {
                    // For output and trainable value, must directly compute the tensor
                    geoContext.getRasterCacheCreateRecursive(t, cmdBufferReal);
                    if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                        des->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
                    }
                }
            }
        }
    }

    
#ifdef MNN_BUILD_CODEGEN
    if(permitCodegen) {
        #ifdef LOG_VERPOSE
        MNN_PRINT("infos : [\n");
        for (auto info : infos) {
            auto& cmds = info.executeBuffer.command;
            for (auto cmd : cmds) {
                MNN_PRINT("\t%s", EnumNameOpType(cmd->op->type()));
                if(cmd->op->type() == OpType_BinaryOp) {
                    MNN_PRINT(" %d ", cmd->op->main_as_BinaryOp()->opType());
                }
                if(cmd->op->type() == OpType_UnaryOp) {
                    MNN_PRINT(" %d ", cmd->op->main_as_UnaryOp()->opType());
                }
                MNN_PRINT("\n");
            }
        }
        MNN_PRINT("]\n");
        MNN_PRINT("==================== opFuse ====================\n");
        #endif

        opFuse(infos, geoContext.forwardType(), geoContext.precisionType());

        #ifdef LOG_VERPOSE
        MNN_PRINT("infos : [\n");
        for (auto info : infos) {
            auto& cmds = info.executeBuffer.command;
            for (auto cmd : cmds) {
                MNN_PRINT("\t%s\n", EnumNameOpType(cmd->op->type()));
            }
        }
        MNN_PRINT("]\n");
        #endif
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
            if (!OpCommonUtils::opNeedContent(op, i)) {
                continue;
            }
            ctx.getRasterCacheCreateRecursive(cmd.inputs[i], dstBuffer);
        }
        dstBuffer.command.emplace_back(srcBuffer.command[index]);
    }
}
std::shared_ptr<Command> GeometryComputerUtils::makeBinary(int type, Tensor* input0, Tensor* input1, Tensor* output) {
    flatbuffers::FlatBufferBuilder builder(DEFAULT_ALLOCATE_SIZE);
    BinaryOpBuilder builder_(builder);
    builder_.add_opType(type);
    auto mainOffset = builder_.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_BinaryOp);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_BinaryOp);
    builder.Finish(opB.Finish());
    std::shared_ptr<Command> cmdP(new Command);
    auto& cmd = *cmdP;
    cmd.buffer.reset(new BufferStorage);
    cmd.buffer->storage = builder.ReleaseRaw(cmd.buffer->allocated_size, cmd.buffer->offset);
    cmd.inputs  = {input0, input1};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
    return cmdP;
}

std::shared_ptr<Command> GeometryComputerUtils::makeReduce(ReductionType type, Tensor* input0, Tensor* output) {
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
    std::shared_ptr<Command> cmdP(new Command);
    auto& cmd = *cmdP;
    cmd.buffer.reset(new BufferStorage);
    cmd.buffer->storage = builder.ReleaseRaw(cmd.buffer->allocated_size, cmd.buffer->offset);
    cmd.inputs  = {input0};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
    return cmdP;
}
std::shared_ptr<Command> GeometryComputerUtils::makeLayerNorm(Tensor* input0, Tensor* output, std::vector<int32_t> axis, float epsilon, std::vector<float> gamma, std::vector<float> beta, std::vector<int64_t> external, int group, bool useRMS) {
    flatbuffers::FlatBufferBuilder builder(DEFAULT_ALLOCATE_SIZE);
    std::vector<float> g, b;
    auto vecaxis = builder.CreateVector(axis);
    auto vecgamma = builder.CreateVector(g);
    auto vecbeta = builder.CreateVector(b);
    if (gamma.size() > 0 && beta.size() > 0) {
        vecgamma = builder.CreateVector(gamma.data(), gamma.size());
        vecbeta = builder.CreateVector(beta.data(), beta.size());
    }

    auto vecexternal = builder.CreateVector(external);
    LayerNormBuilder builder_(builder);
    builder_.add_axis(vecaxis);
    builder_.add_group(group);
    builder_.add_epsilon(epsilon);
    if (gamma.size() > 0 && beta.size() > 0) {
        builder_.add_gamma(vecgamma);
        builder_.add_beta(vecbeta);
    }
    
    builder_.add_useRMSNorm(useRMS);
    builder_.add_external(vecexternal);
    auto mainOffset = builder_.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_LayerNorm);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_LayerNorm);
    builder.Finish(opB.Finish());
    std::shared_ptr<Command> cmdP(new Command);
    auto& cmd = *cmdP;
    cmd.buffer.reset(new BufferStorage);
    cmd.buffer->storage = builder.ReleaseRaw(cmd.buffer->allocated_size, cmd.buffer->offset);
    cmd.inputs  = {input0};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
    return cmdP;
}
std::shared_ptr<Command> GeometryComputerUtils::makeUnary(UnaryOpOperation type, Tensor* input0, Tensor* output) {
    flatbuffers::FlatBufferBuilder builder(DEFAULT_ALLOCATE_SIZE);
    UnaryOpBuilder builder_(builder);
    builder_.add_opType(type);
    auto mainOffset = builder_.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_UnaryOp);
    opB.add_main(mainOffset);
    opB.add_main_type(OpParameter_UnaryOp);
    builder.Finish(opB.Finish());
    std::shared_ptr<Command> cmdP(new Command);
    auto& cmd = *cmdP;
    cmd.buffer.reset(new BufferStorage);
    cmd.buffer->storage = builder.ReleaseRaw(cmd.buffer->allocated_size, cmd.buffer->offset);
    cmd.inputs  = {input0};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
    return cmdP;
}
std::shared_ptr<Command> GeometryComputerUtils::makeCommand(flatbuffers::FlatBufferBuilder& builder, const std::vector<Tensor*>& inputs,
                                           const std::vector<Tensor*>& outputs) {
    std::shared_ptr<Command> cmdP(new Command);
    auto& cmd = *cmdP;
    cmd.buffer.reset(new BufferStorage);
    cmd.buffer->storage = builder.ReleaseRaw(cmd.buffer->allocated_size, cmd.buffer->offset);
    cmd.outputs = outputs;
    cmd.inputs  = inputs;
    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
    return cmdP;
}

std::shared_ptr<Command> GeometryComputerUtils::makeMatMul(Tensor* input0, Tensor* input1, Tensor* output, Tensor* Bias, bool transposeA,
                                          bool transposeB) {
    std::shared_ptr<Command> cmdP(new Command);
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
