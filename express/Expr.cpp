//
//  Expr.cpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define FLATBUFFERS_PREFER_PRINTF
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "Utils.hpp"
#include "RuntimeAttr.hpp"
#include "core/FileLoader.hpp"
#include "core/TensorUtils.hpp"
#include "core/WrapExecution.hpp"
#include "utils/InitNet.hpp"
//#define MNN_OPEN_TIME_TRACE
#include "MNN/AutoTime.hpp"
#include "MNN/expr/ExecutorScope.hpp"
#include "half.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"

//#define MNN_EXPRESS_ERROR_REPORT
static inline std::string numberToString(int index) {
    char s[10];
    snprintf(s, 10, "%d", index);
    return std::string(s);
}

static bool HasUnknownDim(const std::vector<int>& dims) {
    for (const int& dim : dims) {
        if (dim < 0) {
            return true;
        }
    }
    return false;
}

namespace MNN {
namespace Express {
void Variable::Info::syncSize() {
    size = 1;
    for (int i=0; i<dim.size(); ++i) {
        if (dim[i] <= 0) {
            // Not valid
            size = 0;
            return;
        }
        if (order == NC4HW4 && i == 1) {
            size *= (UP_DIV(dim[1], 4) * 4);
        } else {
            size *= dim[i];
        }
    }
}

bool VARP::fix(VARP::InputType type) const {
    if (nullptr == mContent->expr().first->get()) {
        mContent->expr().first->mType = type;
        return true;
    }
    auto info = mContent->getInfo();
    if (nullptr == info) {
        return false;
    }
    auto exprInfo    = mContent->expr();
    auto inside      = exprInfo.first->inside();
    auto mFrom = exprInfo.first;
    auto cache = mFrom->inside()->mCache;
    if (nullptr == cache) {
        ExecutorScope::Current()->makeCache({mFrom}, false);
        cache = mFrom->inside()->mCache;
    }
    if (nullptr == cache) {
        return false;
    }
    if (NO_ERROR != cache->compute()) {
        return false;
    }
    auto inputTensor = inside->mCache->getSession()->getTensor(inside->mCacheOffset + exprInfo.second);
    auto tensor = Tensor::clone(inputTensor);
    VARP newVARP = Express::Variable::create(Express::Expr::create(tensor, true));
    newVARP->expr().first->mType = type;
    auto& pipelineInfo = inside->mCache->getSession()->getPipelineInfo(0);
    if (TensorUtils::getDescribeOrigin(tensor)->getBackend() == pipelineInfo.first.cache.first.get()) {
        newVARP->expr().first->inside()->mHoldBackend = pipelineInfo.first.cache.first;
    } else if (TensorUtils::getDescribeOrigin(tensor)->getBackend() == pipelineInfo.first.cache.second.get()) {
        newVARP->expr().first->inside()->mHoldBackend = pipelineInfo.first.cache.second;
    }
    Variable::replace(VARP(mContent), newVARP);
    return true;
}

Expr::Expr(int outputSize) {
    mInside.reset(new Inside(outputSize));
    mOutputNames.resize(outputSize);
}
Expr::Expr(Tensor* tensor, bool own) {
    mInside.reset(new Inside(tensor, own));
    mOutputNames.resize(1);
}

Expr::~Expr() {
    mInside.reset();
}
Variable::Info* Expr::outputInfo(int index) const {
    return mInside->mOutputInfos.data() + index;
}

void Expr::_addLinkForInputs(EXPRP expr) {
    auto inputs = expr->inputs();
    for (int i=0; i<inputs.size(); ++i) {
        if (inputs[i].get() == nullptr) {
            continue;
        }
        bool findEmpty = false;
        auto inputExpr = inputs[i]->mFrom;
        for (int j=0; j<inputExpr->mTo.size(); ++j) {
            auto ref = inputExpr->mTo[j].lock();
            if (nullptr == ref) {
                inputExpr->mTo[j] = WeakEXPRP(expr);
                findEmpty = true;
                break;
            }
        }
        if (!findEmpty) {
            inputExpr->mTo.emplace_back(WeakEXPRP(expr));
        }
    }
}
EXPRP Expr::create(Tensor* tensor, bool own) {
    EXPRP expr(new Expr(tensor, own));
    expr->mOp = nullptr;
    expr->mType = VARP::CONSTANT;
    auto& dstInfo = expr->mInside->mOutputInfos[0];
    expr->mInside->mInfoDirty = false;
    expr->mInside->mContentDirty = false;
    return expr;
}

EXPRP Expr::create(Variable::Info&& info, const void* ptr, VARP::InputType type, Expr::MemoryType memtype) {
    EXPRP expr(new Expr(1));
    expr->mOp = nullptr;
    auto originPtr = ptr;
    expr->mInside->mOutputInfos[0] = std::move(info);
    auto& dstInfo = expr->mInside->mOutputInfos[0];
    expr->mInside->mInfoDirty = false;
    dstInfo.syncSize();
    Utils::copyInfoToTensor(expr->mInside->mOutputTensors[0], expr->mInside->mOutputInfos.data());
    expr->mType = type;
    if (type == VARP::CONSTANT) {
        TensorUtils::getDescribe(expr->mInside->mOutputTensors[0])->usage = Tensor::InsideDescribe::CONSTANT;
        TensorUtils::getDescribe(expr->mInside->mOutputTensors[0])->isMutable = false;
    } else if (type == VARP::INPUT) {
        TensorUtils::getDescribe(expr->mInside->mOutputTensors[0])->usage = Tensor::InsideDescribe::INPUT;
    } else {
        // VARP::TRAINABLE
        TensorUtils::getDescribe(expr->mInside->mOutputTensors[0])->usage = Tensor::InsideDescribe::TRAINABLE;
    }
    if (dstInfo.size > 0 && memtype == COPY) {
        auto res = Utils::allocMemoryForHostTensor(expr->mInside->mOutputTensors[0]);
        if (!res) {
            MNN_ASSERT(false);
            return nullptr;
        }
    } else {
        expr->mInside->mOutputTensors[0]->buffer().host = nullptr;
    }
    if (nullptr == originPtr) {
        if (type == VARP::INPUT && dstInfo.size > 0) {
            expr->mInside->mContentDirty = true;
        }
        return expr;
    }
    expr->mInside->mContentDirty = false;
    if (memtype == COPY) {
        size_t total_size = dstInfo.size;
        total_size *= dstInfo.type.bytes();
        ::memcpy(expr->mInside->mOutputTensors[0]->buffer().host, originPtr, total_size);
    } else {
        expr->mInside->mOutputTensors[0]->buffer().host = (uint8_t*)originPtr;
        if (memtype == REF) {
            TensorUtils::getDescribe(expr->mInside->mOutputTensors[0])->memoryType = Tensor::InsideDescribe::MEMORY_OUTSIDE;
        }
    }
    return expr;
}
EXPRP Expr::create(std::shared_ptr<BufferStorage> extra, std::vector<VARP>&& inputs, int outputSize) {
    EXPRP expr(new Expr(outputSize));
    expr->mStorage = extra;
    expr->mOp = flatbuffers::GetRoot<Op>(extra->buffer());
    switch (expr->mOp->type()) {
        case OpType_Const:
            expr->mType = VARP::CONSTANT;
            break;
        case OpType_TrainableParam:
            expr->mType = VARP::TRAINABLE;
            break;
        default:
            expr->mType = VARP::INPUT;
            break;
    }
    expr->mInputs   = std::move(inputs);
    auto exe = ExecutorScope::Current();
    expr->mInside->mReq = exe->getRequirement(expr.get());
    if ((!(exe->getLazyMode() & Executor::LAZY_COMPUTE_ONCE)) && exe->lazyEval) {
        _addLinkForInputs(expr);
    }
    return expr;
}

EXPRP Expr::create(const OpT* op, std::vector<VARP> inputs, int outputSize) {
    if (OpType_Input == op->type) {
        Variable::Info info;
        info.dim = op->main.AsInput()->dims;
        if (info.dim.size() >= 1 && -1 == info.dim[0]) {
            info.dim[0] = 1;
        }
        info.order = Utils::revertFormat(op->main.AsInput()->dformat);
        info.type = Utils::revertDataType(op->main.AsInput()->dtype);
        return create(std::move(info), nullptr, VARP::INPUT);
    }
    if (OpType_Const == op->type || OpType_TrainableParam == op->type) {
        if (!op->externalPath.empty()) {
            flatbuffers::FlatBufferBuilder builder;
            auto offset = Op::Pack(builder, op);
            builder.Finish(offset);
            std::shared_ptr<BufferStorage> extra(new BufferStorage);
            extra->storage = builder.ReleaseRaw(extra->allocated_size, extra->offset);
            auto resExpr = Expr::create(extra, std::move(inputs), outputSize);
            resExpr->setName(op->name);
            return resExpr;
        }
        Variable::Info info;
        info.dim = op->main.AsBlob()->dims;
        info.order = Utils::revertFormat(op->main.AsBlob()->dataFormat);
        void* ptr = nullptr;
        info.type = Utils::revertDataType(op->main.AsBlob()->dataType);
        info.syncSize();
        switch (op->main.AsBlob()->dataType) {
            case DataType_DT_INT8:
                ptr = (void*)op->main.AsBlob()->int8s.data();
                break;
            case DataType_DT_INT32:
                ptr = (void*)op->main.AsBlob()->int32s.data();
                break;
            case DataType_DT_UINT8:
                ptr = (void*)op->main.AsBlob()->uint8s.data();
                break;
            case DataType_DT_FLOAT:
                ptr = (void*)op->main.AsBlob()->float32s.data();
                break;
            case DataType_DT_BFLOAT16:
                ptr = (void*)op->main.AsBlob()->uint8s.data();
                break;
            default:
                break;
        }
        Expr::MemoryType memtype = Expr::MemoryType::COPY;
        if (op->main.AsBlob()->dataType == DataType_DT_HALF) {
            auto src = (half_float::half*)op->main.AsBlob()->uint8s.data();
            ptr = MNNMemoryAllocAlign(info.size * sizeof(float), MNN_MEMORY_ALIGN_DEFAULT);
            if (nullptr == src || nullptr == ptr) {
                EXPRP empty;
                return empty;
            }
            auto outputPtr = (float*)ptr;
            for (int i=0; i<info.size; ++i) {
                outputPtr[i] = src[i];
            }
            memtype = Expr::MemoryType::MOVE;
        }
        //MNN_ASSERT(nullptr != ptr);
        auto expr = create(std::move(info), ptr, VARP::CONSTANT, memtype);
        if (OpType_TrainableParam == op->type && nullptr != ptr) {
            expr->mType = VARP::TRAINABLE;
        }
        return expr;
    }
    flatbuffers::FlatBufferBuilder builder;
    auto offset = Op::Pack(builder, op);
    builder.Finish(offset);
    std::shared_ptr<BufferStorage> extra(new BufferStorage);
    extra->storage = builder.ReleaseRaw(extra->allocated_size, extra->offset);
    auto resExpr = Expr::create(extra, std::move(inputs), outputSize);
    resExpr->setName(op->name);
    return resExpr;
}
void Expr::setName(const std::string& name) {
    mName = name;
}
bool Expr::requireInfo() {
    if (!mInside->mInfoDirty) {
        return true;
    }
    if (!mValid) {
        return false;
    }
    if (nullptr == mOp) {
        return !HasUnknownDim(mInside->mOutputInfos[0].dim);
    }
    if (!mCanDecompose) {
        return true;
    }
    bool ready     = true;
    for (int i = 0; i < mInputs.size(); ++i) {
        if (nullptr == mInputs[i] || nullptr == mInputs[i]->mFrom) {
            // The Variable is set nullptr by api
            return false;
        }
        auto inputInfo = mInputs[i]->getInfo();
        if (nullptr == inputInfo) {
#ifdef MNN_EXPRESS_ERROR_REPORT
            MNN_ERROR("%s, %d input not ready\n", mName.c_str(), i);
#endif
            mValid = false;
            return false;
        }
    }
    for (int i = 0; i < mInputs.size(); ++i) {
        auto& v  = mInputs[i];
        if (v->getInfo()->size == 0) {
            // zero shape
            continue;
        }
        if (mInside->mReq.shapeNeedContent[i]) {
            // For shape need content, the content must not be nullptr
            auto ptr = v->readInternal(true);
            if (nullptr == ptr) {
                ready = false;
                break;
            }
        }
    }
    if (!ready) {
        return false;
    }
    //MNN_PRINT("Info %s, %p Start\n", mName.c_str(), this);
    auto res   = ExecutorScope::Current()->computeInfo(this);
    //MNN_PRINT("Info Compute %s\n", mName.c_str());

    if (NO_ERROR == res) {
        mInside->mInfoDirty = false;
    } else {
        mValid = false;
    }
    return NO_ERROR == res;
}

size_t Variable::linkNumber() const {
    return mFrom->outputs().size();
}
const std::vector<WeakEXPRP>& Variable::toExprs() const {
    return mFrom->outputs();
}

VARP Variable::create(EXPRP expr, int index) {
    VARP res(new Variable(expr, index));
#ifdef MNN_EXPR_SHAPE_EAGER
    auto info = expr->requireInfo();
    if (!info) {
#ifdef MNN_EXPRESS_ERROR_REPORT
        MNN_ERROR("Can't compute shape\n");
#endif
    }
#endif
    auto executor = ExecutorScope::Current();
    if (!executor->lazyEval) {
        res.fix(VARP::CONSTANT);
        return res;
    }
    // CONTENT Mode, Use Geometry Computer to Decompress Expr
    do {
        if (!(executor->getLazyMode() & Executor::LAZY_CONTENT)) {
            break;
        }
        if (expr->get() == nullptr) {
            break;
        }
        if (!expr->mCanDecompose) {
            break;
        }
        bool res = expr->requireInfo();
        if (!res) {
            break;
        }
        std::map<Tensor*, VARP> varMap;
        std::vector<Tensor*> inputTensors(expr->mInputs.size());
        std::vector<Tensor*> outputTensors(expr->outputSize());
        for (int i=0; i<inputTensors.size(); ++i) {
            inputTensors[i] = Utils::getTensor(expr->mInputs[i]);
            varMap.insert(std::make_pair(inputTensors[i], expr->mInputs[i]));
        }
        for (int i=0; i<outputTensors.size(); ++i) {
            outputTensors[i] = expr->mInside->mOutputTensors[i];
        }
        auto bn = executor->getAttr()->constantBackend;
        // TODO: Support set mask
        GeometryComputer::Context context(Interpreter::GeometryComputeMask::GEOMETRCOMPUTEMASK_ALL, bn);
        auto geo = GeometryComputer::search(expr->get()->type(), Runtime::Compiler_Loop);
        CommandBuffer cmd;
        res = geo->onCompute(expr->get(), inputTensors, outputTensors, context, cmd);
        if (!res) {
            break;
        }
        for (int i=0; i<outputTensors.size(); ++i) {
            // Avoid release from host tensor, the memory is owned by executor's cpu runtime
            if (TensorUtils::getDescribe(outputTensors[i])->usage == Tensor::InsideDescribe::CONSTANT) {
                TensorUtils::getDescribe(outputTensors[i])->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
            }
        }
        if (TensorUtils::getDescribe(outputTensors[index])->usage == Tensor::InsideDescribe::CONSTANT) {
            auto constExpr = Expr::create(Tensor::clone(outputTensors[index]), true);
            return Variable::create(constExpr);
        }
        // TODO: For multi-output expr, reduce dup compute
        CommandBuffer cmdDst;
        GeometryComputerUtils::makeRaster(cmd, cmdDst, context);
        for (auto t : outputTensors) {
            context.getRasterCacheCreateRecursive(t, cmdDst);
        }
        // Make New Exprs
        for (int cmdIndex=0; cmdIndex < cmdDst.command.size(); ++cmdIndex) {
            auto& cmd = cmdDst.command[cmdIndex];
            std::vector<VARP> cmdInputs(cmd->inputs.size());
            for (int i=0; i<cmd->inputs.size(); ++i) {
                auto iter = varMap.find(cmd->inputs[i]);
                if (iter == varMap.end()) {
                    // Extract Const Value
                    auto constExpr = Expr::create(Tensor::clone(cmd->inputs[i]), true);
                    VARP constVar(new Variable(constExpr, 0));
                    varMap.insert(std::make_pair(cmd->inputs[i], constVar));
                    cmdInputs[i] = constVar;
                } else {
                    cmdInputs[i] = iter->second;
                }
            }
            EXPRP currentExpr;
            if (cmd->op->type() == OpType_Raster) {
                // Rebuild raster buffer
                auto cmdTensor = cmd->outputs[0];
                auto cmdDes = TensorUtils::getDescribe(cmdTensor);
                MNN_ASSERT(cmd->inputs.size() == cmdDes->regions.size());
                std::vector<int> regions(cmdDes->regions.size() * 11);
                for (int j=0; j<cmdDes->regions.size(); ++j) {
                    auto& srcReg = cmdDes->regions[j];
                    auto dstInt = regions.data() + 11 * j;
                    dstInt[0] = srcReg.src.offset;
                    ::memcpy(dstInt + 1, srcReg.src.stride, 3 * sizeof(int));
                    dstInt[4] = srcReg.dst.offset;
                    ::memcpy(dstInt + 5, srcReg.dst.stride, 3 * sizeof(int));
                    ::memcpy(dstInt + 8, srcReg.size, 3 * sizeof(int));
                }
                auto cmdExpr = Utils::makeRaster(cmdInputs, regions, cmdTensor->shape(), cmdTensor->getType(), TensorUtils::getDescribe(cmdTensor)->dimensionFormat);
                cmdExpr->mCanDecompose = false;
                VARP cmdVar(new Variable(cmdExpr, 0));
                varMap.insert(std::make_pair(cmdTensor, cmdVar));
                currentExpr = cmdVar->mFrom;
            } else {
                EXPRP cmdExpr;
                if (cmd->op == expr->get()) {
                    cmdExpr = Expr::create(expr->mStorage, std::move(cmdInputs), cmd->outputs.size());
                } else {
                    cmdExpr = Expr::create(cmd->buffer, std::move(cmdInputs), cmd->outputs.size());
                }
                currentExpr = cmdExpr;
                cmdExpr->mCanDecompose = false;
                for (int j=0; j<cmd->outputs.size(); ++j) {
                    VARP cmdVar(new Variable(cmdExpr, j));
                    varMap.insert(std::make_pair(cmd->outputs[j], cmdVar));
                }
            }
            for (int j=0; j<cmd->outputs.size(); ++j) {
                Utils::copyTensorToInfo(currentExpr->inside()->mOutputInfos.data() + j, cmd->outputs[j]);
                TensorUtils::copyShape(cmd->outputs[j], currentExpr->inside()->mOutputTensors[j], true, true);
            }
        }
        return varMap.find(expr->inside()->mOutputTensors[index])->second;
    } while (false);
    return res;
}
void Expr::replace(EXPRP old, EXPRP from) {
    if (old.get() == from.get()) {
        return;
    }
    for (auto input : old->inputs()) {
        if (input.get() == nullptr) {
            continue;
        }
        for (int j=0; j<input->mFrom->mTo.size(); ++j) {
            auto ref = input->mFrom->mTo[j].lock();
            if (ref.get() == old.get()) {
                input->mFrom->mTo[j].reset();
            }
        }
    }
    for (auto input : from->inputs()) {
        if (input.get() == nullptr) {
            continue;
        }
        bool hasSet = false;
        for (int j=0; j<input->mFrom->mTo.size(); ++j) {
            auto ref = input->mFrom->mTo[j].lock();
            if (ref.get() == old.get()) {
                hasSet = true;
                break;
            }
        }
        if (!hasSet) {
            for (int j=0; j<input->mFrom->mTo.size(); ++j) {
                auto ref = input->mFrom->mTo[j].lock();
                if (nullptr == ref) {
                    input->mFrom->mTo[j] = WeakEXPRP(old);
                    hasSet = true;
                    break;
                }
            }
        }
        if (!hasSet) {
            input->mFrom->mTo.emplace_back(WeakEXPRP(old));
        }
    }
    old->mCanDecompose = from->mCanDecompose;
    old->mOp = from->mOp;
    old->mName = from->mName;
    old->mOutputNames = from->mOutputNames;
    old->mStorage = from->mStorage;
    old->mType = from->mType;
    old->mValid = from->mValid;
    old->mInside = from->mInside;
    old->mInputs = from->mInputs;
    std::vector<Expr*> visited;
    old->visitOutputs([&](EXPRP expr, int index) {
        if (expr->visited()) {
            return false;
        }
        visited.emplace_back(expr.get());
        expr->setVisited(true);
        expr->mInside->mCache.reset();
        expr->mInside->mCacheOffset = 0;
        expr->mValid = true;
        expr->mInside->mInfoDirty = true;
        return true;
    });
    for (auto e : visited) {
        e->setVisited(false);
    }
}

void Variable::setName(const std::string& name) {
    mFrom->mOutputNames[mFromIndex] = name;
    if (mFrom->name().empty()) {
        mFrom->setName(name);
    }
}

bool Variable::setDevicePtr(const void* devicePtr, int memoryType) {
    if (nullptr != mFrom->get()) {
        MNN_ERROR("Can't setDevicePtr to no-input op\n");
        return false;
    }
    informDirty();
    MNN_ASSERT(TensorUtils::getDescribe(mFrom->inside()->mOutputTensors[0])->quantAttr == nullptr || TensorUtils::getDescribe(mFrom->inside()->mOutputTensors[0])->type == DataType_DT_FLOAT);
    mFrom->mInside->mContentDirty = false;
    // Clear host address, Don't malloc hostPtr afterwards
    Utils::releaseMemoryForHostTensor(mFrom->inside()->mOutputTensors[0]);
    return mFrom->inside()->mOutputTensors[0]->setDevicePtr(devicePtr, memoryType);
}

bool Variable::copyToDevicePtr(void* devicePtr, int memoryType) {
    if (nullptr != mFrom->get()) {
        MNN_ERROR("Can't copyToDevicePtr to no-input op\n");
        return false;
    }
    
    auto inside = mFrom->inside();
    auto originTensor = inside->mOutputTensors[mFromIndex];
    
    auto bn = TensorUtils::getDescribeOrigin(originTensor)->getBackend();
    if(bn == nullptr) {
        MNN_ERROR("Error: Varp copyToDevicePtr can't find backend\n");
        return false;
    }

    MNN::Tensor tempTensor(originTensor->dimensions(), originTensor->getDimensionType());
    tempTensor.setDevicePtr(devicePtr, memoryType);
    
    TensorUtils::getDescribeOrigin(originTensor)->getBackend()->onCopyBuffer(originTensor, &tempTensor);
    // Sync the result
    tempTensor.wait(Tensor::MAP_TENSOR_READ, true);
    return true;
}

const std::string& Variable::name() const {
    return mFrom->outputName(mFromIndex);
}
const Tensor* Variable::getTensor() const {
    auto inside      = mFrom->inside();
    auto inputTensor = inside->mOutputTensors[mFromIndex];
    if (nullptr != inside->mCache) {
        inputTensor = inside->mCache->getSession()->getTensor(inside->mCacheOffset + mFromIndex);
    }
    return inputTensor;
}
bool Variable::input(VARP src) {
    if (nullptr != mFrom->get()) {
        MNN_ERROR("Can't input to no-input op\n");
        return false;
    }
    if (nullptr == src) {
        /*Close the Input*/
        mFrom->visitOutputs([](EXPRP expr, int index) {
            auto recurse = expr->mValid; expr->mValid = false;
            return recurse;
        });
        mFrom->mValid = false;
        return false;
    }
    auto info = src->getInfo();
    std::shared_ptr<Variable::Info> tempInfo;
    if (nullptr == info) {
        tempInfo.reset(new Variable::Info);
        tempInfo->size = 0;
        tempInfo->type = halide_type_of<float>();
        info = tempInfo.get();
    }
    auto dstInfo = getInfo();
    bool needChange = nullptr == dstInfo || info->order != dstInfo->order || info->dim.size() != dstInfo->dim.size() || info->type != dstInfo->type;
    if (!needChange) {
        for (int i=0; i<info->dim.size(); ++i) {
            if (dstInfo->dim[i] != info->dim[i]) {
                needChange = true;
                break;
            }
        }
    }

    if (!mFrom->mInside->mCache) {
        ExecutorScope::Current()->makeCache({mFrom}, false);
    }
    if (needChange) {
        mFrom->mInside->mOutputInfos[0] = *info;
        Utils::releaseMemoryForHostTensor(mFrom->inside()->mOutputTensors[0]);
        Utils::copyInfoToTensor(mFrom->inside()->mOutputTensors[0], mFrom->inside()->mOutputInfos.data());
        Utils::allocMemoryForHostTensor(mFrom->inside()->mOutputTensors[0]);
    }
    if (info->size) {
        auto dstPtr = writeInternal(false);
        auto srcPtr = src->readMap<void>();
        if (nullptr == dstPtr || nullptr == srcPtr) {
            //MNN_ERROR("Alloc memory error or compute src error in Variable::Input\n");
            return false;
        }
        ::memcpy(dstPtr, srcPtr, info->size * info->type.bytes());
    }
    if (needChange) {
        mFrom->visitOutputs([](EXPRP expr, int index) { return expr->setInfoDirty(); });
    } else {
        informDirty();
    }
    mFrom->mInside->mContentDirty = false;
    return true;
}

void Variable::replace(VARP dst, VARP src) {
    if (nullptr == src) {
        dst->setExpr(nullptr, 0);
        return;
    }
    if (nullptr == dst) {
        dst.mContent = src.mContent;
        return;
    }
    if (src->mFrom.get() == dst->mFrom.get()) {
        dst->mFromIndex = src->mFromIndex;
        return;
    }
    if (src->mFrom->outputSize() != dst->mFrom->outputSize()) {
        // Can't replace Expr, Just replace VARP
        std::vector<Expr*> visited;
        dst->mFrom->visitOutputs([src, dst, &visited](EXPRP expr, int index) {
            if (expr->visited()) {
                return false;
            }
            expr->setVisited(true);
            visited.emplace_back(expr.get());
            expr->mInside->mCache.reset();
            expr->mInside->mCacheOffset = 0;
            expr->mValid = true;
            expr->mInside->mInfoDirty = true;
            expr->mInside->mContentDirty = true;
            return true;
        });
        for (auto v : visited) {
            v->setVisited(false);
        }
        dst->mFrom->visitOutputs([src, dst](EXPRP expr, int index) {
            for (int i =0; i< expr->inputs().size(); ++i) {
                auto input = expr->inputs()[i];
                if (input == dst) {
                    expr->mInputs[i] = src;
                }
            }
            src->mFrom->mTo.emplace_back(expr);
            return false;
        });

        dst->mFrom = src->mFrom;
        dst->mFromIndex = src->mFromIndex;
        return;
    }
    Expr::replace(dst->mFrom, src->mFrom);
    dst->mFromIndex = src->mFromIndex;
}

const Variable::Info* Variable::getInfo() {
    if (nullptr == mFrom) {
        return nullptr;
    }
    auto res = mFrom->requireInfo();
    if (!res) {
        return nullptr;
    }
    return mFrom->mInside->mOutputInfos.data() + mFromIndex;
}

bool Variable::resize(INTS dims) {
    if (nullptr != mFrom->get() && VARP::INPUT != mFrom->mType) {
        MNN_ERROR("Can't resize variable not from input\n");
        return false;
    }
    auto& info = mFrom->mInside->mOutputInfos[0];
    if (dims.size() == info.dim.size()) {
        bool theSame = true;
        for (int i=0; i<dims.size(); ++i) {
            if (info.dim[i] != dims[i]) {
                theSame = false;
                break;
            }
        }
        if (theSame) {
            return true;
        }
    }
    info.dim = dims;
    info.syncSize();
    Utils::copyInfoToTensor(mFrom->inside()->mOutputTensors[0], mFrom->inside()->mOutputInfos.data());
    Utils::releaseMemoryForHostTensor(mFrom->inside()->mOutputTensors[0]);
    if (0 < info.size) {
        bool res = Utils::allocMemoryForHostTensor(mFrom->inside()->mOutputTensors[0]);
        if (!res) {
            return false;
        }
    }

    mFrom->mValid = true;
    mFrom->inside()->mInfoDirty = false;
    mFrom->inside()->mContentDirty = true;
    mFrom->visitOutputs([](EXPRP expr, int index) { return expr->setInfoDirty(); });
    return true;
}
void Expr::visit(EXPRP expr, const std::function<bool(EXPRP)>& before, const std::function<bool(EXPRP)>& after) {
    bool next = before(expr);
    if (!next) {
        return;
    }
    for (int i = 0; i < expr->inputs().size(); ++i) {
        if (expr->inputs()[i].get() == nullptr) {
            continue;
        }
        visit(expr->inputs()[i]->mFrom, before, after);
    }
    after(expr);
}

void* Variable::readInternal(bool forShape) {
    if (nullptr == mFrom->get()) {
        if (VARP::INPUT == mFrom->mType) {
            if (mFrom->mInside->mContentDirty) {
                return nullptr;
            }
        }
        //MNN_ASSERT(nullptr != mFrom->inside()->mOutputTensors[0]->buffer().host);
        auto inside = mFrom->inside();
        auto originTensor = inside->mOutputTensors[mFromIndex];
        auto des = TensorUtils::getDescribe(originTensor);
        if (WrapExecution::needWrap(originTensor, nullptr) || (des->quantAttr != nullptr && des->type == DataType_DT_INT8)) {
            // For StaticModule will other-device runtime, we may create Variable with other-device's memory
            // The case won't occurred for varibale = INPUT
            // Need Copy
            if (nullptr != inside->mHostTensor) {
                // The Varp will not be created as input, so we just need copy once
                return inside->mHostTensor->host<void>();
            }
            inside->mHostTensor = new Tensor;
            TensorUtils::copyShape(originTensor, inside->mHostTensor, true);
            inside->mHostTensor->buffer().type = originTensor->getType();
            inside->mHostTensor->buffer().host = (uint8_t*)MNNMemoryAllocAlign(inside->mHostTensor->size(), MNN_MEMORY_ALIGN_DEFAULT);
            TensorUtils::getDescribe(inside->mHostTensor)->memoryType = Tensor::InsideDescribe::MEMORY_HOST;
            originTensor->copyToHostTensor(inside->mHostTensor);
            return inside->mHostTensor->host<void>();
        }
        return originTensor->buffer().host;
    }
    auto res = mFrom->requireInfo();
    if (false == res) {
        return nullptr;
    }
    auto cache = mFrom->inside()->mCache;
    if (nullptr == cache) {
        ExecutorScope::Current()->makeCache({mFrom}, forShape);
        cache = mFrom->inside()->mCache;
    }
    if (nullptr == cache) {
        return nullptr;
    }
    if (NO_ERROR != cache->compute()) {
        return nullptr;
    }
    return cache->mapOutput(mFrom->mInside->mCacheOffset + mFromIndex, mFrom->mInside->mOutputTensors[mFromIndex]);
}


void Variable::informDirty() {
    std::vector<Expr*> visited;
    mFrom->visitOutputs([&visited](EXPRP expr, int index) {
        if (expr->visited()) {
            return false;
        }
        visited.emplace_back(expr.get());
        expr->setVisited(true);
        if (expr->inside()->mReq.shapeNeedContent.empty()) {
            // Not init
            return false;
        }
        if (expr->inside()->mReq.shapeNeedContent[index]) {
            expr->setInfoDirty();
            expr->visitOutputs([](EXPRP e, int index) { return e->setInfoDirty(); });
            return false;
        }
        if (expr->inside()->mReq.contentNeedContent[index]) {
            if (expr->inside()->mCache != nullptr) {
                expr->inside()->mCache->setContentDirty();
            }
            return true;
        }
        return false;
    });
    for (auto e : visited) {
        e->setVisited(false);
    }
}
void Variable::prepareCompute(const std::vector<VARP>& vars, bool forceCpu) {
    std::vector<EXPRP> exprs;
    for (auto v : vars) {
        if (nullptr != v && nullptr != v->mFrom->get()) {
            if (!v->expr().first->visited() && nullptr == v->expr().first->inside()->mCache) {
                v->expr().first->requireInfo();
                v->expr().first->setVisited(true);
                exprs.emplace_back(v->expr().first);
            }
        }
    }
    for (auto v : vars) {
        if (nullptr != v && nullptr != v->mFrom->get()) {
            v->expr().first->setVisited(false);
        }
    }
    ExecutorScope::Current()->makeCache(std::move(exprs), forceCpu);
}

void Variable::compute(const std::vector<VARP>& vars, bool forceCPU) {
    prepareCompute(vars, forceCPU);
    for (auto& v : vars) {
        if (nullptr != v && nullptr != v->mFrom->get()) {
            auto inside = v->mFrom->inside();
            if (nullptr != inside && nullptr != inside->mCache) {
                inside->mCache->compute();
            }
        }
    }
}

void* Variable::writeInternal(bool inform) {
    if (nullptr != mFrom->get()) {
        return nullptr;
    }
    if (inform) {
        informDirty();
    }
    MNN_ASSERT(TensorUtils::getDescribe(mFrom->inside()->mOutputTensors[0])->quantAttr == nullptr || TensorUtils::getDescribe(mFrom->inside()->mOutputTensors[0])->type == DataType_DT_FLOAT);
    mFrom->mInside->mContentDirty = false;
    return mFrom->inside()->mOutputTensors[0]->host<void>();
}

void Variable::writeScaleInternal(float scaleValue, float zeroPoint, bool inform) {
    MNN_ASSERT(TensorUtils::getDescribe(mFrom->inside()->mOutputTensors[0])->quantAttr == nullptr || TensorUtils::getDescribe(mFrom->inside()->mOutputTensors[0])->type == DataType_DT_FLOAT);
    if (inform) {
       informDirty();
    }
    mFrom->mInside->mContentDirty = true;
    TensorUtils::getDescribe(mFrom->inside()->mOutputTensors[0])->quantAttr.reset(new QuantAttr);
    auto quant = TensorUtils::getDescribe(mFrom->inside()->mOutputTensors[0])->quantAttr.get();
    quant->scale = scaleValue;
    quant->zero = zeroPoint;
}

void Variable::unMap() {
    //mFrom->inside()->onUnMapContent(mFromIndex);
}

void Expr::visitOutputs(const std::function<bool(EXPRP, int)>& visit) {
    for (auto iter = mTo.begin(); iter != mTo.end();) {
        auto expr = iter->lock();
        if (nullptr == expr) {
            iter = mTo.erase(iter);
            continue;
        }
        bool recurse = false;
        auto inputs = expr->inputs();
        for (int i=0; i<inputs.size(); ++i) {
            if (inputs[i].get() == nullptr) {
                continue;
            }
            if (inputs[i]->mFrom.get() == this) {
                recurse = recurse || visit(expr, i);
            }
        }
        if (recurse) {
            expr->visitOutputs(visit);
        }
        iter++;
    }
}
bool Expr::setInfoDirty() {
    if (mInside->mInfoDirty && mValid) {
        //MNN_PRINT("End Info Dirty for %s\n", mName.c_str());
        return false;
    }
    //MNN_PRINT("Set Info Dirty for %s\n", mName.c_str());
    mInside->mInfoDirty    = true;
    mInside->mContentDirty = true;
    mValid = true;
    if (mInside->mCache != nullptr) {
        mInside->mCache->setShapeDirty();
    }
    for (auto o : mInside->mOutputTensors) {
        Utils::releaseMemoryForHostTensor(o);
    }
    return true;
}

std::vector<VARP> Variable::load(const char* fileName) {
    AutoStorage<uint8_t> buffer;
    {
        FileLoader loader(fileName, true);
        if (!loader.valid()) {
            MNN_ERROR("Error for open %s\n", fileName);
            return {};
        }
        loader.read();
        if (!loader.valid()) {
            return {};
        }
        loader.merge(buffer);
        if (buffer.get() == nullptr) {
            return {};
        }
    }
    return load(buffer.get(), buffer.size());
}
std::vector<VARP> Variable::load(const uint8_t* buffer, size_t length) {
    AUTOTIME;
    flatbuffers::Verifier verify((const uint8_t*)(buffer), length);
    if (false == VerifyNetBuffer(verify)) {
        MNN_PRINT("Invalidate buffer to create variable\n");
        return {};
    }
    std::unique_ptr<NetT> source(UnPackNet(buffer));
    if (nullptr == source) {
        return {};
    }
    if (source->oplists.empty()) {
        MNN_ERROR("Invalid net\n");
        return {};
    }
    // FUNC_PRINT(source->oplists.size());

    auto opSize      = source->oplists.size();
    auto tensorCount = source->tensorName.size();
    if (tensorCount == 0) {
        tensorCount = source->tensorNumber;
    }
    std::vector<VARP> variable;
    variable.reserve(tensorCount);
    std::map<int, VARP> variableMap;
    bool isStatic = source->usage == Usage_INFERENCE_STATIC;
    std::vector<std::shared_ptr<Tensor>> allTensors;
    if (isStatic) {
        allTensors.resize(source->tensorName.size());
        initTensors(allTensors, flatbuffers::GetRoot<MNN::Net>(buffer));
    }

    // Generate All Exprs by order of net
    for (int i = 0; i < opSize; ++i) {
        std::vector<VARP> inputs;
        auto op = source->oplists[i].get();
        for (int index = 0; index < op->inputIndexes.size(); ++index) {
            auto inputIndex = op->inputIndexes[index];
            if (variableMap.find(inputIndex) == variableMap.end()) {
                MNN_ERROR("Can't find variable for %s, the graph is error\n", op->name.c_str());
                break;
            }
            inputs.emplace_back(variableMap[inputIndex]);
        }
        EXPRP expr = Expr::create(source->oplists[i].get(), inputs, (int)op->outputIndexes.size());
        expr->setName(source->oplists[i]->name);
        if (isStatic && nullptr != expr->get()) {
            // Set tensor shape from net
            expr->mCanDecompose = false;
            for (int index = 0; index < op->outputIndexes.size(); ++index) {
                auto outputIndex = op->outputIndexes[index];
                delete expr->inside()->mOutputTensors[index];
                expr->inside()->mOutputTensors[index] = Tensor::clone(allTensors[outputIndex].get());
                Utils::copyTensorToInfo(expr->inside()->mOutputInfos.data() + index, expr->inside()->mOutputTensors[index]);
            }
        }

        for (int index = 0; index < op->outputIndexes.size(); ++index) {
            auto outputIndex = op->outputIndexes[index];
            if (variableMap.find(outputIndex) == variableMap.end()) {
                // just create VARP and don't compute
                VARP newVariable(new Variable(expr, index));
                if (source->tensorName.size() > outputIndex) {
                    newVariable->setName(source->tensorName[outputIndex]);
                }
                variableMap[outputIndex] = newVariable;
                variable.emplace_back(newVariable);
            }
        }
    }
    return variable;
}

std::map<std::string, VARP> Variable::loadMap(const uint8_t* buffer, size_t length) {
    AUTOTIME;
    auto variables = load(buffer, length);
    std::map<std::string, VARP> varMap;
    for (auto v : variables) {
        varMap[v->name()] = v;
    }
    return varMap;
}

std::map<std::string, VARP> Variable::loadMap(const char* fileName) {
    AUTOTIME;
    auto variables = load(fileName);
    std::map<std::string, VARP> varMap;
    for (auto v : variables) {
        varMap[v->name()] = v;
    }
    return varMap;
}
std::vector<VARP> Variable::mapToSequence(const std::map<std::string, VARP>& source) {
    std::vector<VARP> outputs;
    outputs.reserve(source.size());
    for (auto& iter : source) {
        outputs.emplace_back(iter.second);
    }
    return outputs;
}
#define SET_TYPE(TYPE, type) \
if (tensor->getType() == halide_type_of<type##_t>()) {\
blob->dataType = DataType_DT_##TYPE;

void Variable::save(const std::vector<VARP>& vars, NetT* dest) {
    auto executeOrder = getExecuteOrder(vars);
    // Search subgraphs
    std::map<std::string, std::shared_ptr<Executor::SubGraph>> subgraphs;
    auto exe = ExecutorScope::Current();
    for (int index = 0; index < executeOrder.size(); ++index) {
        auto expr = executeOrder[index];
        auto op = expr->get();
        if (nullptr == op || op->type() != OpType_While) {
            continue;
        }
        if (op->main_type() != OpParameter_WhileParam) {
            continue;
        }
        auto whileParam = op->main_as_WhileParam();
        auto name = whileParam->body_graph()->str();
        auto subgraph = exe->findSubGraph(name);
        if (nullptr == subgraph) {
#ifdef MNN_EXPRESS_ERROR_REPORT
            MNN_ERROR("Variable::save: Invalid subgraph name: %s\n", name.c_str());
#endif
            continue;
        }
        MNN_ASSERT(subgraph->depends.size() == 0);
        subgraphs.insert(std::make_pair(name, subgraph));
    }
    // Save Subgraphs
    dest->subgraphs.clear();
    for (auto& graphIter : subgraphs) {
        // Copy Subgraph info
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::SubGraphProto::Pack(builder, graphIter.second->info.get()));
        std::unique_ptr<MNN::SubGraphProtoT> subgraph(flatbuffers::GetRoot<MNN::SubGraphProto>(builder.GetBufferPointer())->UnPack());
        dest->subgraphs.emplace_back(std::move(subgraph));
    }

    // Get Expr - TensorOffset Map
    std::map<EXPRP, int> varIndexInfo;
    {
        int tensorOffset = 0;
        for (int i=0; i<executeOrder.size(); ++i) {
            auto expr = executeOrder[i];
            auto outputSize = executeOrder[i]->outputSize();
            varIndexInfo[expr] = tensorOffset;
            tensorOffset += outputSize;
        }
        dest->tensorName.resize(tensorOffset);
    }

    // Create All Op
    for (int index = 0; index < executeOrder.size(); ++index) {
        auto expr = executeOrder[index];
        auto mOp = expr->get();
        std::unique_ptr<OpT> op;
        if (nullptr != mOp) {
            op.reset(mOp->UnPack());
        } else {
            MNN_ASSERT(1 == expr->outputSize());
            auto& info = expr->mInside->mOutputInfos[0];
            const void* ptr = expr->mInside->mOutputTensors[0]->host<void>();
            VARP temp;
            if (nullptr == ptr || expr->mInside->mOutputTensors[0]->deviceId() > 0) {
                temp = Variable::create(expr);
                ptr = temp->readMap<void>();
            }
            op.reset(new OpT);
            if (expr->mType != VARP::INPUT) {
                auto blob        = new BlobT;
                blob->dataFormat = (MNN_DATA_FORMAT)Utils::convertFormat(info.order);
                blob->dims       = info.dim;
                if (info.type.code == halide_type_float) {
                    blob->dataType = DataType_DT_FLOAT;
                    blob->float32s.resize(info.size);
                    ::memcpy(blob->float32s.data(), ptr, info.size * sizeof(float));
                } else if (info.type.code == halide_type_int && info.type.bits == 32) {
                    blob->dataType = DataType_DT_INT32;
                    blob->int32s.resize(info.size);
                    ::memcpy(blob->int32s.data(), ptr, info.size * sizeof(int));
                } else if (info.type.code == halide_type_int && info.type.bits == 8) {
                    blob->dataType = DataType_DT_INT8;
                    blob->int8s.resize(info.size);
                    ::memcpy(blob->int8s.data(), ptr, info.size * sizeof(int8_t));
                } else if (info.type.code == halide_type_uint && info.type.bits == 8) {
                    blob->dataType = DataType_DT_UINT8;
                    blob->uint8s.resize(info.size);
                    ::memcpy(blob->uint8s.data(), ptr, info.size * sizeof(uint8_t));
                } else if (info.type.code == halide_type_bfloat && info.type.bits == 16) {
                    blob->dataType = DataType_DT_BFLOAT16;
                    blob->uint8s.resize(info.size * 2);
                    ::memcpy(blob->uint8s.data(), ptr, info.size * sizeof(int16_t));
                }
                op->type       = OpType_Const;
                if (expr->mType == VARP::TRAINABLE) {
                    op->type = OpType_TrainableParam;
                }
                op->main.type  = OpParameter_Blob;
                op->main.value = blob;
            } else {
                op->type                    = OpType_Input;
                op->main.type               = OpParameter_Input;
                op->main.value              = new InputT;
                op->main.AsInput()->dtype   = (MNN::DataType)Utils::convertDataType(info.type);
                MNN_ASSERT(op->main.AsInput()->dtype != DataType_DT_INVALID);
                op->main.AsInput()->dims    = info.dim;
                op->main.AsInput()->dformat = (MNN_DATA_FORMAT)Utils::convertFormat(info.order);
            }
        }
        if (!expr->name().empty()) {
            op->name = expr->name();
        }
        op->inputIndexes.resize(expr->inputs().size());
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            if (expr->inputs()[i] == nullptr) {
                op->inputIndexes[i] = -1;
                continue;
            }
            auto inputExpr = expr->inputs()[i]->expr();
            op->inputIndexes[i] = varIndexInfo[inputExpr.first] + inputExpr.second;
        }
        if (op->name.empty()) {
            op->name = EnumNameOpType(op->type) + numberToString(index+1);
        }
        op->outputIndexes.resize(expr->outputSize());
        auto tensorIndexOffset = varIndexInfo[expr];
        for (int v=0; v<expr->outputSize(); ++v) {
            op->outputIndexes[v] = tensorIndexOffset + v;
            dest->tensorName[tensorIndexOffset+v] = expr->outputName(v);
        }
        dest->oplists.emplace_back(std::move(op));
    }
    bool staticModel = ExecutorScope::Current()->getLazyMode() == Executor::LAZY_CONTENT;

    // Fill Empty Tensor Name With Default Op Name
    for (int index = 0; index < executeOrder.size(); ++index) {
        auto expr = executeOrder[index];
        auto op = dest->oplists[index].get();
        auto tensorIndexOffset = varIndexInfo[expr];
        for (int v=0; v<expr->outputSize(); ++v) {
            auto subindex = tensorIndexOffset + v;
            if (dest->tensorName[subindex].empty()) {
                if (v == 0) {
                    dest->tensorName[subindex] = op->name;
                } else {
                    dest->tensorName[subindex] = op->name + numberToString(v);
                }
            }
            auto tensor = expr->inside()->mOutputTensors[v];

            if (staticModel || TensorUtils::getDescribe(tensor)->quantAttr) {
                auto des = TensorUtils::getDescribe(tensor);
                auto describe = std::unique_ptr<MNN::TensorDescribeT>(new MNN::TensorDescribeT);
                describe->index = varIndexInfo[expr] + v;
                describe->name = dest->tensorName[subindex];

                auto tensorDes = TensorUtils::getDescribe(tensor);
                if (nullptr != tensorDes->quantAttr) {
                    describe->quantInfo.reset(new TensorQuantInfoT);
                    describe->quantInfo->max = tensorDes->quantAttr->max;
                    describe->quantInfo->min = tensorDes->quantAttr->min;
                    describe->quantInfo->zero = tensorDes->quantAttr->zero;
                    describe->quantInfo->scale = tensorDes->quantAttr->scale;
                }
                if (staticModel) {
                    describe->blob = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
                    auto& blob = describe->blob;
                    blob->dataFormat = des->dimensionFormat;
                    if (tensor->getType() == halide_type_of<float>()) {
                    blob->dataType = DataType_DT_FLOAT;
                    } else {
                        SET_TYPE(INT8, int8)}
                        SET_TYPE(UINT8, uint8)}
                        SET_TYPE(INT32, int32)}
                        SET_TYPE(INT64, int64)}
                    }
                    for (int d = 0; d < tensor->dimensions();d++) {
                        describe->blob->dims.push_back(tensor->buffer().dim[d].extent);
                    }
                    for (auto& reg : des->regions) {
                        auto regionT = std::unique_ptr<MNN::RegionT>(new MNN::RegionT);
                        regionT->src = std::unique_ptr<MNN::ViewT>(new MNN::ViewT);
                        regionT->dst = std::unique_ptr<MNN::ViewT>(new MNN::ViewT);
                        regionT->src->offset = reg.src.offset;
                        regionT->dst->offset = reg.dst.offset;
                        for (int s = 0; s < 3; s++) {
                            regionT->src->stride.push_back(reg.src.stride[s]);
                            regionT->dst->stride.push_back(reg.dst.stride[s]);
                            regionT->size.push_back(reg.size[s]);
                        }
                        describe->regions.emplace_back(std::move(regionT));
                    }
                }
                dest->extraTensorDescribe.emplace_back(std::move(describe));
            }
        }
    }
    if (staticModel) {
        dest->usage = Usage_INFERENCE_STATIC;
    }
    // add version number
    dest->extraInfo.reset(new ExtraInfoT);
    dest->extraInfo->version = MNN_VERSION;
}
std::vector<int8_t> Variable::save(const std::vector<VARP>& vars) {
    std::unique_ptr<NetT> net(new NetT);
    save(vars, net.get());
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = Net::Pack(builder, net.get());
    builder.Finish(offset);
    std::vector<int8_t> result(builder.GetSize());
    ::memcpy(result.data(), builder.GetBufferPointer(), builder.GetSize());
    return result;
}

void Variable::save(const std::vector<VARP>& vars, const char* fileName) {
    std::unique_ptr<NetT> net(new NetT);
    save(vars, net.get());
    // FUNC_PRINT(net->oplists.size());
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = Net::Pack(builder, net.get());
    builder.Finish(offset);
    // TODO, use FileWriter instead
    FILE* f = fopen(fileName, "wb");
    if (nullptr == f) {
        MNN_ERROR("Open %s error\n", fileName);
        return;
    }
    static const size_t block = 4096;
    size_t totalSize    = builder.GetSize();
    size_t blockSize    = UP_DIV(totalSize, block);
    for (size_t i = 0; i < blockSize; ++i) {
        size_t sta = block * i;
        size_t fin = std::min(sta + block, totalSize);
        if (fin > sta) {
            auto realSize = fwrite((const char*)builder.GetBufferPointer() + sta, 1, fin - sta, f);
            if (realSize != fin - sta) {
                MNN_ERROR("Write %s error\n", fileName);
            }
        }
    }
    fclose(f);
}
std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> Variable::getInputAndOutput(const std::map<std::string, VARP>& allVariable) {
    std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> res;
    for (auto& iter : allVariable) {
        auto var = iter.second;
        if (var->expr().first->get() == nullptr && var->expr().first->mType == VARP::INPUT) {
            res.first[var->name()] = var;
        }
        if (var->linkNumber() == 0) {
            res.second[var->name()] = var;
        }
    }
    return res;
}

std::vector<EXPRP> Variable::getExecuteOrder(const std::vector<VARP>& outputs) {
    std::vector<EXPRP> sequence;
    for (auto output : outputs) {
        Expr::visit(
                        output->mFrom, [](EXPRP expr) { return !expr->visited(); },
                        [&sequence](EXPRP expr) {
                            //FUNC_PRINT_ALL(var->name().c_str(), s);
                            if (!expr->visited()) {
                                sequence.emplace_back(expr);
                                expr->setVisited(true);
                            }
                            return true;
                        });
    }
    for (auto expr : sequence) {
        expr->setVisited(false);
    }
    return sequence;
}

VARP VARP::operator+(VARP var) const {
    return _Add(VARP(mContent), var);
}
VARP VARP::operator-(VARP var) const {
    return _Subtract(VARP(mContent), var);
}
VARP VARP::operator*(VARP var) const {
    return _Multiply(VARP(mContent), var);
}
VARP VARP::operator/(VARP var) const {
    return _Divide(VARP(mContent), var);
}
VARP VARP::mean(INTS dims) const {
    return _ReduceMean(VARP(mContent), dims);
}
VARP VARP::sum(INTS dims) const {
    return _ReduceSum(VARP(mContent), dims);
}

} // namespace Express
} // namespace MNN
