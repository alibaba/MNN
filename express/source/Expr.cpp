//
//  Expr.cpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define FLATBUFFERS_PREFER_PRINTF
#include "Expr.hpp"
#include <map>
#include "FileLoader.hpp"
#include "InsideExpr.hpp"
#include "Utils.hpp"
#include "flatbuffers/util.h"
#include "MNN_generated.h"
#define MNN_OPEN_TIME_TRACE
#include "AutoTime.hpp"

//#define MNN_EXPRESS_ERROR_REPORT
static inline std::string numberToString(int index) {
    return flatbuffers::NumToString(index);
}

namespace MNN {
namespace Express {

struct Expr::Inside {
    std::vector<std::shared_ptr<Variable::Info>> mOutputInfosContent;
    std::vector<const Variable::Info*> mInputInfos;
    std::vector<Variable::Info*> mOutputInfos;

    std::shared_ptr<Solution> mSolution;
    Solution::Requirement mReq;
};
Expr::Expr(int outputSize) : mOutputSize(outputSize) {
    mInside.reset(new Inside);
    mInside->mOutputInfos.resize(outputSize);
    mInside->mOutputInfosContent.resize(outputSize);
    for (int i = 0; i < mInside->mOutputInfosContent.size(); ++i) {
        mInside->mOutputInfosContent[i].reset(new Variable::Info);
        mInside->mOutputInfos[i] = mInside->mOutputInfosContent[i].get();
    }
}

Expr::~Expr() {
    if (nullptr != mExtraBuffer) {
        delete mExtraBuffer;
    }
    mInside.reset();
}
void Expr::set(const OpT* op) {
    MNN_ASSERT(nullptr != op);
    if (nullptr != mExtraBuffer) {
        delete mExtraBuffer;
        mExtraBuffer = nullptr;
    }
    flatbuffers::FlatBufferBuilder builder;
    auto offset = Op::Pack(builder, op);
    builder.Finish(offset);
    mExtraBuffer = new char[builder.GetSize()];
    ::memcpy(mExtraBuffer, builder.GetBufferPointer(), builder.GetSize());
    mOp = flatbuffers::GetMutableRoot<Op>(mExtraBuffer);
    mInside->mSolution.reset();
}

EXPRP Expr::create(const OpT* op, std::vector<VARP> inputs, int outputSize, std::shared_ptr<Executor> exe) {
    if (exe == nullptr && inputs.size() > 0) {
        exe = inputs[0]->expr().first->mExecutor;
    }
    if (exe == nullptr) {
        exe = std::shared_ptr<Executor>(new DefaultSolutionCreator);
    }
    EXPRP expr(new Expr(outputSize));
    expr->set(op);
    expr->mExecutor = exe;
    expr->mInputs   = inputs;
    for (int i=0; i<inputs.size(); ++i) {
        inputs[i]->mTo.emplace_back(std::make_pair(i, WeakEXPRP(expr)));
    }
    return expr;
}
void Expr::setName(const std::string& name) {
    mName = name;
}
Solution* Expr::inside() {
    if (mInside->mSolution == nullptr) {
        mInside->mSolution.reset(mExecutor->onCreate(mOp, (int)mInputs.size(), mOutputSize));
        if (nullptr != mInside->mSolution) {
            mInside->mReq = mInside->mSolution->onGetRequirement();
        }
    }
    return mInside->mSolution.get();
}
const Variable::Info* Expr::outputInfo(int index) const {
    return mInside->mOutputInfos[index];
}

bool Expr::requireInfo() {
    if (!mInfoDirty) {
        return true;
    }
    if (!mValid) {
        return false;
    }
    bool ready     = true;
    auto insidePtr = inside();
    if (nullptr == insidePtr) {
        mValid = false;
        return false;
    }
    mInside->mInputInfos.resize(mInputs.size());
    for (int i = 0; i < mInputs.size(); ++i) {
        if (nullptr == mInputs[i] || nullptr == mInputs[i]->mFrom) {
            // The Variable is set nullptr by api
            return false;
        }
        mInside->mInputInfos[i] = mInputs[i]->getInfo();
        if (nullptr == mInside->mInputInfos[i] && OpType_Concat != mOp->type()) {
#ifdef MNN_EXPRESS_ERROR_REPORT
            MNN_ERROR("%s, %d input not ready\n", mName.c_str(), i);
#endif
            mValid = false;
            return false;
        }
    }
    for (int i = 0; i < mInputs.size(); ++i) {
        auto& v  = mInputs[i];
        if (mInside->mReq.shapeNeedContent[i]) {
            auto res = v->expr().first->requireCompute();
            if (!res) {
#ifdef MNN_EXPRESS_ERROR_REPORT
                MNN_ERROR("%s, Error for compute shape %d\n", mName.c_str(), i);
#endif
                ready = false;
                mValid = false;
                break;
            }
        }
    }
    if (!ready) {
        return false;
    }
    //MNN_PRINT("Info %s, %p Start\n", mName.c_str(), this);
    auto res   = insidePtr->onComputeInfo(mInside->mInputInfos, mInside->mOutputInfos);
    //MNN_PRINT("Info Compute %s\n", mName.c_str());

    if (NO_ERROR == res) {
        mInfoDirty = false;
    } else {
        mValid = false;
    }
    return NO_ERROR == res;
}

bool Expr::requireCompute() {
    if ((!mContentDirty) && mValid) {
        return true;
    }
    if (!mValid) {
        return false;
    }
    //MNN_PRINT("Compute %s, %p Start\n", mName.c_str(), this);
    bool res = requireAlloc();
    if (!res) {
#ifdef MNN_EXPRESS_ERROR_REPORT
        MNN_ERROR("%s Alloc Error \n", mName.c_str());
#endif
        return false;
    }
    auto solution = inside();

    for (int i = 0; i < mInputs.size(); ++i) {
        if (mInside->mReq.contentNeedContent[i]) {
            auto& input = mInputs[i];
            auto expr   = input->expr().first;
            res    = expr->requireCompute();
            if (!res) {
#ifdef MNN_EXPRESS_ERROR_REPORT
                MNN_ERROR("%s compute input %d error , \n", mName.c_str(), i);
#endif
                if (!mInside->mReq.supportError[i]) {
                    mValid = false;
                    return false;
                }
            }
        }
    }
    auto code = solution->onComputeContent(mInside->mInputInfos, mInside->mOutputInfos);
    //MNN_PRINT("Compute %s, %p End\n", mName.c_str(), this);
    res = code == NO_ERROR;
    if (!res) {
#ifdef MNN_EXPRESS_ERROR_REPORT
        MNN_ERROR("Error for compute %s\n", mName.c_str());
#endif
        mValid = false;
        return false;
    }
    mContentDirty = false;
    return true;
}

bool Expr::requireAlloc() {
    if (mAllocated) {
        return true;
    }
    if (!requireInfo()) {
        return false;
    }
    for (int i = 0; i < mInputs.size(); ++i) {
        if (mInside->mReq.contentNeedContent[i]) {
            auto& input = mInputs[i];
            auto expr   = input->expr().first;
            auto res    = expr->requireAlloc();
            if ((!res) && (!mInside->mReq.supportError[i])) {
                mValid = false;
                return false;
            }
        }
    }
    auto code = inside()->onAlloc(mInside->mInputInfos, mInside->mOutputInfos);
    if (NO_ERROR != code) {
#ifdef MNN_EXPRESS_ERROR_REPORT
        MNN_ERROR("Error for alloc, code = %d \n", code);
#endif
        return false;
    }
    mAllocated = true;
    return true;
}

VARP Variable::create(EXPRP expr, int index) {
    VARP res(new Variable(expr, index));
    expr->mOutputs.emplace_back(WeakVARP(res));
    return res;
}
void Variable::setExpr(VARP dst, EXPRP from, int index) {
    if (from.get() == dst->mFrom.get() && index == dst->mFromIndex) {
        return;
    }
    if (from.get() != dst->mFrom.get()) {
        for (auto iter = dst->mFrom->mOutputs.begin(); iter != dst->mFrom->mOutputs.end(); iter++) {
            auto v = iter->lock();
            if (nullptr != v && v.get() == dst.get()) {
                dst->mFrom->mOutputs.erase(iter);
                break;
            }
        }
        dst->mFrom = from;
        if (nullptr != from) {
            from->mOutputs.emplace_back(WeakVARP(dst));
        }
    }
    dst->mFromIndex = index;
    std::set<Variable*> worked;
    dst->visitOutputs([&](VARP var, int index) {
        if (worked.find(var.get()) != worked.end()) {
            return false;
        }
        auto expr = var->mFrom;
        worked.insert(var.get());
        expr->mInside->mSolution.reset();
        expr->mInside->mInputInfos.clear();
        expr->mContentDirty = true;
        expr->mInfoDirty    = true;
        expr->mAllocated    = false;
        return true;
    });
}
void Expr::setInput(EXPRP expr, VARP src, int index) {
    MNN_ASSERT(expr->mInputs.size() > index && index >= 0);
    if (expr->mInputs[index].get() == src.get()) {
        return;
    }
    auto originVar = expr->mInputs[index];
    for (auto iter = originVar->mTo.begin(); iter != originVar->mTo.end(); iter++) {
        auto v = iter->second.lock();
        if (nullptr != v && v.get() == expr.get()) {
            originVar->mTo.erase(iter);
            break;
        }
    }
    expr->mInputs[index] = src;
    if (nullptr != src) {
        src->mTo.emplace_back(std::make_pair(index, WeakEXPRP(expr)));
    }
    expr->mInside->mSolution.reset();
    expr->mInside->mInputInfos.clear();
    expr->mContentDirty = true;
    expr->mInfoDirty    = true;
    expr->mAllocated    = false;
}

void Variable::setName(const std::string& name) {
    mName = name;
    if (mFrom->name().empty()) {
        mFrom->setName(name);
    }
}

bool Variable::input(VARP src) {
    if (mFrom->get()->type() != OpType_Input) {
        MNN_ERROR("Can't input to no-input op\n");
        return false;
    }
    if (nullptr == src) {
        /*Close the Input*/
        visitOutputs([](VARP var, int index) {
            auto recurse = var->mFrom->mValid; var->mFrom->mValid = false;
            return recurse;
        });
        mFrom->mValid = false;
        return false;
    }
    auto info = src->getInfo();
    std::shared_ptr<Variable::Info> tempInfo;
    bool needCopy = true;
    if (nullptr == info || 0 == info->size) {
        tempInfo.reset(new Variable::Info);
        tempInfo->type = halide_type_of<float>();
        info = tempInfo.get();
        needCopy = false;
    }
    auto dstInfo = getInfo();
    bool needChange = nullptr == dstInfo || info->order != dstInfo->order || info->dim.size() != dstInfo->dim.size();
    if (!needChange) {
        for (int i=0; i<info->dim.size(); ++i) {
            if (dstInfo->dim[i] != info->dim[i]) {
                needChange = true;
                break;
            }
        }
    }
    if (needChange) {
        std::unique_ptr<OpT> inputOp(mFrom->get()->UnPack());
        inputOp->main.AsInput()->dims = info->dim;
        inputOp->main.AsInput()->dtype = (MNN::DataType)Utils::convertDataType(info->type);
        inputOp->main.AsInput()->dformat = (MNN::MNN_DATA_FORMAT)Utils::convertFormat(info->order);
        mFrom->set(inputOp.get());
        mFrom->mAllocated    = false;
        mFrom->mContentDirty = true;
        mFrom->mInfoDirty    = true;
        mFrom->mValid = true;
        mFrom->mInside->mSolution.reset();
        mFrom->mInside->mInputInfos.clear();
    }
    if (needCopy) {
        auto dstPtr = writeInternal(false);
        auto srcPtr = src->readMap<void>();
        if (nullptr == dstPtr || nullptr == srcPtr) {
            MNN_ERROR("Alloc memory error or compute src error in Variable::Input\n");
            return false;
        }
        ::memcpy(dstPtr, srcPtr, info->size * info->type.bytes());
    }
    if (needChange) {
        visitOutputs([](VARP var, int index) { return var->mFrom->setInfoDirty(); });
    } else {
        informDirty();
    }
    return true;
}

void Variable::replace(VARP dst, VARP src) {
    if (nullptr == src) {
        Variable::setExpr(dst, nullptr, 0);
        return;
    }
    Variable::setExpr(dst, src->mFrom, src->mFromIndex);
}

const Variable::Info* Variable::getInfo() {
    if (nullptr == mFrom) {
        return nullptr;
    }
    auto res = mFrom->requireInfo();
    if (!res) {
        return nullptr;
    }
    return mFrom->mInside->mOutputInfos[mFromIndex];
}

bool Variable::resize(INTS dims) {
    if (mFrom->get()->type() != OpType_Input) {
        MNN_ERROR("Can't resize variable not from input\n");
        return false;
    }
    auto info = getInfo();
    if (nullptr != info && dims.size() == info->dim.size()) {
        bool theSame = true;
        for (int i=0; i<dims.size(); ++i) {
            if (info->dim[i] != dims[i]) {
                theSame = false;
                break;
            }
        }
        if (theSame) {
            return true;
        }
    }
    std::unique_ptr<OpT> inputOp(mFrom->get()->UnPack());
    inputOp->main.AsInput()->dims = dims;
    mFrom->set(inputOp.get());
    mFrom->mAllocated    = false;
    mFrom->mContentDirty = true;
    mFrom->mInfoDirty    = true;
    mFrom->mValid = true;
    mFrom->mInside->mInputInfos.clear();

    visitOutputs([](VARP var, int index) { return var->mFrom->setInfoDirty(); });
    return true;
}
void Variable::visit(VARP var, const std::function<bool(VARP)>& before, const std::function<bool(VARP)>& after) {
    bool next = before(var);
    if (!next) {
        return;
    }
    for (int i = 0; i < var->mFrom->inputs().size(); ++i) {
        visit(var->mFrom->inputs()[i], before, after);
    }
    after(var);
}

void* Variable::readInternal() {
    auto res = mFrom->requireCompute();
    if (!res) {
        return nullptr;
    }
    return mFrom->inside()->onMapContent(mFromIndex);
}

void Variable::informDirty() {
    visitOutputs([](VARP var, int index) {
        auto expr        = var->mFrom;
        auto needRecurse = expr->setContentDirty(index);
        return needRecurse;
    });
}

void* Variable::writeInternal(bool inform) {
    auto res = mFrom->requireAlloc();
    if (!res) {
        return nullptr;
    }
    if (inform) {
        informDirty();
    }
    mFrom->mContentDirty = false;
    return mFrom->inside()->onMapContent(mFromIndex);
}

void Variable::unMap() {
    mFrom->inside()->onUnMapContent(mFromIndex);
}

void Variable::visitOutputs(const std::function<bool(VARP, int)>& visit) {
    for (auto iter = mTo.begin(); iter != mTo.end();) {
        auto expr = iter->second.lock();
        if (nullptr == expr) {
            iter = mTo.erase(iter);
            continue;
        }
        bool recurse = false;
        for (auto varIter = expr->mOutputs.begin(); varIter != expr->mOutputs.end();) {
            auto var = varIter->lock();
            if (nullptr == var) {
                varIter = expr->mOutputs.erase(varIter);
                continue;
            }
            recurse = recurse || visit(var, iter->first);
            varIter++;
        }
        if (recurse) {
            for (auto varIter = expr->mOutputs.begin(); varIter != expr->mOutputs.end(); varIter++) {
                auto var = varIter->lock();
                var->visitOutputs(visit);
            }
        }
        iter++;
    }
}
void Expr::setExecutor(std::shared_ptr<Executor> exe) {
    mExecutor          = exe;
    mInside->mSolution = nullptr;
}
bool Expr::setContentDirty(int inputIndex) {
    if (mContentDirty) {
        return false;
    }
    if (nullptr != mInside) {
        if (mInside->mReq.shapeNeedContent[inputIndex]) {
            for (auto& w : mOutputs) {
                auto var = w.lock();
                if (nullptr != var) {
                    var->visitOutputs([](VARP var, int index) { return var->mFrom->setInfoDirty(); });
                }
            }
            return setInfoDirty();
        }
        if (!mInside->mReq.contentNeedContent[inputIndex]) {
            return false;
        }
    }
    mContentDirty = true;
    return true;
}
bool Expr::setInfoDirty() {
    if (mInfoDirty && mValid) {
        //MNN_PRINT("End Info Dirty for %s\n", mName.c_str());
        return false;
    }
    //MNN_PRINT("Set Info Dirty for %s\n", mName.c_str());
    mInfoDirty    = true;
    mAllocated    = false;
    mContentDirty = true;
    mValid = true;
    return true;
}

std::vector<VARP> Variable::load(const char* fileName) {
    AUTOTIME;
    FileLoader loader(fileName);
    if (!loader.valid()) {
        MNN_ERROR("Error for open %s\n", fileName);
        return {};
    }
    loader.read();
    if (!loader.valid()) {
        return {};
    }
    AutoStorage<uint8_t> buffer;
    loader.merge(buffer);
    if (buffer.get() == nullptr) {
        return {};
    }
    flatbuffers::Verifier verify((const uint8_t*)(buffer.get()), buffer.size());
    if (false == VerifyNetBuffer(verify)) {
        MNN_PRINT("Invalidate buffer to create variable\n");
        return {};
    }
    std::unique_ptr<NetT> source(UnPackNet(buffer.get()));
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

        for (int index = 0; index < op->outputIndexes.size(); ++index) {
            auto outputIndex = op->outputIndexes[index];
            if (variableMap.find(outputIndex) == variableMap.end()) {
                auto newVariable = Variable::create(expr, index);
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
void Variable::save(const std::vector<VARP>& vars, NetT* dest) {
    auto executeOrder = getExecuteOrder(vars);
    dest->tensorName.resize(executeOrder.size());
    std::map<std::pair<EXPRP, int>, int> varIndex;
    for (int i=0; i<executeOrder.size(); ++i) {
        varIndex[executeOrder[i]->expr()] = i;
    }
    for (int index = 0; index < executeOrder.size(); ++index) {
        auto v = executeOrder[index];
        auto expr = v->expr();
        std::shared_ptr<void> _defer(nullptr, [&](void*) {
            if (!v->name().empty()) {
                dest->tensorName[index] = v->name();
                return;
            }
            auto name = v->expr().first->name();
            if (v->expr().second != 0) {
                name = name  + "_" + numberToString(v->expr().second);
            }
            dest->tensorName[index] = name;
        });
        if (expr.first->visited()) {
            continue;
        }
        auto mOp = expr.first->get();
        std::unique_ptr<OpT> op(mOp->UnPack());
        op->name = expr.first->name();
        op->inputIndexes.resize(expr.first->inputs().size());
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            op->inputIndexes[i] = varIndex[expr.first->inputs()[i]->expr()];
        }
        int outputIndex = (int)dest->tensorName.size();
        if (op->name.empty()) {
            op->name = EnumNameOpType(op->type) + numberToString(outputIndex);
        }
        op->outputIndexes.resize(expr.first->outputSize());
        auto exprOutputs = expr.first->outputs();
        for (auto outputVar : exprOutputs) {
            auto out = outputVar.lock();
            if (nullptr == out) {
                continue;
            }
            op->outputIndexes[out->mFromIndex] = varIndex[out->expr()];
        }
        dest->oplists.emplace_back(std::move(op));
        expr.first->setVisited(true);
    }
    for (int index = 0; index < executeOrder.size(); ++index) {
        executeOrder[index]->expr().first->setVisited(false);
    }

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
        if (var->expr().first->get()->type() == OpType_Input) {
            res.first[var->name()] = var;
        }
        if (var->linkNumber() == 0) {
            res.second[var->name()] = var;
        }
    }
    return res;
}

std::vector<VARP> Variable::getExecuteOrder(const std::vector<VARP>& outputs) {
    std::vector<VARP> sequence;
    for (auto output : outputs) {
        Variable::visit(
                        output, [](VARP var) { return !var->expr().first->visited(); },
                        [&sequence](VARP var) {
                            //FUNC_PRINT_ALL(var->name().c_str(), s);
                            for (auto v : var->expr().first->outputs()) {
                                auto sharedV = v.lock();
                                if (nullptr != sharedV) {
                                    sequence.emplace_back(sharedV);
                                }
                            }
                            var->expr().first->setVisited(true);
                            return true;
                        });
    }
    for (auto var : sequence) {
        var->expr().first->setVisited(false);
    }
    return sequence;
}

} // namespace Express
} // namespace MNN
