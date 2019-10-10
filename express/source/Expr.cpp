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
#include "flatbuffers/util.h"
#include "MNN_generated.h"
#define MNN_OPEN_TIME_TRACE
#include "AutoTime.hpp"

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
    }
    flatbuffers::FlatBufferBuilder builder;
    auto offset = Op::Pack(builder, op);
    builder.Finish(offset);
    mExtraBuffer = new char[builder.GetSize()];
    ::memcpy(mExtraBuffer, builder.GetBufferPointer(), builder.GetSize());
    mOp = flatbuffers::GetMutableRoot<Op>(mExtraBuffer);
}

EXPRP Expr::create(std::unique_ptr<OpT>&& op, std::vector<VARP> inputs, int outputSize, std::shared_ptr<Executor> exe) {
    if (exe == nullptr && inputs.size() > 0) {
        exe = inputs[0]->expr().first->mExecutor;
    }
    if (exe == nullptr) {
        exe = std::shared_ptr<Executor>(new DefaultSolutionCreator);
    }
    EXPRP expr(new Expr(outputSize));
    expr->set(op.get());
    expr->mExecutor = exe;
    expr->mInputs   = inputs;
    for (auto v : inputs) {
        v->mTo.emplace_back(WeakEXPRP(expr));
    }
    return expr;
}
void Expr::render(NetT* dest) {
    if (nullptr == mOp || mOutputIndexes.size() > 0) {
        // Has rendered
        return;
    }
    std::unique_ptr<OpT> op(mOp->UnPack());
    op->name = mName;
    op->inputIndexes.resize(mInputs.size());
    for (int i = 0; i < mInputs.size(); ++i) {
        mInputs[i]->render(dest);
        op->inputIndexes[i] = mInputs[i]->mOutputIndex;
    }
    int outputIndex = (int)dest->tensorName.size();
    if (op->name.empty()) {
        op->name = EnumNameOpType(op->type) + numberToString(outputIndex);
    }
    MNN_ASSERT(mOutputSize >= 1);
    op->outputIndexes.resize(mOutputSize);
    if (mOutputSize == 1) {
        op->outputIndexes[0] = outputIndex;
        dest->tensorName.emplace_back(op->name);
    } else {
        for (int i = 0; i < mOutputSize; ++i) {
            op->outputIndexes[i] = outputIndex + i;
            dest->tensorName.emplace_back(op->name + "_" + numberToString(i));
        }
    }
    mOutputIndexes = op->outputIndexes;
    dest->oplists.emplace_back(std::move(op));
}

void Expr::setName(const std::string& name) {
    if (mName.empty()) {
        mName = name;
    }
}
Solution* Expr::inside() {
    if (mInside->mSolution == nullptr) {
        mInside->mSolution.reset(mExecutor->onCreate(mOp, (int)mInputs.size(), mOutputSize));
        mInside->mReq = mInside->mSolution->onGetRequirement();
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
    bool ready     = true;
    auto insidePtr = inside();
    mInside->mInputInfos.resize(mInputs.size());
    for (int i = 0; i < mInputs.size(); ++i) {
        if (nullptr == mInputs[i] || nullptr == mInputs[i]->mFrom) {
            // The Variable is set nullptr by api
            return false;
        }
        mInside->mInputInfos[i] = mInputs[i]->getInfo();
    }
    for (int i = 0; i < mInputs.size(); ++i) {
        auto& v  = mInputs[i];
        auto res = v->expr().first->requireInfo();
        if (mInside->mReq.shapeNeedContent[i]) {
            res = res && v->expr().first->requireCompute();
        }
        if (!res) {
            MNN_ERROR("Error for compute shape\n");
            ready = false;
            break;
        }
    }
    if (!ready) {
        return false;
    }
    auto res   = insidePtr->onComputeInfo(mInside->mInputInfos, mInside->mOutputInfos);
    mInfoDirty = false;
    return NO_ERROR == res;
}

bool Expr::requireCompute() {
    if (!mContentDirty) {
        return true;
    }
    auto solution = inside();
    for (int i = 0; i < mInputs.size(); ++i) {
        if (mInside->mReq.contentNeedContent[i]) {
            auto& input = mInputs[i];
            auto expr   = input->expr().first;
            auto res    = expr->requireCompute();
            if (!res) {
                return false;
            }
        }
    }
    if (!mAllocated) {
        bool res = requireAlloc();
        if (!res) {
            return false;
        }
    }
    auto code = solution->onComputeContent();
    // MNN_PRINT("Compute %s, %p\n", mName.c_str(), this);
    auto res = code == NO_ERROR;
    if (!res) {
        MNN_ERROR("Error for compute %s\n", mName.c_str());
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
            if (!res) {
                return false;
            }
        }
    }
    auto code = inside()->onAlloc(mInside->mInputInfos, mInside->mOutputInfos);
    if (NO_ERROR != code) {
        MNN_ERROR("Error for alloc, code = %d \n", code);
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
    dst->visitOutputs([](VARP var) {
        auto expr = var->mFrom;
        expr->mInside->mSolution.reset();
        expr->mInside->mInputInfos.clear();
        expr->mContentDirty = true;
        expr->mInfoDirty    = true;
        expr->mAllocated    = false;
        return false;
    });
}
void Expr::setInput(EXPRP expr, VARP src, int index) {
    MNN_ASSERT(expr->mInputs.size() > index && index >= 0);
    if (expr->mInputs[index].get() == src.get()) {
        return;
    }
    auto originVar = expr->mInputs[index];
    for (auto iter = originVar->mTo.begin(); iter != originVar->mTo.end(); iter++) {
        auto v = iter->lock();
        if (nullptr != v && v.get() == expr.get()) {
            originVar->mTo.erase(iter);
            break;
        }
    }
    expr->mInputs[index] = src;
    if (nullptr != src) {
        src->mTo.emplace_back(WeakEXPRP(expr));
    }
    expr->mInside->mSolution.reset();
    expr->mInside->mInputInfos.clear();
    expr->mContentDirty = true;
    expr->mInfoDirty    = true;
    expr->mAllocated    = false;
}

void Variable::setName(const std::string& name) {
    mName = name;
    mFrom->setName(name);
}
void Variable::render(NetT* dest) {
    if (mOutputIndex >= 0) {
        return;
    }
    mFrom->render(dest);
    mOutputIndex = mFrom->mOutputIndexes[mFromIndex];
    if (mName.size() > 0) {
        dest->tensorName[mOutputIndex] = mName;
    }
}
void Variable::clone(VARP dst, VARP src) {
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
    std::unique_ptr<OpT> inputOp(mFrom->get()->UnPack());
    inputOp->main.AsInput()->dims = dims;
    mFrom->set(inputOp.get());
    mFrom->mAllocated    = false;
    mFrom->mContentDirty = true;
    mFrom->mInfoDirty    = true;
    mFrom->mInside->mSolution.reset();
    mFrom->mInside->mInputInfos.clear();

    visitOutputs([](VARP var) { return var->mFrom->setInfoDirty(); });
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

void* Variable::writeInternal() {
    auto res = mFrom->requireAlloc();
    if (!res) {
        return nullptr;
    }
    visitOutputs([](VARP var) {
        auto expr        = var->mFrom;
        auto needRecurse = expr->setContentDirty();
        return needRecurse;
    });
    mFrom->mContentDirty = false;
    return mFrom->inside()->onMapContent(mFromIndex);
}

void Variable::unMap() {
    mFrom->inside()->onUnMapContent(mFromIndex);
}

void Variable::visitOutputs(const std::function<bool(VARP)>& visit) {
    for (auto iter = mTo.begin(); iter != mTo.end();) {
        auto expr = iter->lock();
        if (nullptr == expr) {
            iter = mTo.erase(iter);
            continue;
        }
        for (auto varIter = expr->mOutputs.begin(); varIter != expr->mOutputs.end();) {
            auto var = varIter->lock();
            if (nullptr == var) {
                varIter = expr->mOutputs.erase(varIter);
                continue;
            }
            bool recurse = visit(var);
            if (recurse) {
                var->visitOutputs(visit);
            }
            varIter++;
        }
        iter++;
    }
}
void Expr::setExecutor(std::shared_ptr<Executor> exe) {
    mExecutor          = exe;
    mInside->mSolution = nullptr;
}
bool Expr::setContentDirty() {
    if (mContentDirty) {
        return false;
    }
    mContentDirty = true;
    return true;
}
bool Expr::setInfoDirty() {
    if (mInfoDirty) {
        return false;
    }
    mInfoDirty    = true;
    mAllocated    = false;
    mContentDirty = true;
    return true;
}

std::vector<VARP> Variable::load(const char* fileName) {
    FileLoader loader(fileName);
    loader.read();
    if (!loader.valid()) {
        return {};
    }
    AutoStorage<uint8_t> buffer;
    loader.merge(buffer);
    if (buffer.get() == nullptr) {
        return {};
    }
    std::unique_ptr<NetT> source(UnPackNet(buffer.get()));
    if (nullptr == source) {
        return {};
    }
    AUTOTIME;
    if (source->oplists.empty()) {
        MNN_ERROR("Invalid net\n");
        return {};
    }
    // FUNC_PRINT(source->oplists.size());

    std::map<int, EXPRP> exprMaps;
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
        EXPRP expr = Expr::create(std::move(source->oplists[i]), inputs, (int)op->outputIndexes.size());

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
    auto variables = load(fileName);
    std::map<std::string, VARP> varMap;
    for (auto v : variables) {
        varMap[v->name()] = v;
    }
    return varMap;
}
void Variable::save(const std::vector<VARP>& vars, const char* fileName) {
    std::unique_ptr<NetT> net(new NetT);
    for (auto& output : vars) {
        output->render(net.get());
    }
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
    static size_t block = 4096;
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
Model Model::load(const char* fileName) {
    Model res;
    res.sequence = Variable::load(fileName);
    for (auto var : res.sequence) {
        if (var->expr().first->get()->type() == OpType_Input) {
            res.inputs.emplace_back(var);
        }
        if (var->linkNumber() == 0) {
            res.outputs.emplace_back(var);
        }
    }
    return res;
}
void Model::save(const char* fileName) const {
    Variable::save(outputs, fileName);
}
void Model::reorder() {
    AUTOTIME;
    sequence.clear();
    auto resetVisit = [](VARP var) {
        auto next = var->visited();
        var->setVisited(false);
        return next;
    };
    auto empty = [](VARP var) { return true; };
    for (auto output : outputs) {
        Variable::visit(output, resetVisit, empty);
    }
    for (auto output : outputs) {
        Variable::visit(
            output, [](VARP var) { return !var->visited(); },
            [this](VARP var) {
                sequence.emplace_back(var);
                var->setVisited(true);
                return true;
            });
    }
}

} // namespace Express
} // namespace MNN
