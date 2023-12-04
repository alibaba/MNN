//
//  LoopGrad.cpp
//  MNN
//
//  Created by MNN on b'2022/10/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class LoopGrad : public OpGrad {
public:
    static std::vector<VARP> _getGradExpr(EXPRP expr, std::vector<VARP> inputs, VARP tempOutput) {
        auto gradValue = OpGrad::get(expr->get()->type())->onGrad(expr, {tempOutput});
        MNN_ASSERT(gradValue.size() == inputs.size());
        return gradValue;
    }
    static void _gradForCommandMatMul(const RegionCommandT* command, const std::map<int, int>& backwardMap, int tensorNumber, std::map<int, VARP>& extraInputs, std::map<int, VARP>& extraOutputs, std::vector<std::unique_ptr<RegionCommandT>>& dstCommands) {
        auto AIndex = command->indexes[1];
        auto BIndex = command->indexes[2];
        auto CIndex = command->indexes[0];
        auto transA = command->op->main.AsMatMul()->transposeA;
        auto transB = command->op->main.AsMatMul()->transposeB;
        int e = command->size[0];
        int l = command->size[1];
        int h = command->size[2];

        if (backwardMap.find(CIndex) == backwardMap.end()) {
            return;
        }
        std::map<int, int> originIndexMap;
        for (int i=0; i<command->indexes.size(); ++i) {
            auto index = command->indexes[i];
            originIndexMap.insert(std::make_pair(index, i));
            auto bkIter = backwardMap.find(index);
            if (bkIter != backwardMap.end()) {
                originIndexMap.insert(std::make_pair(bkIter->second, i));
            }
        }
        auto CDiffIndex = backwardMap.find(CIndex)->second;
        if (backwardMap.find(AIndex) != backwardMap.end()) {
            auto ADiffIndex = backwardMap.find(AIndex)->second;
            // Compute A Diff
            std::unique_ptr<RegionCommandT> currentCommand(new RegionCommandT);
            currentCommand->op.reset(new OpT);
            currentCommand->op->type = OpType_MatMul;
            currentCommand->op->main.value = new MatMulT;
            currentCommand->op->main.type = OpParameter_MatMul;
            currentCommand->indexes = {ADiffIndex, CDiffIndex, BIndex};
            currentCommand->view.resize(3);
            for (int j=0; j<currentCommand->view.size(); ++j) {
                currentCommand->view[j].reset(new ViewT);
            }
            currentCommand->iterIndexes.resize(3);
            currentCommand->steps.resize(3);
            for (int j=0; j<currentCommand->indexes.size(); ++j) {
                // Compute output info
                auto index = currentCommand->indexes[j];
                currentCommand->iterIndexes[j] = command->iterIndexes[originIndexMap[index]];
                currentCommand->steps[j] = command->steps[originIndexMap[index]];
                *currentCommand->view[j] = *command->view[originIndexMap[index]];
            }
            // TODO: Optimize fuse option
            currentCommand->fuse = BinaryOpOperation_ADD;

            // Reorder the view's stride by size order change to e, h, l
            std::vector<int> order = {0, 2, 1};
            currentCommand->size = {e, h, l};
            for (int j=0; j<currentCommand->indexes.size(); ++j) {
                auto view = currentCommand->view[j].get();
                auto originStride = view->stride;
                for (int k=0; k<3; ++k) {
                    view->stride[k]  = originStride[order[k]];
                }
            }
            // Set Transpose Info by stride
            auto dstMatMulParam = currentCommand->op->main.AsMatMul();
            {
                auto transAView = currentCommand->view[1].get();
                auto transBView = currentCommand->view[1].get();
                dstMatMulParam->transposeA = transAView->stride[1] != 1;
                dstMatMulParam->transposeB = transBView->stride[1] == 1;
            }
            dstCommands.emplace_back(std::move(currentCommand));
        }

        if (backwardMap.find(BIndex) != backwardMap.end()) {
            auto BDiffIndex = backwardMap.find(BIndex)->second;
            // Compute A Diff
            std::unique_ptr<RegionCommandT> currentCommand(new RegionCommandT);
            currentCommand->op.reset(new OpT);
            currentCommand->op->type = OpType_MatMul;
            currentCommand->op->main.value = new MatMulT;
            currentCommand->op->main.type = OpParameter_MatMul;
            currentCommand->indexes = {BDiffIndex, CDiffIndex, AIndex};
            currentCommand->view.resize(3);
            for (int j=0; j<currentCommand->view.size(); ++j) {
                currentCommand->view[j].reset(new ViewT);
            }
            currentCommand->iterIndexes.resize(3);
            currentCommand->steps.resize(3);
            for (int j=0; j<currentCommand->indexes.size(); ++j) {
                // Compute output info
                auto index = currentCommand->indexes[j];
                currentCommand->iterIndexes[j] = command->iterIndexes[originIndexMap[index]];
                currentCommand->steps[j] = command->steps[originIndexMap[index]];
                *currentCommand->view[j] = *command->view[originIndexMap[index]];
            }
            // TODO: Optimize fuse option
            currentCommand->fuse = BinaryOpOperation_ADD;

            // Reorder the view's stride by size order change to e, h, l
            std::vector<int> order = {2, 0, 1};
            currentCommand->size = {h, e, l};
            for (int j=0; j<currentCommand->indexes.size(); ++j) {
                auto view = currentCommand->view[j].get();
                auto originStride = view->stride;
                for (int k=0; k<3; ++k) {
                    view->stride[k]  = originStride[order[k]];
                }
            }
            // Set Transpose Info by stride
            auto dstMatMulParam = currentCommand->op->main.AsMatMul();
            {
                auto transAView = currentCommand->view[1].get();
                auto transBView = currentCommand->view[1].get();
                dstMatMulParam->transposeA = transAView->stride[1] != 1;
                dstMatMulParam->transposeB = transBView->stride[1] == 1;
            }
            dstCommands.emplace_back(std::move(currentCommand));
        }
    }
    static void _gradForCommand(const RegionCommandT* command, const std::map<int, int>& backwardMap, int tensorNumber, std::map<int, VARP>& extraInputs, std::map<int, VARP>& extraOutputs, std::vector<std::unique_ptr<RegionCommandT>>& dstCommands) {
        if (command->op->type == OpType_MatMul) {
            _gradForCommandMatMul(command, backwardMap, tensorNumber, extraInputs, extraOutputs, dstCommands);
            return;
        }

        if (command->op->type == OpType_UnaryOp) {
            auto inputIndex = command->indexes[1];
            auto outputIndex = command->indexes[0];
            if (backwardMap.find(inputIndex) == backwardMap.end()) {
                return;
            }
            MNN_ASSERT(backwardMap.find(outputIndex) != backwardMap.end());
            auto bpInput = backwardMap.find(outputIndex)->second;
            auto bpOutput = backwardMap.find(inputIndex)->second;

            if (nullptr == command->op->main.value) {
                std::unique_ptr<RegionCommandT> currentCommand(new RegionCommandT);
                currentCommand->op.reset(new OpT);
                currentCommand->op->type = OpType_UnaryOp;
                currentCommand->fuse = BinaryOpOperation_ADD;
                currentCommand->op->main.type = OpParameter_NONE;
                currentCommand->indexes = {bpOutput, bpInput};
                currentCommand->view.resize(2);
                currentCommand->view[0].reset(new ViewT);
                currentCommand->view[1].reset(new ViewT);
                *currentCommand->view[0] = *command->view[1];
                *currentCommand->view[1] = *command->view[0];
                currentCommand->size = command->size;
                currentCommand->iterIndexes = {command->iterIndexes[1], command->iterIndexes[0]};
                currentCommand->steps = {command->steps[1], command->steps[0]};
                dstCommands.emplace_back(std::move(currentCommand));
                return;
            }
            FUNC_PRINT(1);
        }
        int inputSize = 0;
        std::vector<VARP> inputs;
        if (command->op->type == OpType_BinaryOp) {
            inputSize = 2;
        }
        else if (command->op->type == OpType_UnaryOp) {
            inputSize = 1;
        } else {
            MNN_ASSERT(false);
            // TODO: Support MatMul
        }
        for (int i=0; i<inputSize; ++i) {
            auto tempValue = _Const(0.0f, {2, 2}, NHWC);
            inputs.emplace_back(tempValue);
        }
        VARP tempOutput = _Const(0.0f, {2, 2}, NHWC);
        std::map<std::pair<EXPRP, int>, int> allTensors;
        auto commandExpr = Expr::create(command->op.get(), inputs, 1);
        allTensors.insert(std::make_pair(std::make_pair(commandExpr, 0), command->indexes[0]));
        for (int i=0; i<inputSize; ++i) {
            allTensors.insert(std::make_pair(inputs[i]->expr(), command->indexes[i+1]));
        }
        std::map<int, int> originIndexMap;
        for (int i=0; i<command->indexes.size(); ++i) {
            auto index = command->indexes[i];
            originIndexMap.insert(std::make_pair(index, i));
            auto bkIter = backwardMap.find(index);
            if (bkIter != backwardMap.end()) {
                originIndexMap.insert(std::make_pair(bkIter->second, i));
            }
        }

        auto gradOutputIter = backwardMap.find(command->indexes[0]);
        MNN_ASSERT(gradOutputIter != backwardMap.end());
        allTensors.insert(std::make_pair(tempOutput->expr(), gradOutputIter->second));

        auto backwardVars = _getGradExpr(commandExpr, inputs, tempOutput);
        for (int i=0; i<inputSize; ++i) {
            if (nullptr != backwardVars[i]) {
                auto gradInputIter = backwardMap.find(command->indexes[1 + i]);
                if (gradInputIter != backwardMap.end()) {
                    if (backwardVars[i]->expr().first->get() == nullptr) {
                        // Make Copy Command
                        auto inputIndexIter = allTensors.find(backwardVars[i]->expr());
                        MNN_ASSERT(inputIndexIter != allTensors.end());
                        std::unique_ptr<RegionCommandT> currentCommand(new RegionCommandT);
                        currentCommand->op.reset(new OpT);
                        currentCommand->op->type = OpType_UnaryOp;
                        currentCommand->indexes.resize(2);
                        currentCommand->indexes[0] = gradInputIter->second;
                        currentCommand->indexes[1] = inputIndexIter->second;
                        currentCommand->view.resize(2);
                        for (int j=0; j<currentCommand->view.size(); ++j) {
                            currentCommand->view[j].reset(new ViewT);
                        }
                        currentCommand->iterIndexes.resize(2);
                        currentCommand->steps.resize(2);
                        // Compute output info
                        currentCommand->iterIndexes[0] = command->iterIndexes[originIndexMap[currentCommand->indexes[0]]];
                        currentCommand->steps[0] = command->steps[originIndexMap[currentCommand->indexes[0]]];
                        *currentCommand->view[0] = *command->view[originIndexMap[currentCommand->indexes[0]]];
                        currentCommand->size = command->size;
                        // TODO: Optimize fuse option
                        currentCommand->fuse = BinaryOpOperation_ADD;
                        *currentCommand->view[1] = *command->view[originIndexMap[currentCommand->indexes[1]]];
                        currentCommand->iterIndexes[1] = command->iterIndexes[originIndexMap[currentCommand->indexes[1]]];
                        currentCommand->steps[1] = command->steps[originIndexMap[currentCommand->indexes[1]]];
                        dstCommands.emplace_back(std::move(currentCommand));
                    } else {
                        allTensors.insert(std::make_pair(backwardVars[i]->expr(), gradInputIter->second));
                    }
                }
            }
        }
        auto exprLists = Variable::getExecuteOrder(backwardVars);
        for (int i=0; i<exprLists.size(); ++i) {
            auto curExpr = exprLists[i];
            // Find tensor or make cache
            MNN_ASSERT(1 == curExpr->outputSize());
            auto iter = allTensors.find(std::make_pair(curExpr, 0));
            MNN_ASSERT(iter != allTensors.end());
            if (nullptr == curExpr->get()) {
                continue;
            }
            // Make Command
            std::unique_ptr<RegionCommandT> currentCommand(new RegionCommandT);
            currentCommand->op.reset(curExpr->get()->UnPack());
            currentCommand->indexes.resize(curExpr->inputs().size() + curExpr->outputSize());
            currentCommand->indexes[0] = iter->second;
            currentCommand->view.resize(curExpr->inputs().size() + curExpr->outputSize());
            for (int j=0; j<currentCommand->view.size(); ++j) {
                currentCommand->view[j].reset(new ViewT);
            }
            currentCommand->iterIndexes.resize(curExpr->inputs().size() + curExpr->outputSize());
            currentCommand->steps.resize(curExpr->inputs().size() + curExpr->outputSize());
            // Compute output info
            currentCommand->iterIndexes[0] = command->iterIndexes[originIndexMap[iter->second]];
            currentCommand->steps[0] = command->steps[originIndexMap[iter->second]];
            *currentCommand->view[0] = *command->view[originIndexMap[iter->second]];
            currentCommand->size = command->size;
            // TODO: Optimize fuse option
            currentCommand->fuse = BinaryOpOperation_ADD;
            for (int j=0; j<curExpr->inputs().size(); ++j) {
                iter = allTensors.find(curExpr->inputs()[j]->expr());
                if (iter == allTensors.end()) {
                    MNN_ASSERT(false);
                }
                currentCommand->indexes[1 + j] = iter->second;
                MNN_ASSERT(originIndexMap.find(iter->second) != originIndexMap.end());
                *currentCommand->view[1 + j] = *command->view[originIndexMap[iter->second]];
                currentCommand->iterIndexes[1 + j] = command->iterIndexes[originIndexMap[iter->second]];
                currentCommand->steps[1 + j] = command->steps[originIndexMap[iter->second]];
            }
            
            dstCommands.emplace_back(std::move(currentCommand));
        }
        MNN_ASSERT(dstCommands.size() > 0);
    }
    
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        std::vector<VARP> result(inputs.size(), nullptr);
        auto op = expr->get();
        if (op->main_type() != OpParameter_LoopParam) {
            return result;
        }
        std::unique_ptr<LoopParamT> srcParam(op->main_as_LoopParam()->UnPack());
        MNN_ASSERT(srcParam->inputIndexes.size() == inputs.size());
        MNN_ASSERT(srcParam->outputIndexes.size() == backwardOutput.size());
        std::unique_ptr<LoopParamT> dstParam(new LoopParamT);
        dstParam->loopNumber = srcParam->loopNumber;
        dstParam->inputIndexes.resize(srcParam->inputIndexes.size() + srcParam->outputIndexes.size() + backwardOutput.size());
        int dstParamTensorSrcInputOffset = 0;
        int dstParamTensorSrcOutputOffset = srcParam->inputIndexes.size();
        int dstParamTensorDstOutputOffset = srcParam->tensorNumber;
        int dstParamTensorDstInputOffset = srcParam->tensorNumber + backwardOutput.size();
        ::memcpy(dstParam->inputIndexes.data() + 0, srcParam->inputIndexes.data(), srcParam->inputIndexes.size() * sizeof(int));
        ::memcpy(dstParam->inputIndexes.data() + srcParam->inputIndexes.size(), srcParam->outputIndexes.data(), srcParam->outputIndexes.size() * sizeof(int));
        std::map<int, int> backwardMap;
        for (int i=0; i<backwardOutput.size(); ++i) {
            dstParam->inputIndexes[srcParam->inputIndexes.size() + srcParam->outputIndexes.size() + i] = dstParamTensorDstOutputOffset + i;
            backwardMap.insert(std::make_pair(srcParam->outputIndexes[i], dstParamTensorDstOutputOffset + i));
        }
        dstParam->tensorNumber = srcParam->tensorNumber + backwardOutput.size();
        dstParam->outputIndexes.clear();
        std::vector<int> gradInputs;
        for (int i=0; i<inputs.size(); ++i) {
            // Only need compute grad for float tensor
            if (inputs[i]->getInfo()->type.code != halide_type_float) {
                continue;
            }
            gradInputs.emplace_back(i);
            auto curNumber = dstParam->tensorNumber;
            dstParam->outputIndexes.emplace_back(curNumber);
            dstParam->tensorNumber++;
            backwardMap.insert(std::make_pair(srcParam->inputIndexes[i], curNumber));
        }
        // Clear zero firstly for backward inputs
        for (auto index : srcParam->inputIndexes) {
            auto iter = backwardMap.find(index);
            if (iter == backwardMap.end()) {
                continue;
            }
            std::unique_ptr<RegionCommandT> zeroCmd(new RegionCommandT);
            zeroCmd->indexes = {iter->second};
            dstParam->initCommand.emplace_back(std::move(zeroCmd));
        }
        std::map<int, VARP> extraVarps;
        std::map<int, VARP> extraOutputVarps;
        for (int i=0; i<srcParam->commands.size(); ++i) {
            auto reverseI = (int)srcParam->commands.size() - 1 - i;
            auto cmd = srcParam->commands[reverseI].get();
            _gradForCommand(cmd, backwardMap, dstParam->tensorNumber, extraVarps, extraOutputVarps, dstParam->commands);
        }
        // Make Op
        std::vector<VARP> loopInputs(inputs.size() + 2 * expr->outputSize() + extraVarps.size());
        for (int i=0; i<inputs.size(); ++i) {
            loopInputs[i] = inputs[i];
        }
        for (int i=0; i<expr->outputSize(); ++i) {
            loopInputs[i + inputs.size()] = Variable::create(expr, i);
        }
        for (int i=0; i<expr->outputSize(); ++i) {
            loopInputs[i + inputs.size() + expr->outputSize()] = backwardOutput[i];
        }
        for (auto& iter : extraVarps) {
            loopInputs[inputs.size() + 2 * expr->outputSize() + iter.first] = iter.second;
        }
        MNN_ASSERT(dstParam->commands.size() > 0);
        MNN_ASSERT(dstParam->outputIndexes.size() == gradInputs.size());
        for (int i=0; i<gradInputs.size(); ++i) {
            auto info = inputs[gradInputs[i]]->getInfo();
            std::unique_ptr<MNN::TensorDescribeT> describe(new TensorDescribeT);
            describe->index = dstParam->outputIndexes[i];
            describe->blob.reset(new BlobT);
            describe->blob->dims = info->dim;
            describe->blob->dataType = DataType_DT_FLOAT;
            switch (info->order) {
                case MNN::Express::NCHW:
                    describe->blob->dataFormat = MNN_DATA_FORMAT_NCHW;
                    break;
                case MNN::Express::NHWC:
                    describe->blob->dataFormat = MNN_DATA_FORMAT_NHWC;
                    break;
                case MNN::Express::NC4HW4:
                    describe->blob->dataFormat = MNN_DATA_FORMAT_NC4HW4;
                    break;
                default:
                    break;
            }
            dstParam->extraTensorInfos.emplace_back(std::move(describe));
        }

        std::unique_ptr<OpT> loopOp(new OpT);
        loopOp->type = OpType_While;
        loopOp->main.type = OpParameter_LoopParam;
        loopOp->main.value = dstParam.release();

        auto gradExpr = Expr::create(std::move(loopOp), loopInputs, gradInputs.size());
        for (int i=0; i<gradInputs.size(); ++i) {
            result[gradInputs[i]] = Variable::create(gradExpr, i);
        }
        return result;
    }
};

static const auto gRegister = []() {
    static LoopGrad _c;
    OpGrad::insert(OpType_While, &_c);
    return true;
}();
