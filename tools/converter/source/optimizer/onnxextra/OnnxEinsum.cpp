
//
//  OnnxEinsum.cpp
//  MNNConverter
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include <MNN/expr/ExprCreator.hpp>
namespace MNN {
namespace Express {

class OnnxEinsumTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs     = expr->inputs();
        auto op         = expr->get();
        auto extraParam = op->main_as_Extra();
        std::string equation;
        if (nullptr != extraParam->attr()) {
            const int attrSize = extraParam->attr()->size();
            for (int i = 0; i < attrSize; ++i) {
                auto attr       = extraParam->attr()->GetAs<Attribute>(i);
                const auto& key = attr->key()->str();
                if (key == "equation") {
                    equation = attr->s()->str();
                }
            }
        }
        if (equation.empty()) {
            MNN_ERROR("Can't convert Einsum for invalid Equation\n");
            return nullptr;
        }
        // Remove space
        std::vector<char> valid;
        for (int i=0; i<equation.size(); ++i) {
            if (equation[i] != ' ') {
                valid.emplace_back(equation[i]);
            }
        }
        valid.emplace_back('\0');
        equation = std::string(valid.data());
        auto pos = equation.find("->");
        if (pos == std::string::npos) {
            MNN_ERROR("Can't convert Einsum for no support Equation:%s\n", equation.c_str());
            return nullptr;
        }
        auto left = equation.substr(0, pos);
        auto right = equation.substr(pos+2, equation.size());
        if (expr->inputs().size() == 1 ){
            auto currentVar = expr->inputs()[0];
            std::map<char, int> outputPos;
            for (int i=0; i<right.size(); ++i) {
                outputPos.insert(std::make_pair(right[i], i));
            }
            std::vector<int> reduceAxis;
            std::map<char, int> inputPosRemap;
            int pos = 0;
            for (int i=0; i<left.size(); ++i) {
                if (outputPos.find(left[i]) == outputPos.end()) {
                    reduceAxis.emplace_back(i);
                    continue;
                }
                inputPosRemap.insert(std::make_pair(left[i], pos));
                pos++;
            }
            if (!reduceAxis.empty()) {
                currentVar = _ReduceSum(currentVar, reduceAxis, false);
            }
            std::vector<int> permuteDims;
            for (int i=0; i<right.size(); ++i) {
                permuteDims.emplace_back(inputPosRemap[right[i]]);
            }
            currentVar = _Permute(currentVar, permuteDims);
            currentVar->setName(expr->name());
            return currentVar->expr().first;
        }
        if (inputs.size() !=2 ) {
            MNN_ERROR("Can't convert Einsum for input size = %d\n", (int)inputs.size());
            return nullptr;
        }
        auto iPos = left.find(",");
        auto input0 = left.substr(0, iPos);
        auto input1 = left.substr(iPos+1, left.size());
        
        std::map<char, int> input0Pos;
        for (int i=0; i<input0.size(); ++i) {
            input0Pos.insert(std::make_pair(input0[i], i));
        }
        std::map<char, int> input1Pos;
        for (int i=0; i<input1.size(); ++i) {
            input1Pos.insert(std::make_pair(input1[i], i));
        }
        std::map<char, int> outputPos;
        std::vector<char> sumPos;
        std::vector<char> bothPos;
        std::vector<char> aPos;
        std::vector<char> bPos;
        for (int i=0; i<right.size(); ++i) {
            auto c = right[i];
            outputPos.insert(std::make_pair(c, i));
            bool i0Find = input0Pos.find(c) != input0Pos.end();
            bool i1Find = input1Pos.find(c) != input1Pos.end();
            if (i0Find && i1Find) {
                bothPos.emplace_back(c);
                continue;
            }
            if ((!i0Find) && i1Find) {
                bPos.emplace_back(c);
                continue;
            }
            if (i0Find && (!i1Find)) {
                aPos.emplace_back(c);
                continue;
            }
            MNN_ASSERT(false);
        }
        for (int i=0; i<input0.size(); ++i) {
            if (outputPos.find(input0[i]) == outputPos.end()) {
                sumPos.emplace_back(input0[i]);
            }
        }
        auto var0 = expr->inputs()[0];
        auto var1 = expr->inputs()[1];
        if (sumPos.empty()) {
            // Broadcast Mul
            {
                // Reshape + Transpose
                std::vector<int> reshapeDims(outputPos.size(), 0);
                int insertPos = (int)input0Pos.size();
                std::vector<int> transpose;
                for (int i=0; i<right.size(); ++i) {
                    auto iter = input0Pos.find(right[i]);
                    if (iter == input0Pos.end()) {
                        reshapeDims[insertPos] = 1;
                        transpose.emplace_back(insertPos);
                        insertPos++;
                    } else {
                        transpose.emplace_back(iter->second);
                    }
                }
                var0 = _Permute(_Reshape(var0, reshapeDims), transpose);
            }
            {
                // Reshape + Transpose
                std::vector<int> reshapeDims(outputPos.size(), 0);
                int insertPos = (int)input1Pos.size();
                std::vector<int> transpose;
                for (int i=0; i<right.size(); ++i) {
                    auto iter = input1Pos.find(right[i]);
                    if (iter == input1Pos.end()) {
                        reshapeDims[insertPos] = 1;
                        transpose.emplace_back(insertPos);
                        insertPos++;
                    } else {
                        transpose.emplace_back(iter->second);
                    }
                }
                var1 = _Permute(_Reshape(var1, reshapeDims), transpose);
            }
            auto output = var0 * var1;
            output->setName(expr->name());
            return output->expr().first;
        }
        // MatMul
        {
            // Reshape + Transpose
            // AB -> A -> B -> sum
            std::vector<int> reshapeDims(input0Pos.size() + bPos.size(), 0);
            for (int i = (int)input0Pos.size(); i<reshapeDims.size(); ++i) {
                reshapeDims[i] = 1;
            }
            std::vector<int> transpose;
            MNN_ASSERT(bothPos.size() + aPos.size() + bPos.size() + sumPos.size() == reshapeDims.size());
            for (int i=0; i<bothPos.size(); ++i) {
                transpose.emplace_back(input0Pos[bothPos[i]]);
            }
            for (int i=0; i<aPos.size(); ++i) {
                transpose.emplace_back(input0Pos[aPos[i]]);
            }
            for (int i=0; i<bPos.size(); ++i) {
                transpose.emplace_back((int)input0Pos.size() + i);
            }
            for (int i=0; i<sumPos.size(); ++i) {
                transpose.emplace_back(input0Pos[sumPos[i]]);
            }
            var0 = _Permute(_Reshape(var0, reshapeDims), transpose);
        }
        {
            // Reshape + Transpose
            // AB -> A -> B -> sum
            std::vector<int> reshapeDims(input1Pos.size() + aPos.size(), 0);
            for (int i = (int)input1Pos.size(); i<reshapeDims.size(); ++i) {
                reshapeDims[i] = 1;
            }
            std::vector<int> transpose;
            MNN_ASSERT(bothPos.size() + aPos.size() + bPos.size() + sumPos.size() == reshapeDims.size());
            for (int i=0; i<bothPos.size(); ++i) {
                transpose.emplace_back(input1Pos[bothPos[i]]);
            }
            for (int i=0; i<aPos.size(); ++i) {
                transpose.emplace_back((int)input1Pos.size() + i);
            }
            for (int i=0; i<bPos.size(); ++i) {
                transpose.emplace_back(input1Pos[bPos[i]]);
            }
            for (int i=0; i<sumPos.size(); ++i) {
                transpose.emplace_back(input1Pos[sumPos[i]]);
            }
            var1 = _Permute(_Reshape(var1, reshapeDims), transpose);
        }
        auto output = _MatMul(var0, var1, false, true);
        output->setName(expr->name());
        return output->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Einsum", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxEinsumTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
