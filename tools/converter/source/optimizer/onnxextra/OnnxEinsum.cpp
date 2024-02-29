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

static VARP _ReshapeF(VARP x, VARP shape, MNN::MNN_DATA_FORMAT format) {
    MNN_ASSERT(nullptr != x);
    std::unique_ptr<OpT> reshape(new OpT);
    reshape->type                      = OpType_Reshape;
    reshape->main.type                 = OpParameter_Reshape;
    reshape->main.value                = new ReshapeT;
    reshape->main.AsReshape()->dimType = format;
    return (Variable::create(Expr::create(reshape.get(), {x, shape})));
}

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
        auto var0 = expr->inputs()[0];
        auto var1 = expr->inputs()[1];
        // dim = 4
        if (right.size() == 4) {
            // batch align:
            // bhwc,bhkc -> bhwk  batch = `bh`, reduce_dim = `c`
            // bhwc,hkc -> bhwk   batch = `bh`, reduce_dim = `c`, need broadcast
            // bhwc,wkc -> bhwk   batch = `bhw`, reduce_dim = `c`, need unsqeeze, broadcast, squeeze
            int sqeeze_axis = -1;
            if (input0.size() != input1.size()) {
                int pos0 = 0, pos1 = 0;
                if (input0.size() > input1.size()) {
                    for (int i = 0; i < input1.size(); i++) {
                        auto c = input1[i];
                        bool right_has = right.find(c) != std::string::npos;
                        auto upos0 = input0.find(c);
                        bool input0_has = upos0 != std::string::npos;
                        if (right_has && input0_has) {
                            pos0 = static_cast<int>(upos0);
                            pos1 = i;
                        }
                    }
                } else {
                    for (int i = 0; i < input0.size(); i++) {
                        auto c = input0[i];
                        bool right_has = right.find(c) != std::string::npos;
                        auto upos1 = input1.find(c);
                        bool input1_has = upos1 != std::string::npos;
                        if (right_has && input1_has) {
                            pos0 = i;
                            pos1 = static_cast<int>(upos1);
                        }
                    }
                }
                if (input0.size() - pos0 < 3) {
                    sqeeze_axis = pos0 + 1;
                    var0 = _Unsqueeze(var0, {sqeeze_axis});
                }
                if (input1.size() - pos1 < 3) {
                    sqeeze_axis = pos1 + 1;
                    var1 = _Unsqueeze(var1, {sqeeze_axis});
                }
            }
            // find reduce dim
            char reduce_dim;
            for (int i = 0; i < input0.size(); ++i) {
                auto c = input0[i];
                if (right.find(c) == std::string::npos) {
                    reduce_dim = c;
                    break;
                }
            }
            auto need_transpose = input1.find(reduce_dim) == (input1.size() - 1);
            // matmul: matmul auto broadcast such: `bhwc @ hkc` -> `bhwc @ bhkc`
            auto output = _MatMul(var0, var1, false, need_transpose);
            // squeeze
            if (sqeeze_axis >= 0) {
                output = _Squeeze(output, {sqeeze_axis});
            }
            output->setName(expr->name());
            return output->expr().first;
        }
        
        if(right.size() == 3) {
            // bid, bjd -> bij
            if(input0.size() == 3 && input1.size() == 3) {
                if(input0[0] == input1[0] && input0[0] == right[0]) {
                    if (input0[2] == input1[2]) {// bid, bjd
                        auto output = _MatMul(var0, var1, false, true);
                        output->setName(expr->name());
                        return output->expr().first;
                    } else if (input0[2] == input1[1]) {// bid, bdj
                        auto output = _MatMul(var0, var1, false, false);
                        output->setName(expr->name());
                        return output->expr().first;
                    } else if (input0[1] == input1[1]) {// bdi, bdj
                        auto output = _MatMul(var0, var1, true, false);
                        output->setName(expr->name());
                        return output->expr().first;
                    } else if (input0[1] == input1[2]) {// bdi, bjd
                        auto output = _MatMul(var0, var1, true, true);
                        output->setName(expr->name());
                        return output->expr().first;
                    }
                }
            }
        }
        
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
        // dim < 4
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
                auto _shape  = _Const(reshapeDims.data(), {static_cast<int32_t>(right.size())}, NHWC, halide_type_of<int>());
                var0 = _ReshapeF(var0, _shape, MNN::MNN_DATA_FORMAT_NCHW);
                var0 = _Permute(var0, transpose);
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
                auto _shape  = _Const(reshapeDims.data(), {static_cast<int>(right.size())}, NHWC, halide_type_of<int>());
                var1 = _ReshapeF(var1, _shape, MNN::MNN_DATA_FORMAT_NCHW);
                var1 = _Permute(var1, transpose);
            }
            auto output = var0 * var1;
            output->setName(expr->name());
            return output->expr().first;
        }
        auto aShape = _Shape(var0, NCHW);
        auto bShape = _Shape(var1, NCHW);
        auto one = _Unsqueeze(_Scalar<int>(1), {0});

        // MatMul
        // Remove sum pos from aPos and bPos
        std::vector<char> tempA;
        for (int i=0; i<aPos.size(); ++i) {
            bool find = false;
            for (int j=0; j<sumPos.size(); ++j) {
                if (sumPos[j] == aPos[i]) {
                    find = true;
                    break;
                }
            }
            if (!find) {
                tempA.emplace_back(aPos[i]);
            }
        }
        aPos = tempA;
        std::vector<char> tempB;
        for (int i=0; i<bPos.size(); ++i) {
            bool find = false;
            for (int j=0; j<sumPos.size(); ++j) {
                if (sumPos[j] == bPos[i]) {
                    find = true;
                    break;
                }
            }
            if (!find) {
                tempB.emplace_back(bPos[i]);
            }
        }
        bPos = tempB;
        // outside and sum is common for A and B
        VARP outsideLength = _Unsqueeze(_Scalar<int>(1), {0});
        int needBroadcast0 = 0, needBroadcast1 = 0;
        for (int i=0; i<bothPos.size(); ++i) {
            auto size0 = _Slice(aShape, _Unsqueeze(_Scalar<int>(input0Pos[bothPos[i]]), {0}), one);
            auto size1 = _Slice(bShape, _Unsqueeze(_Scalar<int>(input1Pos[bothPos[i]]), {0}), one);
            auto bothsize = size0;
            if (size0 < size1) {
                bothsize = size1;
                needBroadcast0 = 1;
            } else if (size0 == size1) {
                // do nothing.
            } else {
                needBroadcast1 = 1;
            }
            outsideLength = outsideLength * bothsize;
        }
        
        VARP sumLength = _Unsqueeze(_Scalar<int>(1), {0});
        for (int i=0; i<sumPos.size(); ++i) {
            sumLength = sumLength * _Slice(aShape, _Unsqueeze(_Scalar<int>(input0Pos[sumPos[i]]), {0}), one);
        }
        {
            // Transpose and reshape as 3 dimension
            // AB -> A -> sum
            std::vector<int> transpose;
            for (int i=0; i<bothPos.size(); ++i) {
                transpose.emplace_back(input0Pos[bothPos[i]]);
            }
            VARP ALength = _Unsqueeze(_Scalar<int>(1), {0});
            for (int i=0; i<aPos.size(); ++i) {
                transpose.emplace_back(input0Pos[aPos[i]]);
                ALength = ALength * _Slice(aShape, _Unsqueeze(_Scalar<int>(input0Pos[aPos[i]]), {0}), one);
            }
            for (int i=0; i<sumPos.size(); ++i) {
                transpose.emplace_back(input0Pos[sumPos[i]]);
            }
            var0 = _Permute(var0, transpose);
            if (needBroadcast0) {
                var0 = _BroadcastTo(var0, _Concat({outsideLength, ALength, sumLength}, 0));
            } else {
                var0 = _ReshapeF(var0, _Concat({outsideLength, ALength, sumLength}, 0), MNN::MNN_DATA_FORMAT_NCHW);
            }
        }
        {
            // Transpose
            // AB -> B -> sum
            std::vector<int> transpose;
            for (int i=0; i<bothPos.size(); ++i) {
                transpose.emplace_back(input1Pos[bothPos[i]]);
            }
            VARP BLength = _Unsqueeze(_Scalar<int>(1), {0});
            for (int i=0; i<bPos.size(); ++i) {
                transpose.emplace_back(input1Pos[bPos[i]]);
                BLength = BLength * _Slice(bShape, _Unsqueeze(_Scalar<int>(input1Pos[bPos[i]]), {0}), one);
            }
            for (int i=0; i<sumPos.size(); ++i) {
                transpose.emplace_back(input1Pos[sumPos[i]]);
            }
            var1 = _Permute(var1, transpose);
            if (needBroadcast1) {
                var1 = _BroadcastTo(var1, _Concat({outsideLength, BLength, sumLength}, 0));
            } else {
                var1 = _ReshapeF(var1, _Concat({outsideLength, BLength, sumLength}, 0), MNN::MNN_DATA_FORMAT_NCHW);
            }
        }
        auto output = _MatMul(var0, var1, false, true);
        std::vector<VARP> cShapeGroup;

        // Permute output if needed, origin dimension pos is AB - A - B
        std::map<char, int> originOutputPos;
        for (int i=0; i<bothPos.size(); ++i) {
            originOutputPos.insert(std::make_pair(bothPos[i], i));
            cShapeGroup.emplace_back(_Slice(aShape, _Unsqueeze(_Scalar<int>(input0Pos[bothPos[i]]), {0}), one));
        }
        for (int i=0; i<aPos.size(); ++i) {
            originOutputPos.insert(std::make_pair(aPos[i], i + bothPos.size()));
            cShapeGroup.emplace_back(_Slice(aShape, _Unsqueeze(_Scalar<int>(input0Pos[aPos[i]]), {0}), one));
        }
        for (int i=0; i<bPos.size(); ++i) {
            originOutputPos.insert(std::make_pair(bPos[i], i + bothPos.size() + aPos.size()));
            cShapeGroup.emplace_back(_Slice(bShape, _Unsqueeze(_Scalar<int>(input1Pos[bPos[i]]), {0}), one));
        }
        auto cShape = _Concat(cShapeGroup, 0);
        output = _ReshapeF(output, cShape, MNN::MNN_DATA_FORMAT_NCHW);
        bool needPermute = false;
        std::vector<int> transpose(right.size());
        for (int i=0; i<right.size(); ++i) {
            transpose[i] = originOutputPos[right[i]];
            if (transpose[i] != i) {
                needPermute = true;
            }
        }
        if (needPermute) {
            output = _Permute(output, transpose);
        }
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
