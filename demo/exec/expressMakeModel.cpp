//
//  expressMakeModel.cpp
//  MNN
//
//  Created by MNN on b'2021/10/18'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace MNN::Express;
#define UP_DIV(x) (((x)+3)/4)

static void initSnow() {
    int n = 100000;
    auto time = _Input({}, NHWC, halide_type_of<float>());
    time = time + _Scalar<float>(30.0f);
    std::vector<int> shape = {3, n};
    auto shapeVar = _Const(shape.data(), {2}, NHWC, halide_type_of<int>());
    auto rateSpeed = _RandomUnifom(shapeVar, halide_type_of<float>(), 0.0f, 1.0f);
    auto ratePos = _RandomUnifom(shapeVar, halide_type_of<float>(), 0.0f, 1.0f);
    std::vector<float> maxSpeed = {2.0f, -20.f, 2.f};
    std::vector<float> minSpeed = {-2.0f, -5.f, -2.f};
    auto maxSpeedVar = _Const(maxSpeed.data(), {3, 1}, NHWC, halide_type_of<float>());
    auto minSpeedVar = _Const(minSpeed.data(), {3, 1}, NHWC, halide_type_of<float>());
    auto speedVar = (maxSpeedVar - minSpeedVar) * rateSpeed + minSpeedVar;
    std::vector<float> minPos = {-100.0f, -100.0f, -100.0f};
    std::vector<float> rangePos = {200.0f, 200.0f, 200.0f};
    auto minPosVar = _Const(minPos.data(), {3, 1}, NHWC, halide_type_of<float>());
    auto rangePosVar = _Const(rangePos.data(), {3, 1}, NHWC, halide_type_of<float>());
    auto rangePosVarDiv = _Reciprocal(rangePosVar);
    rangePosVarDiv.fix(MNN::Express::VARP::CONSTANT);
    auto timePosVar = rangePosVar * ratePos + speedVar * time;
    timePosVar = timePosVar - _Floor(timePosVar * rangePosVarDiv) * rangePosVar;
    timePosVar = timePosVar + minPosVar;
    std::vector<int> shapeZero = {1, n};
    auto shapeZeroVar = _Const(shapeZero.data(), {2}, NHWC, halide_type_of<int>());
    auto zeroVar = _Fill(shapeZeroVar, _Scalar<float>(1.0f));
    zeroVar.fix(MNN::Express::VARP::CONSTANT);
    timePosVar = _Concat({timePosVar, zeroVar}, 0);
    timePosVar = _Transpose(timePosVar, {1, 0});
    Variable::save({timePosVar}, "pos.mnn");
}
static int addPostPretreat() {
    auto varMap = Variable::loadMap("seg.mnn");
    auto input = varMap["sub_7"];
    auto output = varMap["ResizeBilinear_3"];
    output = _Convert(output, NHWC);
    auto width = output->getInfo()->dim[2];
    auto height = output->getInfo()->dim[1];
    auto channel = output->getInfo()->dim[3];

    const int humanIndex = 15;
    output = _Reshape(output, {-1, channel});
    auto kv = _TopKV2(output, _Scalar<int>(1));
    // Use indice in TopKV2's C axis
    auto index = kv[1];
    // If is human, set 255, else set 0
    //auto mask = _Select(_Equal(index, _Scalar<int>(humanIndex)), _Scalar<int>(255), _Scalar<int>(0));
    auto mask = _Equal(index, _Scalar<int>(humanIndex)) * _Scalar<int>(255);
    mask = _Cast<uint8_t>(mask);
    Variable::save({mask}, "mask.mnn");
    return 0;
}
int main(int argc, const char* argv[]) {
    initSnow();
    addPostPretreat();
    return 0;
}
