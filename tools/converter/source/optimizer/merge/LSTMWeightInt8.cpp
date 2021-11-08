//
//  LSTMWeightInt8.cpp
//  MNNConverter
//
//  Created by MNN on 2020/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <flatbuffers/util.h>
#include "../TemplateMerge.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"
#include "cli.hpp"

namespace MNN {
namespace Express {

class LSTMWeightInt8 {
public:
    LSTMWeightInt8();
};

LSTMWeightInt8::LSTMWeightInt8() {
    auto match = [](EXPRP expr) -> bool {
        auto gConverterConfig = Global<modelConfig>::Get();
        if (!gConverterConfig->weightQuantBits) {
            return false;
        }
        if (!expr->get()) {
            return false;
        }
        if (expr->get()->type() != OpType_LSTM) {
            return false;
        }
        if (expr->inputs().size() != 6) {
            return false;
        }
        for (int i = 1; i < 4; i++) {
            auto input   = expr->inputs()[i];
            const Op* op = input->expr().first->get();
            if (input->expr().first->inputType() != VARP::CONSTANT) {
                return false;
            }
            auto inputInfo = input->getInfo();
            if (inputInfo->type != halide_type_of<float>()) {
                return false;
            }
        }

        return true /*matched*/;
    };

    auto fold = [this](EXPRP expr) -> bool {
        auto allInputs = expr->inputs();
        std::vector<VARP> lstmInputs;
        lstmInputs.emplace_back(allInputs[0]);

        for (int i = 1; i < 4; i++) {
            VARP weightVar  = allInputs[i];
            auto weightInfo = weightVar->getInfo();

            int lastDimSize = weightInfo->dim[weightInfo->dim.size() - 1];
            std::vector<int> reduceDims;
            for (int j = 0; j < weightInfo->dim.size(); j++) {
                reduceDims.push_back(j);
            }
            VARP mins        = _ReduceMin(weightVar, reduceDims, true);
            VARP maxs        = _ReduceMax(weightVar, reduceDims, true);
            VARP scales      = (maxs - mins) / _Scalar<float>(127. + 128.);
            VARP quantWeight = _Cast<int8_t>(_Round((weightVar - mins) / scales) - _Scalar<float>(128.));

            VARP minsConst =
                _Const(mins->readMap<void>(), mins->getInfo()->dim, mins->getInfo()->order, mins->getInfo()->type);
            VARP scalesConst      = _Const(scales->readMap<void>(), scales->getInfo()->dim, scales->getInfo()->order,
                                      scales->getInfo()->type);
            VARP quantWeightConst = _Const(quantWeight->readMap<void>(), quantWeight->getInfo()->dim,
                                           quantWeight->getInfo()->order, quantWeight->getInfo()->type);

            VARP deQuantWeight = (_Cast<float>(quantWeightConst) + _Scalar<float>(128.)) * scalesConst + minsConst;

            lstmInputs.emplace_back(deQuantWeight);
        }

        lstmInputs.emplace_back(allInputs[4]);
        lstmInputs.emplace_back(allInputs[5]);

        auto lstm_op    = expr->get();
        EXPRP lstm_expr = Expr::create(lstm_op->UnPack(), lstmInputs, 3);
        lstm_expr->setName(expr->name());
        VARP lstm_var0 = Variable::create(lstm_expr, 0);
        lstm_var0->setName(expr->outputName(0));
        VARP lstm_var1 = Variable::create(lstm_expr, 1);
        lstm_var1->setName(expr->outputName(1));
        VARP lstm_var2 = Variable::create(lstm_expr, 2);
        lstm_var2->setName(expr->outputName(2));
        Expr::replace(expr, lstm_expr);

        return true /*modified*/;
    };
    TemplateMerge::getInstance("Merge").insertTemplate("LSTMWeightInt8", match, fold, PASS_PRIORITY_LOW);
}

static LSTMWeightInt8 g_lstm_weight_int8;

} // namespace Express
} // namespace MNN
