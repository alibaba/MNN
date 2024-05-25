//
//  Conv1dSqueezeMove.cpp
//  MNNConverter
//
//  Created by MNN on 2021/03/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../TemplateMerge.hpp"
#include "MNN_generated.h"
#include "MergeHelpers.hpp"
#include "cli.hpp"
#include "commonKit.hpp"
#include "MNN_compression.pb.h"
#include <fstream>

namespace MNN {
namespace Express {

enum Conv1dPostCases {
    None,
    BiasAdd,
    Relu,
    // don't need BiasAddRelu
};

auto getConv1dPostCase = [](EXPRP expr) {
    auto noPost = Conv1dPostCases::None;
    auto returnPost = noPost;

    if (nullptr == expr->get()) {
        return noPost;
    }

    auto opType = expr->get()->type();

    auto gConverterConfig = Global<modelConfig>::Get();
    std::string compressFileName = gConverterConfig->compressionParamsFile;
    Compression::Pipeline proto;
    if (compressFileName != "") {
        std::string jsonSuffix = "json";
        std::string suffix = compressFileName.substr(compressFileName.find_last_of('.') + 1);
        if (suffix.compare(jsonSuffix) != 0) {
            std::fstream input(compressFileName.c_str(), std::ios::in | std::ios::binary);
            if (!proto.ParseFromIstream(&input)) {
                MNN_ERROR("Failed to parse compression pipeline proto.\n");
            }
        } else {
            CommonKit::json2protobuf(compressFileName.c_str(), nullptr, &proto);
        }
    }

    auto findQuantParameters = [&](Compression::Pipeline& proto, std::string outputTensorName) {
        for (const auto& algo : proto.algo()) {
            if (algo.type() == Compression::CompressionAlgo::QUANTIZE) {
                auto quantParams = algo.quant_params();
                for (const auto& layerProto : quantParams.layer()) {
                    const std::string& outputName = layerProto.output(0).name();
                    if (outputName == outputTensorName) {
                        return layerProto;
                    }
                }
            }
        }
        MNN::Compression::LayerQuantizeParams empty;
        return empty;
    };

    EXPRP squeezeExpr = nullptr;

    // BiasAdd
    if (opType == OpType::OpType_BinaryOp) {
        auto binaryOp     = expr->get();
        auto binaryParams = binaryOp->main_as_BinaryOp();
        if (binaryParams->opType() != BinaryOpOperation_ADD) {
            return noPost;
        }

        auto input0 = expr->inputs()[0];
        auto expr0  = input0->expr().first;
        auto input1 = expr->inputs()[1];
        auto expr1  = input1->expr().first;

        EXPRP constExpr = nullptr;
        VARP constVar = nullptr;

        if (helpers::IsConstant(expr0) && helpers::IsConstant(expr1)) {
            return noPost;
        }
        if (helpers::IsConstant(expr0)) {
            constExpr = expr0;
            constVar = input0;
            squeezeExpr = expr1;
        } else if (helpers::IsConstant(expr1)) {
            constExpr = expr1;
            constVar = input1;
            squeezeExpr = expr0;
        } else {
            return noPost;
        }

        if (constExpr->get() == nullptr) { // expr const
            if (constVar->getInfo()->dim.size() > 1) {
                return noPost;
            }
        } else { // op const
            auto constParam = constExpr->get()->main_as_Blob();
            if (constParam->dims()->size() > 1) {
                return noPost;
            }
        }

        if (!squeezeExpr->get() || squeezeExpr->get()->type() != OpType::OpType_Squeeze) {
            return noPost;
        }
        if (OpParameter_SqueezeParam != squeezeExpr->get()->main_type()) {
            return noPost;
        }
        auto squeezeDims = squeezeExpr->get()->main_as_SqueezeParam()->squeezeDims();
        if (nullptr == squeezeDims) {
            return noPost;
        }
        if (squeezeDims->size() != 1) {
            return noPost;
        }
        if ((squeezeDims->data()[0] == -1) || (squeezeDims->data()[0] == 3)) {
            return noPost;
        }

        returnPost = Conv1dPostCases::BiasAdd;
    }
    // relu
    else if (opType == OpType::OpType_ReLU || opType == OpType::OpType_ReLU6) {
        auto input = expr->inputs()[0];
        auto inputExpr  = input->expr().first;

        if (!inputExpr->get() || inputExpr->get()->type() != OpType::OpType_Squeeze) {
            return noPost;
        }
        squeezeExpr = inputExpr;

        returnPost = Conv1dPostCases::Relu;
    }
    else {
        return noPost;
    }

    if (squeezeExpr != nullptr) {
        auto squeezeInput = squeezeExpr->inputs()[0];
        auto squeezeInputExpr = squeezeInput->expr().first;
        if (squeezeInputExpr->get() && squeezeInputExpr->get()->main_type() == OpParameter_Convolution2D && squeezeInputExpr->outputs().size() == 1) {
            if (compressFileName != "") {
                auto quantParams = findQuantParameters(proto, squeezeInputExpr->outputName(0));
                // some conv1d squeeze may not be considered
                if (quantParams.weight_size() != 0) {
                    return noPost;
                }
            }
        }
    }

    return returnPost;
};

static auto gRegister = []() {
    auto match = [](EXPRP expr) {
        auto postCase = getConv1dPostCase(expr);
        if (postCase != Conv1dPostCases::None) {
            return true;
        }

        return false;
    };

    auto transform = [](EXPRP expr) {
        auto postCase = getConv1dPostCase(expr);

        if (postCase == Conv1dPostCases::BiasAdd) {
            auto input0 = expr->inputs()[0];
            auto expr0  = input0->expr().first;
            auto input1 = expr->inputs()[1];
            auto expr1  = input1->expr().first;

            EXPRP constExpr = nullptr;
            VARP constVar = nullptr;
            EXPRP squeezeExpr = nullptr;
            VARP squeezeInput = nullptr;
            int constIndex = 0;
            std::vector<VARP> newBiasAddInputs;

            if (helpers::IsConstant(expr0)) {
                constExpr = expr0;
                constVar = input0;
                squeezeExpr = expr1;
                squeezeInput = expr1->inputs()[0];
                constIndex = 0;
            } else if (helpers::IsConstant(expr1)) {
                constExpr = expr1;
                constVar = input1;
                squeezeExpr = expr0;
                squeezeInput = expr0->inputs()[0];
                constIndex = 1;
            }

            auto squeezeInputExpr = squeezeInput->expr().first;
            if (squeezeInputExpr->get() && squeezeInputExpr->get()->main_type() == OpParameter_Convolution2D && squeezeInputExpr->outputs().size() == 1) {
                auto convInput = squeezeInputExpr->inputs();
                auto newConvExpr = Expr::create(squeezeInputExpr->extra(), std::move(convInput));
                newConvExpr->setName(squeezeInputExpr->name());
                auto newConvOutput = Variable::create(newConvExpr, 0);
                newConvOutput->setName(squeezeInputExpr->outputName(0));
                squeezeInput = newConvOutput;
            }

            if (constIndex == 0) {
                newBiasAddInputs.push_back(constVar);
                newBiasAddInputs.push_back(squeezeInput);
            } else {
                newBiasAddInputs.push_back(squeezeInput);
                newBiasAddInputs.push_back(constVar);
            }
            
            auto newBiasAddExpr = Expr::create(expr->extra(), std::move(newBiasAddInputs));
            newBiasAddExpr->setName(expr->name());
            auto newBiasAddVar = Variable::create(newBiasAddExpr, 0);
            newBiasAddVar->setName(expr->outputName(0));
            auto squeezeExprInputs = squeezeExpr->inputs();
            squeezeExprInputs[0] = newBiasAddVar;
            auto newSqueezeExpr = Expr::create(squeezeExpr->extra(), std::move(squeezeExprInputs));
            newSqueezeExpr->setName(squeezeExpr->name());
            auto newSqueezeVar = Variable::create(newSqueezeExpr, 0);
            newSqueezeVar->setName(squeezeExpr->outputName(0));

            Expr::replace(expr, newSqueezeExpr);
            return true;
        }

        if (postCase == Conv1dPostCases::Relu) {
            auto input = expr->inputs()[0];
            auto squeezeExpr  = input->expr().first;
            auto squeezeInput = squeezeExpr->inputs()[0];
            auto squeezeInputExpr = squeezeInput->expr().first;

            if (squeezeInputExpr->get() && squeezeInputExpr->get()->main_type() == OpParameter_Convolution2D && squeezeInputExpr->outputs().size() == 1) {
                auto convInput = squeezeInputExpr->inputs();
                auto newConvExpr = Expr::create(squeezeInputExpr->extra(), std::move(convInput));
                newConvExpr->setName(squeezeInputExpr->name());
                auto newConvOutput = Variable::create(newConvExpr, 0);
                newConvOutput->setName(squeezeInputExpr->outputName(0));
                squeezeInput = newConvOutput;
            }

            auto newReluExpr = Expr::create(expr->extra(), {squeezeInput});
            newReluExpr->setName(expr->name());
            auto newReluVar = Variable::create(newReluExpr, 0);
            newReluVar->setName(expr->outputName(0));
            auto squeezeExprInputs = squeezeExpr->inputs();
            squeezeExprInputs[0] = newReluVar;
            auto newSqueezeExpr = Expr::create(squeezeExpr->extra(), std::move(squeezeExprInputs));
            newSqueezeExpr->setName(squeezeExpr->name());
            auto newSqueezeVar = Variable::create(newSqueezeExpr, 0);
            newSqueezeVar->setName(squeezeExpr->outputName(0));

            Expr::replace(expr, newSqueezeExpr);
            return true;
        }

        return false;
    };

    TemplateMerge::getInstance("Merge").insertTemplate("Conv1dSqueezeMove", match, transform,
                                                       PASS_PRIORITY_HIGH);
    return true;
}();

}
} // namespace MNN
