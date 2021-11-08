//
//  TRTUnary.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTUnary.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "schema/current/MNNPlugin_generated.h"

using namespace std;

namespace MNN {

TRTUnary::TRTUnary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTUnary::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    printf("TRTUnary in\n");
#endif
    if (mOp->main_as_UnaryOp()->opType() == UnaryOpOperation_TANH) {
        auto activationLayer = mTrtBackend->getNetwork()->addActivation(*(xOp[0]), nvinfer1::ActivationType::kTANH);
        return {activationLayer->getOutput(0)};
    }
    if (mOp->main_as_UnaryOp()->opType() == UnaryOpOperation_RSQRT) {
        auto l1       = mTrtBackend->getNetwork()->addUnary(*(xOp[0]), UnaryOperation::kSQRT);
        auto l1Output = l1->getOutput(0);
        auto l2       = mTrtBackend->getNetwork()->addUnary(*l1Output, UnaryOperation::kRECIP);
        return {l2->getOutput(0)};
    }
    if (mOp->main_as_UnaryOp()->opType() == UnaryOpOperation_SIGMOID) {
        auto activationLayer = mTrtBackend->getNetwork()->addActivation(*(xOp[0]), nvinfer1::ActivationType::kSIGMOID);
        return {activationLayer->getOutput(0)};
    }
    if(mOp->main_as_UnaryOp()->opType() == UnaryOpOperation_SIGN){
        // Use SIGN plugin
        auto plu         = createPluginWithOutput(mOutputs);
        auto signPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
        nvinfer1::IPluginLayer *plugin =
            mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 1, *((nvinfer1::IPluginExt *)signPlugin));
        if (plugin == nullptr) {
            printf("plugin == nullptr !!!");
        }
        mTrtBackend->pushReleaseLayer(signPlugin);
        return {plugin->getOutput(0)};
    }
    UnaryOperation operation = UnaryOperation::kEXP;
    switch (mOp->main_as_UnaryOp()->opType()) {
        case UnaryOpOperation_ABS:
            operation = UnaryOperation::kABS;
            break;
        // case UnaryOpOperation_SQUARE:
        //    operation = UnaryOperation::kSQRT;
        //    break;
        case UnaryOpOperation_NEG:
            operation = UnaryOperation::kNEG;
            break;
        case UnaryOpOperation_EXP:
            operation = UnaryOperation::kEXP;
            break;
        case UnaryOpOperation_COS:
            operation = UnaryOperation::kCOS;
            break;
        case UnaryOpOperation_SIN:
            operation = UnaryOperation::kSIN;
            break;
        case UnaryOpOperation_TAN:
            operation = UnaryOperation::kTAN;
            break;
        case UnaryOpOperation_ATAN:
            operation = UnaryOperation::kATAN;
            break;
        case UnaryOpOperation_SQRT:
            operation = UnaryOperation::kSQRT;
            break;
        case UnaryOpOperation_CEIL:
            operation = UnaryOperation::kCEIL;
            break;
        case UnaryOpOperation_RECIPROCAL:
            operation = UnaryOperation::kRECIP;
            break;
        // case UnaryOpOperation_LOG1P:
        // operation = UnaryOperation::kCOS;
        // break;
        case UnaryOpOperation_LOG:
            operation = UnaryOperation::kLOG;
            break;
        case UnaryOpOperation_FLOOR:
            operation = UnaryOperation::kFLOOR;
            break;
        // case UnaryOpOperation_BNLL:
        // operation = UnaryOperation::kCOS;
        // break;
        case UnaryOpOperation_ACOSH:
            operation = UnaryOperation::kACOSH;
            break;
        case UnaryOpOperation_SINH:
            operation = UnaryOperation::kSINH;
            break;
        case UnaryOpOperation_ASINH:
            operation = UnaryOperation::kASINH;
            break;
        case UnaryOpOperation_ATANH:
            operation = UnaryOperation::kATANH;
            break;
        // //case UnaryOpOperation_SIGN:
        //     operation = UnaryOperation::kCOS;
        //     break;
        // //case UnaryOpOperation_ROUND:
        //     operation = UnaryOperation::kCOS;
        //     break;
        case UnaryOpOperation_COSH:
            operation = UnaryOperation::kCOSH;
            break;
#ifdef MNN_USE_TRT7
        case UnaryOpOperation_ERF:
            operation = UnaryOperation::kERF;
            break;
#else
        case UnaryOpOperation_ERF:
            {
                // Use SIGN plugin
                auto plu         = createPluginWithOutput(mOutputs);
                auto signPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
                nvinfer1::IPluginLayer *plugin =
                    mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 1, *((nvinfer1::IPluginExt *)signPlugin));
                if (plugin == nullptr) {
                    printf("plugin == nullptr !!!");
                }
                mTrtBackend->pushReleaseLayer(signPlugin);
                return {plugin->getOutput(0)};
            }
#endif
        // case UnaryOpOperation_ERFC:
        //    return _unaryOp<UnaryErfc<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(),
        //    backend());
        // case UnaryOpOperation_ERFINV:
        //    return _unaryOp<UnaryErfinv<float>, float>(input->host<void>(), output->host<void>(),
        //    input->elementSize(), backend());
        // case UnaryOpOperation_EXPM1:
        //    return _unaryOp<UnaryExpm1<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(),
        //    backend());
        case UnaryOpOperation_ASIN:
            operation = UnaryOperation::kASIN;
            break;
        case UnaryOpOperation_ACOS:
            operation = UnaryOperation::kACOS;
            break;
        case UnaryOpOperation_HARDSWISH:
            {
                auto plu         = createPluginWithOutput(mOutputs);
                auto signPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
                nvinfer1::IPluginLayer *plugin =
                    mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 1, *((nvinfer1::IPluginExt *)signPlugin));
                if (plugin == nullptr) {
                    printf("plugin == nullptr !!!");
                }
                mTrtBackend->pushReleaseLayer(signPlugin);
                return {plugin->getOutput(0)};
            }
        default:
            MNN_PRINT("unary not support this type : %d \n", mOp->main_as_UnaryOp()->opType());
            MNN_ASSERT(false);
            break;
    }

    auto Unary_layer = mTrtBackend->getNetwork()->addUnary(*(xOp[0]), operation);
    auto output      = Unary_layer->getOutput(0);
    return {output};
}

TRTCreatorRegister<TypedCreator<TRTUnary>> __Unary_op(OpType_UnaryOp);

} // namespace MNN
