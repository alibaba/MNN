#include "HexagonExecutionFactory.hpp"

#include "HexagonConvolution.hpp"
#include "HexagonDeconvolution.hpp"
#include "HexagonTMac.hpp"
#include "HexagonConvolutionDepthwise.hpp"
#include "HexagonPooling.hpp"
#include "HexagonRaster.hpp"
#include "HexagonScale.hpp"
#include "HexagonLoop.hpp"
#include "HexagonUnary.hpp"
#include "HexagonBinary.hpp"
#include "HexagonCast.hpp"
#include "HexagonLayerNorm.hpp"
#include "HexagonRoPE.hpp"
#include "HexagonAttention.hpp"
#include "HexagonSelect.hpp"
#include "HexagonTopKV2.hpp"
#include "HexagonSoftmax.hpp"
#include "HexagonReduction.hpp"
#include "HexagonRelu.hpp"
#include "HexagonRelu6.hpp"
#include "HexagonPRelu.hpp"
#include "HexagonLSTM.hpp"
#include "core/Macro.h"

namespace MNN {

static void logUnsupportedOp(const Op* op, const std::vector<Tensor*>& inputs,
                             const std::vector<Tensor*>& outputs) {
    if (op == nullptr) {
        MNN_PRINT("[MNN::Hexagon] create unsupported: null op, inputs=%zu, outputs=%zu\n", inputs.size(),
                  outputs.size());
        return;
    }
    const char* typeName = EnumNameOpType(op->type());
    const char* opName = op->name() == nullptr ? "" : op->name()->c_str();
    MNN_PRINT("[MNN::Hexagon] create unsupported: type=%s(%d), name=%s, inputs=%zu, outputs=%zu\n",
              typeName == nullptr ? "Unknown" : typeName, op->type(), opName, inputs.size(), outputs.size());
}

Execution* HexagonExecutionFactory::create(const Op* op, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs, Backend* backend) {
    if (op == nullptr) {
        logUnsupportedOp(op, inputs, outputs);
        return nullptr;
    }
    Execution* execution = nullptr;
    switch (op->type()) {
        case OpType_LayerNorm:
            execution = HexagonLayerNorm::create(backend, op);
            break;
        case OpType_Convolution:
            if (inputs.size() > 1) {
                break;
            }
            if (auto exe = HexagonTMac::create(backend, op, inputs, outputs)) {
                return exe;
            }
            execution = HexagonConvolution::create(backend, op);
            break;
        case OpType_ConvolutionDepthwise:
            execution = HexagonConvolutionDepthwise::create(backend, op);
            break;
        case OpType_Deconvolution:
            execution = HexagonDeconvolution::create(backend, op, inputs, outputs);
            break;
        case OpType_Scale:
            execution = HexagonScale::create(backend, op);
            break;
        case OpType_Pooling:
            if (outputs.size() > 1) {
                break;
            }
            execution = HexagonPooling::create(backend, op);
            break;
        case OpType_Raster:
            execution = HexagonRaster::create(backend, op);
            break;
        case OpType_While:
            execution = HexagonLoop::create(backend, op);
            break;
        case OpType_UnaryOp:
            execution = HexagonUnary::create(backend, op);
            break;
        case OpType_Reduction:
            execution = HexagonReduction::create(backend, op, inputs, outputs);
            break;
        case OpType_ReLU:
            execution = HexagonRelu::create(backend, op);
            break;
        case OpType_ReLU6:
            execution = HexagonRelu6::create(backend, op);
            break;
        case OpType_PReLU:
            execution = HexagonPRelu::create(backend, op, inputs, outputs);
            break;
        case OpType_BinaryOp:
            execution = HexagonBinary::create(backend, op);
            break;
        case OpType_Cast:
            execution = HexagonCast::create(backend, op, inputs, outputs);
            break;
        case OpType_RoPE:
            return HexagonRoPE::create(backend, op);
        case OpType_Attention:
            execution = HexagonAttention::create(backend, op);
            break;
        case OpType_Select:
            execution = HexagonSelect::create(backend, op);
            break;
        case OpType_TopKV2:
            execution = HexagonTopKV2::create(backend, op, inputs, outputs);
            break;
        case OpType_Softmax:
            execution = HexagonSoftmax::create(backend, op, inputs, outputs);
            break;
        case OpType_LSTM:
            execution = HexagonLSTM::create(backend, op, inputs, outputs);
            break;
        default:
            break;
    }
    if (execution == nullptr) {
        logUnsupportedOp(op, inputs, outputs);
    }
    return execution;
}

} // namespace MNN
