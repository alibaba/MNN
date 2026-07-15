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

namespace MNN {

Execution* HexagonExecutionFactory::create(const Op* op, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs, Backend* backend) {
    switch (op->type()) {
        case OpType_LayerNorm:
            return HexagonLayerNorm::create(backend, op);
        case OpType_Convolution:
            if (inputs.size() > 1) {
                return nullptr;
            }
            if (auto exe = HexagonTMac::create(backend, op, inputs, outputs)) {
                return exe;
            }
            return HexagonConvolution::create(backend, op);
        case OpType_ConvolutionDepthwise:
            return HexagonConvolutionDepthwise::create(backend, op);
        case OpType_Deconvolution:
            return HexagonDeconvolution::create(backend, op, inputs, outputs);
        case OpType_Scale:
            return HexagonScale::create(backend, op);
        case OpType_Pooling:
            if (outputs.size() > 1) {
                return nullptr;
            }
            return HexagonPooling::create(backend, op);
        case OpType_Raster:
            return HexagonRaster::create(backend, op);
        case OpType_While:
            return HexagonLoop::create(backend, op);
        case OpType_UnaryOp:
            return HexagonUnary::create(backend, op);
        case OpType_Reduction:
            return HexagonReduction::create(backend, op, inputs, outputs);
        case OpType_ReLU:
            return HexagonRelu::create(backend, op);
        case OpType_ReLU6:
            return HexagonRelu6::create(backend, op);
        case OpType_PReLU:
            return HexagonPRelu::create(backend, op, inputs, outputs);
        case OpType_BinaryOp:
            return HexagonBinary::create(backend, op);
        case OpType_Cast:
            return HexagonCast::create(backend, op, inputs, outputs);
        case OpType_RoPE:
            return HexagonRoPE::create(backend, op);
        case OpType_Attention:
            return HexagonAttention::create(backend, op);
        case OpType_Select:
            return HexagonSelect::create(backend, op);
        case OpType_TopKV2:
            return HexagonTopKV2::create(backend, op, inputs, outputs);
        case OpType_Softmax:
            return HexagonSoftmax::create(backend, op, inputs, outputs);
        case OpType_LSTM:
            return HexagonLSTM::create(backend, op, inputs, outputs);
        default:
            break;
    }
    return nullptr;
}

} // namespace MNN
