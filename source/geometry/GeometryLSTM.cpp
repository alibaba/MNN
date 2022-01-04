//
//  GeometryLSTM.cpp
//  MNN
//
//  Created by MNN on 2020/07/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/Macro.h"
#include <cmath>

namespace MNN {
static void easyUnaryEncode(const std::vector<int>& indexes, UnaryOpOperation opType, LoopParamT* loop, int length) {
    std::unique_ptr<RegionCommandT> rcmd(new RegionCommandT);
    rcmd->size = {1, 1, length};
    rcmd->indexes = indexes;
    rcmd->iterIndexes = {-1, -1};
    rcmd->steps = {0, 0};
    rcmd->view.resize(2);
    rcmd->view[1].reset(new ViewT);
    rcmd->view[1]->offset = 0;
    rcmd->view[1]->stride = {0, 0, 1};
    rcmd->view[0].reset(new ViewT);
    rcmd->view[0]->offset = 0;
    rcmd->view[0]->stride = {0, 0, 1};
    rcmd->op.reset(new OpT);
    rcmd->op->type = OpType_UnaryOp;
    rcmd->op->main.type = OpParameter_UnaryOp;
    rcmd->op->main.value = new UnaryOpT;
    rcmd->op->main.AsUnaryOp()->opType = opType;
    loop->commands.emplace_back(std::move(rcmd));
}
static void easyBinaryEncode(int length, const std::vector<int>& indexes, int opType, LoopParamT* loop, int lastOffset = 0, int outStep = 0, int outOffset = 0) {
    std::unique_ptr<RegionCommandT> rcmd(new RegionCommandT);
    rcmd->size = {1, 1, length};
    rcmd->indexes = indexes;
    rcmd->iterIndexes = {-1, -1, -1};
    rcmd->steps = {outStep, 0, 0};
    rcmd->view.resize(3);
    rcmd->view[1].reset(new ViewT);
    rcmd->view[1]->offset = 0;
    rcmd->view[1]->stride = {0, 0, 1};
    rcmd->view[2].reset(new ViewT);
    rcmd->view[2]->offset = lastOffset;
    rcmd->view[2]->stride = {0, 0, 1};
    rcmd->view[0].reset(new ViewT);
    rcmd->view[0]->offset = outOffset;
    rcmd->view[0]->stride = {0, 0, 1};
    rcmd->op.reset(new OpT);
    rcmd->op->type = OpType_BinaryOp;
    rcmd->op->main.type = OpParameter_BinaryOp;
    rcmd->op->main.value = new BinaryOpT;
    rcmd->op->main.AsBinaryOp()->opType = opType;
    loop->commands.emplace_back(std::move(rcmd));
}

class GeometryLSTM : public GeometryComputer {
public:

    void _ComputeLSTMOnnx(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Context& context,
                          CommandBuffer& res, const LSTM* lstm, OpType type) const {
        /* inputs:
        X: T The input sequences packed (and potentially padded) into one 3-D tensor with the shape of [seq_length,
        batch_size, input_size].

        W: T
        The weight tensor for the gates. Concatenation of W[iofc] and WB[iofc] (if bidirectional) along dimension 0. The
        tensor has shape [num_directions, 4*hidden_size, input_size].

        R: T
        The recurrence weight tensor. Concatenation of R[iofc] and RB[iofc] (if bidirectional) along dimension 0. This
        tensor has shape [num_directions, 4*hidden_size, hidden_size].

        B: T (optional)
        The bias tensor for input gate. [Wb[iofc] + Rb[iofc]], and [WBb[iofc] + RBb[iofc]] (if bidirectional) along
        dimension 0. This tensor has shape [num_directions, 4*hidden_size]. Optional: If not specified - assumed to be
        0.
         */
        MNN_ASSERT(inputs.size() >= 4);
        auto X_Input      = inputs[0];
        auto W            = inputs[1];
        auto R            = inputs[2];
        auto B            = inputs[3];
        Tensor* O_Init    = nullptr;
        Tensor* Cell_Init = nullptr;
        if (inputs.size() >= 5) {
            O_Init = inputs[4];
        }
        if (inputs.size() >= 6) {
            Cell_Init = inputs[5];
        }

        /** Outputs:
         Y: T (optional)
         A tensor that concats all the intermediate output values of the hidden. It has shape [seq_length,
         num_directions, batch_size, hidden_size].

         Y_h: T (optional)
         The last output value of the hidden. It has shape [num_directions, batch_size, hidden_size].

         Y_c: T (optional)
         The last output value of the cell. It has shape [num_directions, batch_size, hidden_size].
         */
        auto Y = outputs[0];
        if (outputs.size() >= 2) {
            TensorUtils::getDescribe(outputs[1])->regions.clear();
            TensorUtils::getDescribe(outputs[1])->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        }
        if (outputs.size() >= 3) {
            TensorUtils::getDescribe(outputs[2])->regions.clear();
            TensorUtils::getDescribe(outputs[2])->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        }

        auto seqLength     = X_Input->length(0);
        auto inputSize     = X_Input->length(2);
        auto batchSize     = X_Input->length(1);
        auto hiddenSize    = Y->length(3);
        auto numDirections = Y->length(1);
        auto encode = [&](Tensor* X, int direction) {
            const int N = (type == OpType_RNN ? 1 : 4);
            // FirstPart: Gate = MatMul(X, W, B) :  N * hiddenSize, seqLength * batchSize
            std::shared_ptr<Tensor> Gate(Tensor::createDevice<float>({seqLength * batchSize, N * hiddenSize}, Tensor::CAFFE));
            res.extras.emplace_back(Gate);
            {
                auto h = N * hiddenSize;
                auto e = seqLength * batchSize;
                auto l = inputSize;
                std::unique_ptr<OpT> newop(new OpT);
                newop->type = OpType_While;
                newop->main.value = new LoopParamT;
                newop->main.type = OpParameter_LoopParam;
                auto loop = newop->main.AsLoopParam();
                loop->tensorNumber = 4;
                loop->inputIndexes = {0, 1, 2};
                loop->outputIndexes = {3};
                loop->loopNumber = 1;
                std::unique_ptr<RegionCommandT> rcmd(new RegionCommandT);
                rcmd->size = {e, l, h};
                rcmd->view.resize(4);
                rcmd->view[1].reset(new ViewT);
                rcmd->view[1]->offset = 0;
                rcmd->view[1]->stride = {l, 1, 0};
                // W
                rcmd->view[2].reset(new ViewT);
                rcmd->view[2]->offset = direction * N * hiddenSize * inputSize;
                rcmd->view[2]->stride = {0, 1, l};
                // Bias
                rcmd->view[3].reset(new ViewT);
                rcmd->view[3]->offset = direction * N * hiddenSize;
                rcmd->view[3]->stride = {0, 0, 1};

                // C
                rcmd->view[0].reset(new ViewT);
                rcmd->view[0]->offset = 0;
                rcmd->view[0]->stride = {h, 0, 1};

                rcmd->indexes = {3, 0, 1, 2};// C, A, B, Bias
                rcmd->steps = {0, 0, 0, 0};
                rcmd->iterIndexes = {-1, -1, -1, -1};
                rcmd->op.reset(new OpT);
                rcmd->op->type = OpType_MatMul;
                rcmd->op->main.type = OpParameter_MatMul;
                rcmd->op->main.value = new MatMulT;
                rcmd->op->main.AsMatMul()->transposeB = true;
                rcmd->op->main.AsMatMul()->transposeA = false;
                loop->commands.emplace_back(std::move(rcmd));
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(Op::Pack(builder, newop.get()));
                auto cmd = GeometryComputerUtils::makeCommand(builder, {X, W, B}, {Gate.get()});
                res.command.emplace_back(std::move(cmd));
            }

            // SecondPart: Compute outputs
            // Initial
            std::shared_ptr<Tensor> I(Tensor::createDevice<float>({batchSize, hiddenSize}, Tensor::CAFFE));
            std::shared_ptr<Tensor> C(Tensor::createDevice<float>({batchSize, hiddenSize}, Tensor::CAFFE));
            std::shared_ptr<Tensor> F(Tensor::createDevice<float>({batchSize, hiddenSize}, Tensor::CAFFE));
            std::shared_ptr<Tensor> O(Tensor::createDevice<float>({batchSize, hiddenSize}, Tensor::CAFFE));
            std::shared_ptr<Tensor> Cell(Tensor::createDevice<float>({batchSize, hiddenSize}, Tensor::CAFFE));
            res.extras.insert(res.extras.end(), {I, C, F, O, Cell});
            // First Output
            const int I_Y = 0;
            const int I_Cell = 1;
            const int I_Gate = 3;
            const int I_I = 4;
            const int I_C = 5;
            const int I_F = 6;
            const int I_R = 7;
            const int I_HR = 8;
            const int I_Temp = 9;
            auto subEncoder = [&](int dstIndex, UnaryOpOperation unOp, int biOp, int offsetGate, int offsetHR, LoopParamT* loop) {
                // Binary
                {
                    std::unique_ptr<RegionCommandT> rcmd(new RegionCommandT);
                    rcmd->size = {1, batchSize, hiddenSize};
                    rcmd->indexes = {I_Temp, I_Gate, I_HR};
                    rcmd->iterIndexes = {-1, -1, -1};
                    rcmd->steps = {0, batchSize * hiddenSize * N, 0};
                    rcmd->view.resize(3);
                    rcmd->view[0].reset(new ViewT);
                    rcmd->view[0]->offset = 0;
                    rcmd->view[0]->stride = {hiddenSize * batchSize, hiddenSize, 1};
                    rcmd->view[1].reset(new ViewT);
                    rcmd->view[1]->offset = offsetGate;
                    rcmd->view[1]->stride = {N * hiddenSize * seqLength * batchSize, N * hiddenSize, 1};
                    rcmd->view[2].reset(new ViewT);
                    rcmd->view[2]->offset = offsetHR;
                    rcmd->view[2]->stride = {N * hiddenSize * batchSize, N * hiddenSize, 1};
                    rcmd->op.reset(new OpT);
                    rcmd->op->type = OpType_BinaryOp;
                    rcmd->op->main.type = OpParameter_BinaryOp;
                    rcmd->op->main.value = new BinaryOpT;
                    rcmd->op->main.AsBinaryOp()->opType = biOp;
                    loop->commands.emplace_back(std::move(rcmd));
                }
                // Unary
                {
                    std::unique_ptr<RegionCommandT> rcmd(new RegionCommandT);
                    rcmd->size = {1, 1, hiddenSize * batchSize};
                    rcmd->indexes = {dstIndex, I_Temp};
                    rcmd->iterIndexes = {-1, -1};
                    rcmd->steps = {0, 0};
                    rcmd->view.resize(2);
                    rcmd->view[1].reset(new ViewT);
                    rcmd->view[1]->offset = 0;
                    rcmd->view[1]->stride = {0, 0, 1};
                    rcmd->view[0].reset(new ViewT);
                    rcmd->view[0]->offset = 0;
                    rcmd->view[0]->stride = {0, 0, 1};
                    rcmd->op.reset(new OpT);
                    rcmd->op->type = OpType_UnaryOp;
                    rcmd->op->main.type = OpParameter_UnaryOp;
                    rcmd->op->main.value = new UnaryOpT;
                    rcmd->op->main.AsUnaryOp()->opType = unOp;
                    loop->commands.emplace_back(std::move(rcmd));
                }
            };
            std::shared_ptr<Tensor> HRTotal(Tensor::createDevice<float>({batchSize, N * hiddenSize}, Tensor::CAFFE));
            res.extras.emplace_back(HRTotal);
            std::shared_ptr<Tensor> Temp(Tensor::createDevice<float>({batchSize, hiddenSize}, Tensor::CAFFE));
            res.extras.emplace_back(Temp);

            auto sequenceEncode = [&](int start, int oInit, int cellInit, LoopParamT* loop) {
                int pos = start;
                int step = hiddenSize * batchSize * numDirections;
                if (direction) {
                    pos = seqLength - 1 - start;
                    step = -step;
                }
                int offset = hiddenSize * batchSize * pos * numDirections + direction * batchSize * hiddenSize;

                // Compute HR = MatMul(R, O)
                {
                    std::unique_ptr<RegionCommandT> rcmd(new RegionCommandT);
                    rcmd->size = {N * hiddenSize, hiddenSize, batchSize};
                    rcmd->indexes = {I_HR, I_R, oInit};
                    rcmd->iterIndexes = {-1, -1, -1};
                    rcmd->steps = {0, 0, step};
                    rcmd->op.reset(new OpT);
                    rcmd->op->type = OpType_MatMul;
                    rcmd->op->main.type = OpParameter_MatMul;
                    rcmd->op->main.value = new MatMulT;
                    rcmd->op->main.AsMatMul()->transposeB = true;
                    rcmd->op->main.AsMatMul()->transposeA = false;
                    rcmd->view.resize(3);
                    rcmd->view[0].reset(new ViewT);
                    rcmd->view[0]->offset = 0;
                    rcmd->view[0]->stride = {1, 0, N * hiddenSize};
                    rcmd->view[1].reset(new ViewT);
                    rcmd->view[1]->offset = direction * N * hiddenSize * hiddenSize;
                    rcmd->view[1]->stride = {batchSize, 1, 0};
                    rcmd->view[2].reset(new ViewT);
                    if (oInit != I_Y) {
                        rcmd->view[2]->offset = O->elementSize() * direction;
                    } else {
                        int pre = start - 1;
                        if (direction) {
                            pre = seqLength - 1 - pre;
                        }
                        rcmd->view[2]->offset = hiddenSize * batchSize * pre * numDirections + direction * batchSize * hiddenSize;
                    }
                    rcmd->view[2]->stride = {0, batchSize, 1};
                    loop->commands.emplace_back(std::move(rcmd));
                }

                if (type == OpType_RNN) {
                    subEncoder(I_Y, UnaryOpOperation_TANH, BinaryOpOperation_ADD, start * batchSize * hiddenSize, 0, loop);
                    loop->commands[loop->commands.size() - 1]->view[0]->offset = offset;
                    loop->commands[loop->commands.size() - 1]->steps[0] = step;
                    return;
                }
                // I = Sigmoid(WI * XI + BI + HRI)
                {
                    subEncoder(I_I, UnaryOpOperation_SIGMOID, BinaryOpOperation_ADD, start * batchSize * 4 * hiddenSize, 0, loop);
                }
                // C = tanh(WC * XC + BC + HRC)
                {
                    subEncoder(I_C, UnaryOpOperation_TANH, BinaryOpOperation_ADD, 3 * hiddenSize + start * batchSize * 4 * hiddenSize, 3 * hiddenSize, loop);
                }
                // F = Sigmoid(WF * XF + BF + HRF)
                {
                    subEncoder(I_F, UnaryOpOperation_SIGMOID, BinaryOpOperation_ADD, 2 * hiddenSize + start * batchSize * 4 * hiddenSize, 2 * hiddenSize, loop);
                }
                // Cell = I * C + F * Cell
                {
                    easyBinaryEncode(hiddenSize * batchSize, {I_Temp, I_I, I_C}, BinaryOpOperation_MUL, loop);
                    auto cellOffset = cellInit == I_Cell ? 0 : Cell->elementSize() * direction;
                    easyBinaryEncode(hiddenSize * batchSize, {I_I, I_F, cellInit}, BinaryOpOperation_MUL, loop, cellOffset);
                    easyBinaryEncode(hiddenSize * batchSize, {I_Cell, I_Temp, I_I}, BinaryOpOperation_ADD, loop);
                }
                // C = Sigmoid(WO * XO + BO + HRO)
                {
                    subEncoder(I_C, UnaryOpOperation_SIGMOID, BinaryOpOperation_ADD, 1 * hiddenSize + start * batchSize * 4 * hiddenSize, 1 * hiddenSize, loop);
                }
                // I = tanh(Cell), O = I * C
                {
                    easyUnaryEncode({I_I, I_Cell}, UnaryOpOperation_TANH, loop, hiddenSize * batchSize);
                    easyBinaryEncode(hiddenSize * batchSize, {I_Y, I_I, I_C}, BinaryOpOperation_MUL, loop, 0, step, offset);
                }
            };
            if (nullptr == O_Init && nullptr == Cell_Init) {
                std::unique_ptr<OpT> newop(new OpT);
                newop->type = OpType_While;
                newop->main.value = new LoopParamT;
                newop->main.type = OpParameter_LoopParam;
                auto loop = newop->main.AsLoopParam();
                // Y, Cell, O, Gate, I, C, F
                loop->tensorNumber = 7;
                loop->inputIndexes = {3};
                loop->outputIndexes = {0, 1, 2, 4, 5, 6};
                loop->loopNumber = 1;
                auto unaryGateEncode = [&](UnaryOpOperation unOp, int dstIndex, int index, LoopParamT* loop) {
                    std::unique_ptr<RegionCommandT> rcmd(new RegionCommandT);
                    rcmd->size = {1, batchSize, hiddenSize};
                    rcmd->indexes = {dstIndex, I_Gate};
                    rcmd->iterIndexes = {-1, -1};
                    rcmd->steps = {0, 0};
                    rcmd->view.resize(2);
                    rcmd->view[1].reset(new ViewT);
                    rcmd->view[1]->offset = index * hiddenSize;
                    rcmd->view[1]->stride = {N * hiddenSize * seqLength * batchSize, N * hiddenSize, 1};
                    rcmd->view[0].reset(new ViewT);
                    rcmd->view[0]->offset = 0;
                    rcmd->view[0]->stride = {hiddenSize * batchSize, hiddenSize, 1};
                    rcmd->op.reset(new OpT);
                    rcmd->op->type = OpType_UnaryOp;
                    rcmd->op->main.type = OpParameter_UnaryOp;
                    rcmd->op->main.value = new UnaryOpT;
                    rcmd->op->main.AsUnaryOp()->opType = unOp;
                    loop->commands.emplace_back(std::move(rcmd));
                };
                if (type == OpType_RNN) {
                    unaryGateEncode(UnaryOpOperation_TANH, I_Y, 0, loop);
                    loop->commands[loop->commands.size() - 1]->view[0]->offset = direction * (batchSize * hiddenSize) * (1 + (seqLength - 1) * numDirections);
                } else {
                    // I = Sigmoid(WI * XI + BI)
                    unaryGateEncode(UnaryOpOperation_SIGMOID, I_I, 0, loop);

                    // C = tanh(WC * XC + BC)
                    unaryGateEncode(UnaryOpOperation_TANH, I_C, 3, loop);

                    // Cell = I * C
                    easyBinaryEncode(hiddenSize * batchSize, {I_Cell, I_I, I_C}, BinaryOpOperation_MUL, loop);

                    // C = Sigmoid(WO * XO + BO)
                    unaryGateEncode(UnaryOpOperation_SIGMOID, I_C, 1, loop);

                    // I = tanh(Cell)
                    easyUnaryEncode({I_I, I_Cell}, UnaryOpOperation_TANH, loop, hiddenSize * batchSize);

                    // O = I * C
                    easyBinaryEncode(hiddenSize * batchSize, {I_Y, I_I, I_C}, BinaryOpOperation_MUL, loop, 0, 0, direction * ((batchSize * hiddenSize) + (seqLength - 1) * numDirections * batchSize * hiddenSize));
                }
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(Op::Pack(builder, newop.get()));
                auto cmd = GeometryComputerUtils::makeCommand(builder, {Gate.get()}, {Y, Cell.get(), O.get(), I.get(), C.get(), F.get()});
                res.command.emplace_back(std::move(cmd));
            } else {
                // Has Init O and Cell
                std::unique_ptr<OpT> newop(new OpT);
                newop->type = OpType_While;
                newop->main.value = new LoopParamT;
                newop->main.type = OpParameter_LoopParam;
                auto loop = newop->main.AsLoopParam();
                // Y, Cell, O, Gate, I, C, F, O_Init, Cell_Init
                const int I_OInit = 10;
                const int I_CellInit = 11;
                std::vector<Tensor*> inputs;
                if (type == OpType_RNN) { // only provide initial_h
                    loop->tensorNumber = 11;
                    loop->inputIndexes = {3, 7, 10};
                    inputs.assign({Gate.get(), R, O_Init});
                } else {
                    loop->tensorNumber = 12;
                    loop->inputIndexes = {3, 7, 10, 11};
                    inputs.assign({Gate.get(), R, O_Init, Cell_Init});
                }
                loop->outputIndexes = {0, 4, 5, 6, 8, 9, 2, 1};
                loop->loopNumber = 1;
                std::vector<Tensor*> suboutputs = {
                    Y, I.get(), C.get(), F.get(), HRTotal.get(), Temp.get(), O.get(), Cell.get()
                };
                sequenceEncode(0, I_OInit, I_CellInit, loop);
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(Op::Pack(builder, newop.get()));
                auto cmd = GeometryComputerUtils::makeCommand(builder, inputs, suboutputs);
                res.command.emplace_back(std::move(cmd));
            }
            // 1 - seqLength
            {
                std::unique_ptr<OpT> newop(new OpT);
                newop->type = OpType_While;
                newop->main.value = new LoopParamT;
                newop->main.type = OpParameter_LoopParam;
                auto loop = newop->main.AsLoopParam();
                loop->parallel = false;
                // Y, Cell, O, Gate, I, C, F, R, Temp
                loop->tensorNumber = 10;
                loop->inputIndexes = {3, 7, 2, 1};
                loop->outputIndexes = {0, 4, 5, 6, 8, 9};
                loop->loopNumber = seqLength - 1;
                std::vector<Tensor*> inputs = {
                    Gate.get(), R, O.get(), Cell.get()
                };
                std::vector<Tensor*> suboutputs = {
                    Y, I.get(), C.get(), F.get(), HRTotal.get(), Temp.get()
                };
                sequenceEncode(1, I_Y, I_Cell, loop);
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(Op::Pack(builder, newop.get()));
                auto cmd = GeometryComputerUtils::makeCommand(builder, inputs, suboutputs);
                res.command.emplace_back(std::move(cmd));
            }
            if (outputs.size() >= 2) {
                int pos = seqLength - 1;
                if (direction) {
                    pos = 0;
                }
                int offset = hiddenSize * batchSize * pos * numDirections + direction * batchSize * hiddenSize;

                TensorUtils::getDescribe(outputs[1])->regions.emplace_back(GeometryComputerUtils::makeRawAddressRef(Y, offset, O->elementSize(), O->elementSize() * direction));
            }
            if (outputs.size() >= 3) {
                TensorUtils::getDescribe(outputs[2])->regions.emplace_back(GeometryComputerUtils::makeRawAddressRef(Cell.get(), 0, Cell->elementSize(), Cell->elementSize() * direction));
            }
        };
        std::shared_ptr<Tensor> XWrap(Tensor::createDevice<float>({seqLength * batchSize, inputSize}, Tensor::CAFFE));
        GeometryComputerUtils::makeRawAddressRef(XWrap.get(), X_Input, 0, seqLength * batchSize * inputSize);
        res.extras.emplace_back(XWrap);
        encode(XWrap.get(), 0);
        if (numDirections > 1) {
            // Create Reverse X
            std::shared_ptr<Tensor> XReverse(Tensor::createDevice<float>({seqLength * batchSize, inputSize}, Tensor::CAFFE));
            res.extras.emplace_back(XReverse);
            auto des = TensorUtils::getDescribe(XReverse.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions.resize(1);
            auto& reg = des->regions[0];
            reg.size[0] = 1;
            reg.size[1] = seqLength;
            reg.size[2] = batchSize * inputSize;
            reg.src.offset = batchSize * inputSize * (seqLength-1);
            reg.src.stride[0] = 0;
            reg.src.stride[1] = -(batchSize * inputSize);
            reg.src.stride[2] = 1;
            reg.dst.offset = 0;
            reg.dst.stride[0] = 0;
            reg.dst.stride[1] = batchSize * inputSize;
            reg.dst.stride[2] = 1;
            reg.origin = X_Input;
            // Encode XReverse
            encode(XReverse.get(), 1);
        }
    }
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if (2 < inputs.size()) {
            // Onnx 's LSTM, use origin way
            _ComputeLSTMOnnx(inputs, outputs, context, res, op->main_as_LSTM(), op->type());
            return true;
        }
        if (op->type() == OpType_RNN) {
            MNN_ERROR("Navie RNN only support onnx model\n");
            return false;
        }
        // For Old version's Caffe LSTM compute
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(1 == inputs.size());
        auto& input  = inputs[0];
        auto& output = outputs[0];
        MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
        const int batch       = input->buffer().dim[0].extent; // batchSize
        const int timeSteps   = input->buffer().dim[1].extent;
        const int numFeatures = input->buffer().dim[3].extent;  // inputSize
        const int numUnits    = output->buffer().dim[3].extent; // hiddenSize
        int batchSize         = batch;
        int seqLength         = timeSteps;
        int inputSize         = numFeatures;
        int hiddenSize        = numUnits;
        auto& tensors         = context.searchConst(op);
        Tensor* W             = nullptr;
        Tensor* R             = nullptr;
        Tensor* B             = nullptr;

        if (!tensors.empty()) {
            MNN_ASSERT(3 == tensors.size());
            W = tensors[0].get();
            R = tensors[1].get();
            B = tensors[2].get();
        } else {
            auto WW   = context.allocConst(op, {1, 4 * hiddenSize, inputSize}, halide_type_of<float>(), Tensor::CAFFE);
            auto RW   = context.allocConst(op, {1, 4 * hiddenSize, hiddenSize}, halide_type_of<float>(), Tensor::CAFFE);
            auto bias = context.allocConst(op, {4 * numUnits}, halide_type_of<float>(), Tensor::CAFFE);
            if (nullptr == bias || nullptr == WW || nullptr == RW) {
                return false;
            }
            W          = WW.get();
            R          = RW.get();
            B          = bias.get();
            auto mLSTM = op->main_as_LSTM();
            // divide weight & bias if needed
            auto weightI   = mLSTM->weightI();
            auto weightH   = mLSTM->weightH();
            int weightSize = weightI->dims()->data()[0];
            // If devide, order is IFCO, else IFOC
            auto devide = weightI && !weightH && weightSize == 4 * numUnits * (numFeatures + numUnits + 2);
            {
                // Bias
                const float* biasPtr = nullptr;
                size_t biasLength    = 0;
                if (nullptr != mLSTM->bias() && nullptr != mLSTM->bias()->float32s()) {
                    biasLength = mLSTM->bias()->float32s()->size();
                    biasPtr    = mLSTM->bias()->float32s()->data();
                } else {
                    biasLength = 4 * hiddenSize;
                    biasPtr =
                        mLSTM->weightI()->float32s()->data() + 4 * numUnits * numFeatures + 4 * numUnits * numUnits;
                }
                if (4 * hiddenSize == biasLength) {
                    ::memcpy(bias->host<float>(), biasPtr, 4 * hiddenSize * sizeof(float));
                } else {
                    MNN_ASSERT(8 * hiddenSize == biasLength);
                    auto dst = bias->host<float>();
                    auto src = biasPtr;
                    for (int i = 0; i < 4 * hiddenSize; ++i) {
                        dst[i] = src[i] + src[i + 4 * hiddenSize];
                    }
                }
                auto destBias = bias->host<float>();
                if (devide) {
                    // IFCO -> IOFC
                    auto bf = destBias + 1 * hiddenSize;
                    auto bc = destBias + 2 * hiddenSize;
                    auto bo = destBias + 3 * hiddenSize;
                    for (int i = 0; i < hiddenSize; ++i) {
                        auto temp = bc[i];
                        bc[i]     = bf[i];
                        bf[i]     = bo[i];
                        bo[i]     = temp;
                    }
                } else {
                    // IFOC -> IOFC
                    auto bf = destBias + 1 * hiddenSize;
                    auto bo = destBias + 2 * hiddenSize;
                    for (int i = 0; i < hiddenSize; ++i) {
                        auto temp = bo[i];
                        bo[i]     = bf[i];
                        bf[i]     = temp;
                    }
                }
            }

            // gate space
            // cell space
            if (mLSTM->weightH()) {
                MNN_ASSERT(mLSTM->weightH()->float32s()->size() == numUnits * numUnits * 4);
            }
            // W: IFOC -> IOFC
            {
                auto srcWPtr = mLSTM->weightI()->float32s()->data();
                auto dI      = W->host<float>() + 0 * hiddenSize * inputSize;
                auto dC      = W->host<float>() + 3 * hiddenSize * inputSize;
                auto dF      = W->host<float>() + 2 * hiddenSize * inputSize;
                auto dO      = W->host<float>() + 1 * hiddenSize * inputSize;

                auto sI = srcWPtr + 0 * hiddenSize * inputSize;
                auto sF = srcWPtr + 1 * hiddenSize * inputSize;
                auto sO = srcWPtr + 3 * hiddenSize * inputSize;
                auto sC = srcWPtr + 2 * hiddenSize * inputSize;
                if (!devide) {
                    sI = srcWPtr + 0 * hiddenSize * inputSize;
                    sF = srcWPtr + 1 * hiddenSize * inputSize;
                    sO = srcWPtr + 2 * hiddenSize * inputSize;
                    sC = srcWPtr + 3 * hiddenSize * inputSize;
                }

                ::memcpy(dI, sI, hiddenSize * inputSize * sizeof(float));
                ::memcpy(dF, sF, hiddenSize * inputSize * sizeof(float));
                ::memcpy(dC, sC, hiddenSize * inputSize * sizeof(float));
                ::memcpy(dO, sO, hiddenSize * inputSize * sizeof(float));
            }
            // R: IFOC -> IOFC
            {
                auto srcHPtr = mLSTM->weightI()->float32s()->data() + 4 * numUnits * numFeatures;
                if (!devide) {
                    srcHPtr = mLSTM->weightH()->float32s()->data();
                }
                auto dI = R->host<float>() + 0 * hiddenSize * hiddenSize;
                auto dC = R->host<float>() + 3 * hiddenSize * hiddenSize;
                auto dF = R->host<float>() + 2 * hiddenSize * hiddenSize;
                auto dO = R->host<float>() + 1 * hiddenSize * hiddenSize;

                auto sI = srcHPtr + 0 * hiddenSize * hiddenSize;
                auto sC = srcHPtr + 2 * hiddenSize * hiddenSize;
                auto sF = srcHPtr + 1 * hiddenSize * hiddenSize;
                auto sO = srcHPtr + 3 * hiddenSize * hiddenSize;
                if (!devide) {
                    sI = srcHPtr + 0 * hiddenSize * hiddenSize;
                    sC = srcHPtr + 3 * hiddenSize * hiddenSize;
                    sF = srcHPtr + 1 * hiddenSize * hiddenSize;
                    sO = srcHPtr + 2 * hiddenSize * hiddenSize;
                }
                ::memcpy(dI, sI, hiddenSize * hiddenSize * sizeof(float));
                ::memcpy(dF, sF, hiddenSize * hiddenSize * sizeof(float));
                ::memcpy(dC, sC, hiddenSize * hiddenSize * sizeof(float));
                ::memcpy(dO, sO, hiddenSize * hiddenSize * sizeof(float));
            }
        }

        std::shared_ptr<Tensor> tempInput(Tensor::createDevice<float>({seqLength, batchSize, inputSize}, Tensor::CAFFE));
        {
            // Transpose for input
            auto des        = TensorUtils::getDescribe(tempInput.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions.resize(1);
            auto& reg         = des->regions[0];
            reg.size[0]       = seqLength;
            reg.size[1]       = batchSize;
            reg.size[2]       = inputSize;
            reg.dst.offset    = 0;
            reg.dst.stride[0] = batchSize * inputSize;
            reg.dst.stride[1] = inputSize;
            reg.dst.stride[2] = 1;
            reg.src.offset    = 0;
            reg.src.stride[0] = inputSize;
            reg.src.stride[1] = inputSize * seqLength;
            reg.src.stride[2] = 1;
            reg.origin        = inputs[0];
        }
        std::shared_ptr<Tensor> tempOutput(Tensor::createDevice<float>({seqLength, 1, batchSize, hiddenSize}, Tensor::CAFFE));
        _ComputeLSTMOnnx({tempInput.get(), W, R, B}, {tempOutput.get()}, context, res, op->main_as_LSTM(), op->type());
        res.extras.emplace_back(tempInput);
        res.extras.emplace_back(tempOutput);
        {
            // Transpose for output
            auto des = TensorUtils::getDescribe(output);
            des->regions.resize(1);
            des->memoryType   = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            auto& reg         = des->regions[0];
            reg.origin        = tempOutput.get();
            reg.size[0]       = seqLength;
            reg.size[1]       = batchSize;
            reg.size[2]       = hiddenSize;
            reg.dst.offset    = 0;
            reg.src.stride[0] = batchSize * hiddenSize;
            reg.src.stride[1] = hiddenSize;
            reg.src.stride[2] = 1;
            reg.dst.offset    = 0;
            reg.dst.stride[0] = hiddenSize;
            reg.dst.stride[1] = hiddenSize * seqLength;
            reg.dst.stride[2] = 1;
        }
        return true;
    }
};

// LSTMBlockCell
class GeometryLSTMBlockCell : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        /*
         shapes:
         x: [batchSize, inputSize]
         cs_prev, i, cs, f, o, ci, co, h: [batchSize, hiddenSize]
         wci, wcf, wco: [hiddenSize]
         w: [inputSize + hiddenSize, 4 * hiddenSize]
         b: [4 * hiddenSize]
         */
        // inputs
        auto x       = inputs[0];
        auto cs_prev = inputs[1];
        auto h_prev  = inputs[2];
        auto w       = inputs[3];
        auto wci     = inputs[4];
        auto wcf     = inputs[5];
        auto wco     = inputs[6];
        auto b       = inputs[7];
        // outputs
        auto i       = outputs[0];
        auto cs      = outputs[1];
        auto f       = outputs[2];
        auto o       = outputs[3];
        auto ci      = outputs[4];
        auto co      = outputs[5];
        auto h       = outputs[6];
        int batchSize  = x->length(0);
        int inputSize  = x->length(1);
        int hiddenSize = h_prev->length(1);
        // params
        auto param = op->main_as_LSTMBlockCell();
        float cell_clip = param->cell_clip();
        float forget_bias = param->forget_bias();
        bool use_peephole = param->use_peephole();
        // xh = [x, h_prev]
        std::shared_ptr<Tensor> xh(Tensor::createDevice<float>({batchSize, inputSize + hiddenSize}));
        {
            auto xhDes        = TensorUtils::getDescribe(xh.get());
            xhDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            xhDes->regions.resize(2);
            xhDes->regions[0].origin = x;
            xhDes->regions[0].size[0] = batchSize;
            xhDes->regions[0].size[1] = inputSize;
            xhDes->regions[0].src.stride[0] = inputSize;
            xhDes->regions[0].dst.stride[0] = inputSize + hiddenSize;
            xhDes->regions[1].origin = h_prev;
            xhDes->regions[1].size[0] = batchSize;
            xhDes->regions[1].size[1] = hiddenSize;
            xhDes->regions[1].dst.offset = inputSize;
            xhDes->regions[1].src.stride[0] = hiddenSize;
            xhDes->regions[1].dst.stride[0] = inputSize + hiddenSize;
            res.extras.emplace_back(xh);
        }
        // icfo = xh * w + b
        std::shared_ptr<Tensor> icfo(Tensor::createDevice<float>({batchSize, 4 * hiddenSize}));
        {
            res.command.emplace_back(GeometryComputerUtils::makeMatMul(xh.get(), w, icfo.get(), b, false, false));
            res.extras.emplace_back(icfo);
        }
        // [i, ci, f, o] = icfo
        std::shared_ptr<Tensor> iTensor(Tensor::createDevice<float>({batchSize, hiddenSize}));
        std::shared_ptr<Tensor> fTensor(Tensor::createDevice<float>({batchSize, hiddenSize}));
        std::shared_ptr<Tensor> ciTensor(Tensor::createDevice<float>({batchSize, hiddenSize}));
        std::shared_ptr<Tensor> oTensor(Tensor::createDevice<float>({batchSize, hiddenSize}));
        {
            // using ICFO order
            // ref: https://github.com/tensorflow/tensorflow/blob/dec8e0b11f4f87693b67e125e67dfbc68d26c205/tensorflow/core/kernels/rnn/lstm_ops.h
            std::vector<std::shared_ptr<Tensor>> ifcioArray = { iTensor, ciTensor, fTensor, oTensor };
            // std::vector<std::shared_ptr<Tensor>> ifcioArray = { iTensor, fTensor, ciTensor, oTensor };
            for (int n = 0; n < 4; n++) {
                auto des        = TensorUtils::getDescribe(ifcioArray[n].get());
                des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->regions.resize(1);
                des->regions[0].origin = icfo.get();
                des->regions[0].size[0] = batchSize;
                des->regions[0].size[1] = hiddenSize;
                des->regions[0].src.offset = n * hiddenSize;
                des->regions[0].src.stride[0] = 4 * hiddenSize;
                des->regions[0].dst.stride[0] = hiddenSize;
            }
            res.extras.insert(res.extras.end(), { iTensor, fTensor, ciTensor, oTensor });
        }
        // f = f + forget_bias
        std::shared_ptr<Tensor> ffTensor(Tensor::createDevice<float>({batchSize, hiddenSize}));
        {
            auto constTensor = context.allocConst(op, {}, halide_type_of<float>());
            constTensor->host<float>()[0] = forget_bias;
            res.extras.emplace_back(ffTensor);
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, fTensor.get(), constTensor.get(), ffTensor.get()));
        }
        // if not use_peephole:
        //      wci = wcf = wco = 0
        if (!use_peephole) {
            auto zeroTensor = context.allocConst(op, {}, halide_type_of<float>());
            zeroTensor->host<float>()[0] = 0;
            wci = zeroTensor.get();
            wcf = wci;
            wco = wci;
        }
        if (use_peephole) {
            // i = sigmoid(cs_prev * wci + i)
            // f = sigmoid(cs_prev * wcf + f)
            // ci = tanh(ci)
            std::shared_ptr<Tensor> cs_prev_wci(Tensor::createDevice<float>({batchSize, hiddenSize}));
            std::shared_ptr<Tensor> cs_prev_wcf(Tensor::createDevice<float>({batchSize, hiddenSize}));
            std::shared_ptr<Tensor> cs_prev_wci_i(Tensor::createDevice<float>({batchSize, hiddenSize}));
            std::shared_ptr<Tensor> cs_prev_wcf_f(Tensor::createDevice<float>({batchSize, hiddenSize}));
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, cs_prev, wci, cs_prev_wci.get()));
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, cs_prev, wcf, cs_prev_wcf.get()));
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, cs_prev_wci.get(), iTensor.get(), cs_prev_wci_i.get()));
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, cs_prev_wcf.get(), ffTensor.get(), cs_prev_wcf_f.get()));
            res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, cs_prev_wci_i.get(), i));
            res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, cs_prev_wcf_f.get(), f));
            res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_TANH, ciTensor.get(), ci));
            res.extras.insert(res.extras.end(), { cs_prev_wci, cs_prev_wcf, cs_prev_wci_i, cs_prev_wcf_f });
        } else {
            // i = sigmoid(i)
            // f = sigmoid(f)
            // ci = tanh(ci)
            res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, iTensor.get(), i));
            res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, ffTensor.get(), f));
            res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_TANH, ciTensor.get(), ci));
        }

        Tensor* csTmp = cs;
        if (cell_clip > 0) {
            std::shared_ptr<Tensor> csTensor(Tensor::createDevice<float>({batchSize, hiddenSize}));
            csTmp = csTensor.get();
            res.extras.emplace_back(csTensor);
        }
        // cs = ci .* i + cs_prev .* f
        std::shared_ptr<Tensor> ci_i(Tensor::createDevice<float>({batchSize, hiddenSize}));
        std::shared_ptr<Tensor> cs_prev_f(Tensor::createDevice<float>({batchSize, hiddenSize}));
        {
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, ci, i, ci_i.get()));
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, cs_prev, f, cs_prev_f.get()));
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, ci_i.get(), cs_prev_f.get(), csTmp));
            res.extras.insert(res.extras.end(), { ci_i, cs_prev_f });
        }
        if (cell_clip > 0) {
            // cs = clip(cs, cell_clip)
            std::shared_ptr<Tensor> upValue(Tensor::createDevice<float>({batchSize, hiddenSize}));
            std::shared_ptr<Tensor> downValue(Tensor::createDevice<float>({batchSize, hiddenSize}));
            std::shared_ptr<Tensor> midTensor(Tensor::createDevice<float>({batchSize, hiddenSize}));
            auto posConst = context.allocConst(op, {}, halide_type_of<float>());
            posConst->host<float>()[0] = std::fabs(cell_clip);
            auto negConst = context.allocConst(op, {}, halide_type_of<float>());
            negConst->host<float>()[0] = -std::fabs(cell_clip);
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_GREATER, csTmp, posConst.get(), upValue.get()));
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_LESS, csTmp, negConst.get(), downValue.get()));
            flatbuffers::FlatBufferBuilder builder;
            OpBuilder opBuilder(builder);
            opBuilder.add_type(OpType_Select);
            builder.Finish(opBuilder.Finish());
            res.command.emplace_back(GeometryComputerUtils::makeCommand(builder, {upValue.get(), posConst.get(), csTmp}, {midTensor.get()}));
            res.command.emplace_back(GeometryComputerUtils::makeCommand(builder, {downValue.get(), negConst.get(), midTensor.get()}, {cs}));
            res.extras.insert(res.extras.end(), { upValue, downValue, midTensor });
        }
        if (use_peephole) {
            // o = sigmoid(cs * wco + o)
            std::shared_ptr<Tensor> cs_wco(Tensor::createDevice<float>({batchSize, hiddenSize}));
            std::shared_ptr<Tensor> cs_wco_o(Tensor::createDevice<float>({batchSize, hiddenSize}));
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, cs, wco, cs_wco.get()));
            res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, cs_wco.get(), oTensor.get(), cs_wco_o.get()));
            res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, cs_wco_o.get(), o));
            res.extras.insert(res.extras.end(), { cs_wco, cs_wco_o });
        } else {
            // o = sigmoid(o)
            res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, oTensor.get(), o));
        }
        // co = tanh(cs)
        // h = co .* o
        res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_TANH, cs, co));
        res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, co, o, h));
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryLSTM);
    GeometryComputer::registerGeometryComputer(comp, {OpType_LSTM, OpType_RNN}, Runtime::Compiler_Loop);
    std::shared_ptr<GeometryComputer> comp1(new GeometryLSTMBlockCell);
    GeometryComputer::registerGeometryComputer(comp1, {OpType_LSTMBlockCell});
}

REGISTER_GEOMETRY(GeometryLSTM, _create);
} // namespace MNN
