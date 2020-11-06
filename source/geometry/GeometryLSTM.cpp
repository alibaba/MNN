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
namespace MNN {
class GeometryLSTM : public GeometryComputer {
public:
    void _ComputeLSTMOnnx(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Context& context,
                          CommandBuffer& res, const LSTM* lstm) const {
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
        // Output contain seqLength * numDirection's region
        auto outputDes        = TensorUtils::getDescribe(Y);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outputDes->regions.resize(seqLength * numDirections);

        auto encode = [&](Tensor* X, int direction) {
            // FirstPart: Gate = MatMul(X, W, B) :  4 * hiddenSize, seqLength * batchSize
            std::shared_ptr<Tensor> Gate(Tensor::createDevice<float>({4 * hiddenSize, seqLength * batchSize}));
            res.extras.emplace_back(Gate);
            std::shared_ptr<Tensor> Bias(Tensor::createDevice<float>({4 * hiddenSize}));
            res.extras.emplace_back(Bias);
            GeometryComputerUtils::makeRawAddressRef(Bias.get(), B, direction * 4 * hiddenSize, 4 * hiddenSize);
            {
                std::shared_ptr<Tensor> WWrap(Tensor::createDevice<float>({4 * hiddenSize, inputSize}));
                std::shared_ptr<Tensor> GateWrap(Tensor::createDevice<float>({seqLength * batchSize, 4 * hiddenSize}));
                GeometryComputerUtils::makeRawAddressRef(WWrap.get(), W, direction * 4 * hiddenSize * inputSize, 4 * hiddenSize * inputSize);
                res.command.emplace_back(
                                         GeometryComputerUtils::makeMatMul(X, WWrap.get(), GateWrap.get(), Bias.get(), false, true));
                res.extras.emplace_back(WWrap);
                res.extras.emplace_back(GateWrap);
                auto gateDes        = TensorUtils::getDescribe(Gate.get());
                gateDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                gateDes->regions.resize(1);
                gateDes->regions[0].origin        = GateWrap.get();
                gateDes->regions[0].size[0]       = 1;
                gateDes->regions[0].size[1]       = 4 * hiddenSize;
                gateDes->regions[0].size[2]       = seqLength * batchSize;
                gateDes->regions[0].src.offset    = 0;
                gateDes->regions[0].src.stride[0] = 1;
                gateDes->regions[0].src.stride[1] = 1;
                gateDes->regions[0].src.stride[2] = 4 * hiddenSize;
                gateDes->regions[0].dst.offset    = 0;
                gateDes->regions[0].dst.stride[0] = 1;
                gateDes->regions[0].dst.stride[1] = seqLength * batchSize;
                gateDes->regions[0].dst.stride[2] = 1;
            }

            // SecondPart: Compute outputs
            std::shared_ptr<Tensor> RWrap(Tensor::createDevice<float>({4 * hiddenSize, hiddenSize}));
            res.extras.emplace_back(RWrap);
            GeometryComputerUtils::makeRawAddressRef(RWrap.get(), R, direction * 4 * hiddenSize * hiddenSize, 4 * hiddenSize * hiddenSize);

            // Initial
            std::shared_ptr<Tensor> I(Tensor::createDevice<float>({hiddenSize, batchSize}));
            std::shared_ptr<Tensor> C(Tensor::createDevice<float>({hiddenSize, batchSize}));
            std::shared_ptr<Tensor> F(Tensor::createDevice<float>({hiddenSize, batchSize}));
            std::shared_ptr<Tensor> O(Tensor::createDevice<float>({hiddenSize, batchSize}));
            std::shared_ptr<Tensor> Cell(Tensor::createDevice<float>({hiddenSize, batchSize}));
            res.extras.insert(res.extras.end(), {I, C, F, O, Cell});
            int seqStart = 0;
            if (O_Init == nullptr && Cell_Init == nullptr) {
                seqStart = 1;
                // IO: WI * XI + BI
                std::shared_ptr<Tensor> IO(Tensor::createDevice<float>({hiddenSize, batchSize}));
                GeometryComputerUtils::makeSliceRef(IO.get(), Gate.get(), {1, 4 * hiddenSize, seqLength * batchSize},
                                                    {0, 0, 0}, {1, hiddenSize, batchSize});
                std::shared_ptr<Tensor> CO(Tensor::createDevice<float>({hiddenSize, batchSize}));
                GeometryComputerUtils::makeSliceRef(CO.get(), Gate.get(), {1, 4 * hiddenSize, seqLength * batchSize},
                                                    {0, 3 * hiddenSize, 0}, {1, hiddenSize, batchSize});
                std::shared_ptr<Tensor> OO(Tensor::createDevice<float>({hiddenSize, batchSize}));
                GeometryComputerUtils::makeSliceRef(OO.get(), Gate.get(), {1, 4 * hiddenSize, seqLength * batchSize},
                                                    {0, 1 * hiddenSize, 0}, {1, hiddenSize, batchSize});
                res.extras.insert(res.extras.end(), {IO, CO, OO});

                // I = Sigmoid(WI * XI + BI)
                res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, IO.get(), I.get()));
                // C = tanh(WC * XC + BC)
                res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_TANH, CO.get(), C.get()));
                // Cell = I * C
                res.command.emplace_back(
                    GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, I.get(), C.get(), Cell.get()));
                // C = Sigmoid(WO * XO + BO)
                res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, OO.get(), C.get()));
                // I = tanh(Cell), O = I * C
                res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_TANH, Cell.get(), I.get()));
                res.command.emplace_back(
                    GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, I.get(), C.get(), O.get()));

                // Transpose
                auto& outReg         = outputDes->regions[0 + direction * seqLength];
                outReg.origin        = O.get();
                outReg.size[0]       = 1;
                outReg.size[1]       = batchSize;
                outReg.size[2]       = hiddenSize;
                outReg.dst.offset    = direction * ((batchSize * hiddenSize) + (seqLength - 1) * numDirections * batchSize * hiddenSize);
                outReg.dst.stride[0] = 0;
                outReg.dst.stride[1] = hiddenSize;
                outReg.dst.stride[2] = 1;
                outReg.src.offset    = 0;
                outReg.src.stride[0] = 0;
                outReg.src.stride[1] = 1;
                outReg.src.stride[2] = batchSize;
            }
            for (int t = seqStart; t < seqLength; ++t) {
                if (0 == t) {
                    GeometryComputerUtils::makeRawAddressRef(O.get(), O_Init, O->elementSize() * direction, O->elementSize());
                    GeometryComputerUtils::makeRawAddressRef(Cell.get(), Cell_Init, Cell->elementSize() * direction, Cell->elementSize());
                }
                std::shared_ptr<Tensor> HRTotal(Tensor::createDevice<float>({4 * hiddenSize, batchSize}));
                std::shared_ptr<Tensor> HRI(Tensor::createDevice<float>({hiddenSize, batchSize}));
                std::shared_ptr<Tensor> HRC(Tensor::createDevice<float>({hiddenSize, batchSize}));
                std::shared_ptr<Tensor> HRO(Tensor::createDevice<float>({hiddenSize, batchSize}));
                std::shared_ptr<Tensor> HRF(Tensor::createDevice<float>({hiddenSize, batchSize}));
                std::shared_ptr<Tensor> Temp(Tensor::createDevice<float>({hiddenSize, batchSize}));
                res.extras.insert(res.extras.end(), {HRTotal, HRI, HRC, HRF, HRO, Temp});

                GeometryComputerUtils::makeSliceRef(HRI.get(), HRTotal.get(), {1, 4 * hiddenSize, batchSize}, {0, 0, 0},
                                                    {1, hiddenSize, batchSize});
                GeometryComputerUtils::makeSliceRef(HRO.get(), HRTotal.get(), {1, 4 * hiddenSize, batchSize},
                                                    {0, 1 * hiddenSize, 0}, {1, hiddenSize, batchSize});
                GeometryComputerUtils::makeSliceRef(HRF.get(), HRTotal.get(), {1, 4 * hiddenSize, batchSize},
                                                    {0, 2 * hiddenSize, 0}, {1, hiddenSize, batchSize});
                GeometryComputerUtils::makeSliceRef(HRC.get(), HRTotal.get(), {1, 4 * hiddenSize, batchSize},
                                                    {0, 3 * hiddenSize, 0}, {1, hiddenSize, batchSize});
                // HRTotal = MatMul(O, RWrap)
                res.command.emplace_back(
                    GeometryComputerUtils::makeMatMul(RWrap.get(), O.get(), HRTotal.get(), nullptr, false, false));

                std::shared_ptr<Tensor> newO(Tensor::createDevice<float>({hiddenSize, batchSize}));
                // Transpose
                auto& outReg         = outputDes->regions[t + direction * seqLength];
                outReg.origin        = newO.get();
                outReg.size[0]       = 1;
                outReg.size[1]       = batchSize;
                outReg.size[2]       = hiddenSize;
                int pos = t;
                if (direction) {
                    pos = seqLength - t - 1;
                }
                outReg.dst.offset    = hiddenSize * batchSize * pos * numDirections + direction * batchSize * hiddenSize;
                outReg.dst.stride[0] = 0;
                outReg.dst.stride[1] = hiddenSize;
                outReg.dst.stride[2] = 1;
                outReg.src.offset    = 0;
                outReg.src.stride[0] = 0;
                outReg.src.stride[1] = 1;
                outReg.src.stride[2] = batchSize;

                // IO: WI * XI + BI
                std::shared_ptr<Tensor> IO(Tensor::createDevice<float>({hiddenSize, batchSize}));
                GeometryComputerUtils::makeSliceRef(IO.get(), Gate.get(), {1, 4 * hiddenSize, seqLength * batchSize},
                                                    {0, 0, t * batchSize}, {1, hiddenSize, batchSize});
                std::shared_ptr<Tensor> CO(Tensor::createDevice<float>({hiddenSize, batchSize}));
                GeometryComputerUtils::makeSliceRef(CO.get(), Gate.get(), {1, 4 * hiddenSize, seqLength * batchSize},
                                                    {0, 3 * hiddenSize, t * batchSize}, {1, hiddenSize, batchSize});
                std::shared_ptr<Tensor> FO(Tensor::createDevice<float>({hiddenSize, batchSize}));
                GeometryComputerUtils::makeSliceRef(FO.get(), Gate.get(), {1, 4 * hiddenSize, seqLength * batchSize},
                                                    {0, 2 * hiddenSize, t * batchSize}, {1, hiddenSize, batchSize});
                std::shared_ptr<Tensor> OO(Tensor::createDevice<float>({hiddenSize, batchSize}));
                GeometryComputerUtils::makeSliceRef(OO.get(), Gate.get(), {1, 4 * hiddenSize, seqLength * batchSize},
                                                    {0, 1 * hiddenSize, t * batchSize}, {1, hiddenSize, batchSize});
                res.extras.insert(res.extras.end(), {IO, CO, FO, OO, newO});

                // I = Sigmoid(WI * XI + BI + HRI)
                res.command.emplace_back(
                    GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, IO.get(), HRI.get(), Temp.get()));
                res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, Temp.get(), I.get()));
                // C = tanh(WC * XC + BC + HRC)
                res.command.emplace_back(
                    GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, CO.get(), HRC.get(), Temp.get()));
                res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_TANH, Temp.get(), C.get()));

                // F = Sigmoid(WF * XF + BF + HRF)
                res.command.emplace_back(
                    GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, FO.get(), HRF.get(), Temp.get()));
                res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, Temp.get(), F.get()));

                // Cell = I * C + F * Cell
                res.command.emplace_back(
                    GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, I.get(), C.get(), Temp.get()));
                res.command.emplace_back(
                    GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, F.get(), Cell.get(), I.get()));
                if (0 == seqStart) {
                    std::shared_ptr<Tensor> newCell(Tensor::createDevice<float>({hiddenSize, batchSize}));
                    Cell = newCell;
                    res.extras.emplace_back(newCell);
                }
                res.command.emplace_back(
                    GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, I.get(), Temp.get(), Cell.get()));

                // C = Sigmoid(WO * XO + BO + HRO)
                res.command.emplace_back(
                    GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, OO.get(), HRO.get(), Temp.get()));
                res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_SIGMOID, Temp.get(), C.get()));
                // I = tanh(Cell), O = I * C
                res.command.emplace_back(GeometryComputerUtils::makeUnary(UnaryOpOperation_TANH, Cell.get(), I.get()));
                res.command.emplace_back(
                    GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, I.get(), C.get(), newO.get()));
                O = newO;
            }
            if (outputs.size() >= 2) {
                TensorUtils::getDescribe(outputs[1])->regions.emplace_back(GeometryComputerUtils::makeRawAddressRef(O.get(), 0, O->elementSize(), O->elementSize() * direction));
            }
            if (outputs.size() >= 3) {
                TensorUtils::getDescribe(outputs[2])->regions.emplace_back(GeometryComputerUtils::makeRawAddressRef(Cell.get(), 0, Cell->elementSize(), Cell->elementSize() * direction));
            }
        };
        std::shared_ptr<Tensor> XWrap(Tensor::createDevice<float>({seqLength * batchSize, inputSize}));
        GeometryComputerUtils::makeRawAddressRef(XWrap.get(), X_Input, 0, seqLength * batchSize * inputSize);
        res.extras.emplace_back(XWrap);
        encode(XWrap.get(), 0);
        if (numDirections > 1) {
            // Create Reverse X
            std::shared_ptr<Tensor> XReverse(Tensor::createDevice<float>({seqLength * batchSize, inputSize}));
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
            _ComputeLSTMOnnx(inputs, outputs, context, res, op->main_as_LSTM());
            return true;
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
            auto WW   = context.allocConst(op, {1, 4 * hiddenSize, inputSize}, halide_type_of<float>());
            auto RW   = context.allocConst(op, {1, 4 * hiddenSize, hiddenSize}, halide_type_of<float>());
            auto bias = context.allocConst(op, {4 * numUnits}, halide_type_of<float>());
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

        std::shared_ptr<Tensor> tempInput(Tensor::createDevice<float>({seqLength, batchSize, inputSize}));
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
        std::shared_ptr<Tensor> tempOutput(Tensor::createDevice<float>({seqLength, 1, batchSize, hiddenSize}));
        _ComputeLSTMOnnx({tempInput.get(), W, R, B}, {tempOutput.get()}, context, res, op->main_as_LSTM());
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
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryLSTM);
    GeometryComputer::registerGeometryComputer(comp, {OpType_LSTM});
}

REGISTER_GEOMETRY(GeometryLSTM, _create);

} // namespace MNN
