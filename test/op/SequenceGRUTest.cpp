
//
//  SequenceGRUTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/06/08.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited
//
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "core/MemoryFormater.h"
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;


VARP _SliceConst(VARP x, const std::vector<int>& starts, const std::vector<int>& sizes) {
    auto startVAR = _Const((const void*)starts.data(), {static_cast<int>(starts.size())}, NHWC, halide_type_of<int>());
    auto sizeVAR = _Const((const void*)sizes.data(), {static_cast<int>(sizes.size())}, NHWC, halide_type_of<int>());
    return _Slice(x, startVAR, sizeVAR);
}

VARP _SequenceGRU(const int hiddenSize, const int inputSize, const int direction, const int linearBeforeReset, VARP X, VARP W, VARP R, VARP B, VARP initial_h) {
    auto rnnGRUParam = new MNN::RNNParamT;
    rnnGRUParam->numUnits = hiddenSize;
    rnnGRUParam->isBidirectionalRNN = direction;
    rnnGRUParam->linearBeforeReset = linearBeforeReset;
    rnnGRUParam->keepAllOutputs = true;

    std::unique_ptr<OpT> gru(new OpT);
    gru->name       = "testGRU";
    gru->type       = OpType_RNNSequenceGRU;
    gru->main.type  = OpParameter_RNNParam;
    gru->main.value = rnnGRUParam;

    const int forwardParamNumber = 5;
    std::vector<VARP> gruInput(1 + forwardParamNumber * (rnnGRUParam->isBidirectionalRNN + 1));
    gruInput[0] = X;

    auto W_R = _Concat({W, R}, 2);
    // forward gru
    auto forward_W_R = _Squeeze(_SliceConst(W_R, {0, 0, 0}, {1, 3 * hiddenSize, inputSize + hiddenSize}), {0});
    forward_W_R = _Transpose(forward_W_R, {1, 0});
    gruInput[1] = _SliceConst(forward_W_R, {0, 0}, {inputSize + hiddenSize , 2 * hiddenSize}); // gateWeight
    gruInput[3] = _SliceConst(forward_W_R, {0, 2 * hiddenSize}, {inputSize + hiddenSize, hiddenSize}); // candidateWeight

    auto forward_B = _SliceConst(B, {0, 0}, {1, 6 * hiddenSize});
    gruInput[2] = _SliceConst(forward_B, {0, 0}, {1, 2 * hiddenSize}); // gateBias
    gruInput[4] = _SliceConst(forward_B, {0, 2 * hiddenSize}, {1, hiddenSize});// candidateBias
    gruInput[5] = _SliceConst(forward_B, {0, 3 * hiddenSize}, {1, 3 * hiddenSize});// recurrentBias

    // backward gru
    if(rnnGRUParam->isBidirectionalRNN) {
        auto backward_W_R = _Squeeze(_SliceConst(W_R, {1, 0, 0}, {1, 3 * hiddenSize, inputSize + hiddenSize}), {0});
        backward_W_R = _Transpose(backward_W_R, {1, 0});
        gruInput[6] = _SliceConst(backward_W_R, {0, 0}, {inputSize + hiddenSize , 2 * hiddenSize}); // backward gateWeight
        gruInput[8] = _SliceConst(backward_W_R, {0, 2 * hiddenSize}, {inputSize + hiddenSize, hiddenSize}); //backward candidateWeight
        auto backward_B = _SliceConst(B, {1, 0}, {1, 6 * hiddenSize});
        gruInput[7] = _SliceConst(backward_B, {0, 0}, {1, 2 * hiddenSize}); // backward gateBias
        gruInput[9] = _SliceConst(backward_B, {0, 2 * hiddenSize}, {1, hiddenSize});// backward candidateBias
        gruInput[10] = _SliceConst(backward_B, {0, 3 * hiddenSize}, {1, 3 * hiddenSize});// backward recurrentBias
    }
    gruInput.push_back(initial_h);

    return (Variable::create(Expr::create(gru.get(), gruInput, 2)));

}

class SequenceGRUTest : public MNNTestCase {
public:
    virtual ~SequenceGRUTest() = default;
    virtual bool run(int precision) {

        std::vector<float> inputVector;
        std::vector<float> expectedOutput;

        int seq_length = 1;
        int batch_size = 1;
        int input_size = 4;
        int hidden_size = 5;
        int direction = 0;
        int linear_before_reset = 1;
        inputVector = {.02f, .03f, .04f, .05f};
        expectedOutput = {0.13871552, 0.16327533, 0.22743227, 0.31147933, 0.40466055}; // expectedOutput is corresponding to process generating W R B init_h
        if (!runOneCase(seq_length, batch_size, input_size, hidden_size, linear_before_reset, direction, inputVector, expectedOutput)) {
          return false;
        }

        seq_length = 2;
        batch_size = 3;
        input_size = 4;
        hidden_size = 5;
        direction = 0;
        linear_before_reset = 0;
        inputVector = {0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25};
        expectedOutput = {
          0.13871552, 0.16327533, 0.22743227, 0.31147933, 0.40466055,
          0.54353297, 0.6047565 , 0.7004549 , 0.80003834, 0.90000236,
          1.        , 1.0998089 , 1.1999872 , 1.2999994 , 1.4       ,
          0.24931014, 0.20349398, 0.2405751 , 0.31553924, 0.40586844,
          0.58072   , 0.6082874 , 0.7007291 , 0.80005693, 0.9000033 ,
          1.        , 1.0996624 , 1.199979  , 1.299999  , 1.4
        };
        if (!runOneCase(seq_length, batch_size, input_size, hidden_size, linear_before_reset, direction, inputVector, expectedOutput)) {
          return false;
        }

        seq_length = 2;
        batch_size = 3;
        input_size = 4;
        hidden_size = 5;
        direction = 0;
        linear_before_reset = 1;
        inputVector = {0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25};
        expectedOutput = {
          0.13871552, 0.16327533, 0.22743227, 0.31147933, 0.40466055,
          0.54353297, 0.6047565 , 0.7004549 , 0.80003834, 0.90000236,
          1.        , 1.0998089 , 1.1999872 , 1.2999994 , 1.4       ,
          0.24931014, 0.20349398, 0.2405751 , 0.31553924, 0.40586844,
          0.58072   , 0.6082874 , 0.7007291 , 0.80005693, 0.9000033 ,
          1.        , 1.0996624 , 1.199979  , 1.299999  , 1.4
        };
        if (!runOneCase(seq_length, batch_size, input_size, hidden_size, linear_before_reset, direction, inputVector, expectedOutput)) {
          return false;
        }

        seq_length = 2;
        batch_size = 3;
        input_size = 4;
        hidden_size = 5;
        direction = 1;
        linear_before_reset = 1;
        inputVector = {0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25};
        expectedOutput = {
          0.13871552, 0.16327533, 0.22743227, 0.31147933, 0.40466055,
          0.54353297, 0.6047565 , 0.7004549 , 0.80003834, 0.90000236,
          1.        , 1.0998089 , 1.1999872 , 1.2999994 , 1.4       ,
          1.5000002 , 1.6000003 , 1.7000003 , 1.8000004 , 1.9000005 ,
          2.0000005 , 2.1000004 , 2.2000005 , 2.3000004 , 2.4000006 ,
          2.5000007 , 2.6000009 , 2.700001  , 2.800001  , 2.900001  ,
          0.24931014, 0.20349398, 0.2405751 , 0.31553924, 0.40586844,
          0.58072   , 0.6082874 , 0.7007291 , 0.80005693, 0.9000033 ,
          1.        , 1.0996624 , 1.199979  , 1.299999  , 1.4       ,
          1.5000001 , 1.6000001 , 1.7000002 , 1.8000002 , 1.9000002 ,
          2.0000002 , 2.1000001 , 2.2000003 , 2.3000002 , 2.4000003 ,
          2.5000002 , 2.6000004 , 2.7000005 , 2.8000004 , 2.9000006
        };

        if (!runOneCase(seq_length, batch_size, input_size, hidden_size, linear_before_reset, direction, inputVector, expectedOutput)) {
          return false;
        }

        return true;
    }

    bool runOneCase(const int seq_length, const int batch_size, const int input_size, const int hidden_size, const int linear_before_reset,const int direction, std::vector<float>& inputVector, std::vector<float>& expectedOutput) {

      // set input data
      auto input = _Input({seq_length, batch_size, input_size}, NCHW, halide_type_of<float>());
      ::memcpy(input->writeMap<float>(), inputVector.data(), inputVector.size() * sizeof(float));

      const int number_of_gates = 3;
      // set weight data
      auto W = _Input({direction + 1, number_of_gates * hidden_size, input_size}, NCHW, halide_type_of<float>());
      auto WPtr = W->writeMap<float>();
      for (int i = 0; i < (direction + 1) * number_of_gates * hidden_size * input_size; i++) {
        WPtr[i] = (float)i / 10;
      }

      auto R = _Input({direction + 1, number_of_gates * hidden_size, hidden_size}, NCHW, halide_type_of<float>());
      auto RPtr = R->writeMap<float>();
      for (int i = 0; i < (direction + 1) * number_of_gates * hidden_size * hidden_size; i++) {
        RPtr[i] = (float)i / 10;
      }

      auto B = _Input({direction + 1, 2 * number_of_gates * hidden_size}, NCHW, halide_type_of<float>());
      auto BPtr = B->writeMap<float>(); // W->writeMap<float>() would be wrong
      for (int i = 0; i < (direction + 1) * number_of_gates * hidden_size * 2; i++) {
        BPtr[i] = (float)i / 10;
      }

      auto initial_h = _Input({direction + 1, batch_size, hidden_size}, NCHW, halide_type_of<float>());
      auto initial_hPtr = initial_h->writeMap<float>(); // W->writeMap<float>() would be wrong
      for (int i = 0; i < (direction + 1) * batch_size * hidden_size; i++) {
        initial_hPtr[i] = (float)i / 10;
      }

      auto output = _SequenceGRU(hidden_size, input_size, direction, linear_before_reset, input, W, R, B, initial_h);
      auto gotOutput                          = output->readMap<float>();
      std::vector<int> expectedDim = {seq_length, direction + 1, batch_size, hidden_size};
      if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), expectedOutput.size(), 0.01)) {
          MNN_ERROR("SequenceGRUTest value test failed\nreal:\t expected:\n");
          formatMatrix(gotOutput, output->getInfo()->dim);
          formatMatrix(expectedOutput.data(), expectedDim);
          return false;
      }

      auto gotDim                        = output->getInfo()->dim;
      if (!checkVector<int>(gotDim.data(), expectedDim.data(), expectedDim.size(), 0)) {
          MNN_ERROR("SequenceGRUTest shape test failed!\n");
          return false;
      }
      return true;

    }
};

MNNTestSuiteRegister(SequenceGRUTest, "op/rnn/SequenceGRU");
