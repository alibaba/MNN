//
//  aoa_nlu_decoder1.cpp
//  MNN
//
//  Created by MNN on b'2021/09/06'.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited
//

#include <MNN/expr/Module.hpp>
// #define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "aoa_nlu_decoder.hpp"
#include <fstream>
#include <sstream>
#include <stdio.h>
#include<string.h>
using namespace MNN::Express;
using namespace MNN;
using namespace std;

class AOANLUDecoder1 : public AOANLUDecoder
{
public:
    AOANLUDecoder1() {};
    ~AOANLUDecoder1() {};

    virtual void getInputOutput(std::string& outputTensorName, std::vector<std::string>& inputTensorName, std::vector<std::vector<int>>& inputTensorShape, const int sequenceLength) {

    // decoder1.mnn
    // input name:encoder1_cif_output,shape:   **Tensor shape**: 1, 1, 320,
    // input name:encoder1_outputs,shape:  **Tensor shape**: 1, -1, 320,
    // input name:memory_keys_layer_0_decoder1,shape:  **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_keys_layer_1_decoder1,shape:  **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_keys_layer_2_decoder1,shape:  **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_keys_layer_3_decoder1,shape:  **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_keys_layer_4_decoder1,shape:  **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_keys_layer_5_decoder1,shape:  **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_values_layer_0_decoder1,shape:    **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_values_layer_1_decoder1,shape:    **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_values_layer_2_decoder1,shape:    **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_values_layer_3_decoder1,shape:    **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_values_layer_4_decoder1,shape:    **Tensor shape**: 1, 4, -1, 64,
    // input name:memory_values_layer_5_decoder1,shape:    **Tensor shape**: 1, 4, -1, 64,
    // input name:queries_layer_0_decoder1,shape:  **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_10_decoder1,shape: **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_11_decoder1,shape: **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_1_decoder1,shape:  **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_2_decoder1,shape:  **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_3_decoder1,shape:  **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_4_decoder1,shape:  **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_5_decoder1,shape:  **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_6_decoder1,shape:  **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_7_decoder1,shape:  **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_8_decoder1,shape:  **Tensor shape**: 1, 11, 256,
    // input name:queries_layer_9_decoder1,shape:  **Tensor shape**: 1, 11, 256,
    // input name:start_tokens_decoder1,shape: **Tensor shape**: 1,

        outputTensorName = "seq2seq/decoder_1/dense_1/BiasAdd";
        inputTensorName = {
            "encoder1_cif_output",
            "encoder1_outputs",
            "memory_keys_layer_0_decoder1",
            "memory_keys_layer_1_decoder1",
            "memory_keys_layer_2_decoder1",
            "memory_keys_layer_3_decoder1",
            "memory_keys_layer_4_decoder1",
            "memory_keys_layer_5_decoder1",
            "memory_values_layer_0_decoder1",
            "memory_values_layer_1_decoder1",
            "memory_values_layer_2_decoder1",
            "memory_values_layer_3_decoder1",
            "memory_values_layer_4_decoder1",
            "memory_values_layer_5_decoder1",
            "queries_layer_0_decoder1",
            "queries_layer_10_decoder1",
            "queries_layer_11_decoder1",
            "queries_layer_1_decoder1",
            "queries_layer_2_decoder1",
            "queries_layer_3_decoder1",
            "queries_layer_4_decoder1",
            "queries_layer_5_decoder1",
            "queries_layer_6_decoder1",
            "queries_layer_7_decoder1",
            "queries_layer_8_decoder1",
            "queries_layer_9_decoder1",
            "start_tokens_decoder1"
        };

        inputTensorShape = {
            { 1, 1, 320},
            { 1, sequenceLength, 320},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 4, sequenceLength, 64},
            { 1, 11, 256},
            { 1, 11, 256},
            { 1, 11, 256},
            { 1, 11, 256},
            { 1, 11, 256},
            { 1, 11, 256},
            { 1, 11, 256},
            { 1, 11, 256},
            { 1, 11, 256},
            { 1, 11, 256},
            { 1, 11, 256},
            { 1, 11, 256}
        };
    }
};


int main(int argc, const char* argv[]) {
    return AOANLUDecoder1().run(argc, argv);
}

