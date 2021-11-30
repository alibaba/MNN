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

class AOANLUDecoder2 : public AOANLUDecoder
{
public:
    AOANLUDecoder2() {};
    ~AOANLUDecoder2() {};

    virtual void getInputOutput(std::string& outputTensorName, std::vector<std::string>& inputTensorName, std::vector<std::vector<int>>& inputTensorShape, const int sequenceLength) {

    // decoder2.mnn
    // input name:encoder2_cif_output,shape:   **Tensor shape**: 1, 1, 320,
    // input name:encoder2_outputs,shape:  **Tensor shape**: 1, -1, 320,
    // input name:memory_keys_layer_0_decoder2,shape:  **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_keys_layer_1_decoder2,shape:  **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_keys_layer_2_decoder2,shape:  **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_keys_layer_3_decoder2,shape:  **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_keys_layer_4_decoder2,shape:  **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_keys_layer_5_decoder2,shape:  **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_values_layer_0_decoder2,shape:    **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_values_layer_1_decoder2,shape:    **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_values_layer_2_decoder2,shape:    **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_values_layer_3_decoder2,shape:    **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_values_layer_4_decoder2,shape:    **Tensor shape**: 1, 4, -1, 80,
    // input name:memory_values_layer_5_decoder2,shape:    **Tensor shape**: 1, 4, -1, 80,
    // input name:queries_layer_0_decoder2,shape:  **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_10_decoder2,shape: **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_11_decoder2,shape: **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_1_decoder2,shape:  **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_2_decoder2,shape:  **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_3_decoder2,shape:  **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_4_decoder2,shape:  **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_5_decoder2,shape:  **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_6_decoder2,shape:  **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_7_decoder2,shape:  **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_8_decoder2,shape:  **Tensor shape**: 1, 11, 320,
    // input name:queries_layer_9_decoder2,shape:  **Tensor shape**: 1, 11, 320,
    // input name:start_tokens_decoder2,shape: **Tensor shape**: 1,

    outputTensorName = "seq2seq/decoder2/dense/BiasAdd";
    inputTensorName = {
        "encoder2_cif_output",
        "encoder2_outputs",
        "memory_keys_layer_0_decoder2",
        "memory_keys_layer_1_decoder2",
        "memory_keys_layer_2_decoder2",
        "memory_keys_layer_3_decoder2",
        "memory_keys_layer_4_decoder2",
        "memory_keys_layer_5_decoder2",
        "memory_values_layer_0_decoder2",
        "memory_values_layer_1_decoder2",
        "memory_values_layer_2_decoder2",
        "memory_values_layer_3_decoder2",
        "memory_values_layer_4_decoder2",
        "memory_values_layer_5_decoder2",
        "queries_layer_0_decoder2",
        "queries_layer_10_decoder2",
        "queries_layer_11_decoder2",
        "queries_layer_1_decoder2",
        "queries_layer_2_decoder2",
        "queries_layer_3_decoder2",
        "queries_layer_4_decoder2",
        "queries_layer_5_decoder2",
        "queries_layer_6_decoder2",
        "queries_layer_7_decoder2",
        "queries_layer_8_decoder2",
        "queries_layer_9_decoder2",
        "start_tokens_decoder2"
    };

    inputTensorShape = {
        {1, 1, 320},
        {1, sequenceLength, 320},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 4, sequenceLength, 80},
        {1, 11, 320},
        {1, 11, 320},
        {1, 11, 320},
        {1, 11, 320},
        {1, 11, 320},
        {1, 11, 320},
        {1, 11, 320},
        {1, 11, 320},
        {1, 11, 320},
        {1, 11, 320},
        {1, 11, 320},
        {1, 11, 320}
    };
        return;
    }
};



int main(int argc, const char* argv[]) {
    return AOANLUDecoder2().run(argc, argv);
}

