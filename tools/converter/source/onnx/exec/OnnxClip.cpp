//
//  OnnxClip.cpp
//  MNNConverter
//
//  Created by MNN on 2020/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include "../OnnxUtils.hpp"
#include "onnx.pb.h"
static void _removeWeight(int argc, const char* argv[]) {
    auto modelName = argv[1];
    auto dstModelName = argv[2];
    FUNC_PRINT_ALL(modelName, s);
    FUNC_PRINT_ALL(dstModelName, s);
    onnx::ModelProto onnxModel;

    // read ONNX Model
    bool success = onnx_read_proto_from_binary(modelName, &onnxModel);
    if (!success) {
        MNN_PRINT("Load onnx model failed\n");
        return;
    }
    auto onnxGraph = onnxModel.mutable_graph();
    int size = onnxGraph->initializer_size();
    for (int i=0; i<size; ++i) {
        auto initial = onnxGraph->mutable_initializer(i);
        initial->clear_raw_data();
        initial->clear_float_data();
        initial->clear_double_data();
    }
    std::string output;
    google::protobuf::util::JsonOptions options;
    options.add_whitespace = true;
    options.always_print_primitive_fields = true;
    google::protobuf::util::MessageToJsonString(onnxModel, &output, options);
    std::ofstream outputOs(dstModelName);
    outputOs << output;
}


int main(int argc, const char* argv[]) {
    if (argc < 4) {
        if (argc < 3) {
            MNN_PRINT("Usage: ./OnnxClip SRC.onnx DST.onnx layerName\n");
            MNN_PRINT("Or: ./OnnxClip SRC.onnx struct.json\n");
            return 0;
        }
        _removeWeight(argc, argv);
        return 0;
    }
    auto modelName = argv[1];
    auto dstModelName = argv[2];
    auto layerName = argv[3];
    FUNC_PRINT_ALL(modelName, s);
    FUNC_PRINT_ALL(dstModelName, s);
    FUNC_PRINT_ALL(layerName, s);
    
    onnx::ModelProto onnxModel;
    // read ONNX Model
    bool success = onnx_read_proto_from_binary(modelName, &onnxModel);
    if (!success) {
        MNN_PRINT("Load onnx model failed\n");
        return 0;
    }
    auto onnxGraph = onnxModel.mutable_graph();
    auto output = onnxGraph->mutable_output(0);
    output->set_name(layerName);
    output->clear_type();
    const int nodeCount   = onnxGraph->node_size();
    int index = -1;
    for (int i=0; i<nodeCount; ++i) {
        auto node = onnxGraph->node(i);
        if (node.name() == layerName) {
            index = i;
            break;
        }
    }
    if (index < 0) {
        MNN_ERROR("Can't find %s in model, use outputName\n", layerName);
        for (int i=0; i<nodeCount; ++i) {
            auto node = onnxGraph->node(i);
            for (int v=0; v<node.output_size(); ++v) {
                if (node.output(v) == layerName) {
                    index = i;
                    break;
                }
            }
            if (index >= 0) {
                break;
            }
        }
        if (index < 0) {
            MNN_ERROR("Can't find outputName %s in model, Failed\n", layerName);
            return 0;
        }
    }
    int clipSize = nodeCount - index - 1;
    for (int i =0; i<clipSize; ++i) {
        onnxGraph->mutable_node()->RemoveLast();
    }
    onnxGraph->mutable_output()->Clear();
    auto info = onnxGraph->mutable_output()->Add();
    info->set_name(layerName);
    onnx_write_proto_from_binary(dstModelName, &onnxModel);
    return 0;
}
