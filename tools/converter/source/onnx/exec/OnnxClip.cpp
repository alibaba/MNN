//
//  OnnxClip.cpp
//  MNNConverter
//
//  Created by MNN on 2020/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include "../OnnxUtils.hpp"
#include "onnx.pb.h"

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./OnnxClip SRC.onnx DST.onnx layerName\n");
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
