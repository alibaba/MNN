//
//  tensorflowConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TfUtils.hpp"
#include "logkit.h"

#include "TFGraphResolver.hpp"
#include "tensorflowConverter.hpp"

int tensorflow2MNNNet(const std::string inputModel, const std::string bizCode,
                      std::unique_ptr<MNN::NetT> &netT) {
    // Load tensorflow model.
    tensorflow::GraphDef tfGraph;
    bool success = tf_read_proto_from_binary(inputModel.c_str(), &tfGraph);
    DCHECK(success) << "read_proto_from_binary failed!";
    if (!success) {
        MNN_ERROR("[ERROR] MNNConvert just support tensorflow frozen graph model. Model file is not tf frozen graph model.\n");
        return 1;
    }

    TFGraphResolver resolver(tfGraph);
    for (int i = 0; i < resolver.graph_size(); ++i) {
        const TFGraph *graph = resolver.graph(i);
        auto graph_proto = graph->ToProto();
        // The graph indexed by 0 is main graph.
        if (i == 0) {
            netT->oplists = std::move(graph_proto->nodes);
            netT->tensorName = graph_proto->tensors;
        } else {
            netT->subgraphs.push_back(std::move(graph_proto));
        }
    }
    netT->sourceType = MNN::NetSource_TENSORFLOW;
    netT->bizCode    = bizCode;
    return 0;
}
