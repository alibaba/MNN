//
//  caffeConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include "MNN_generated.h"
#include "OpConverter.hpp"
#include "caffe.pb.h"
#include "logkit.h"

#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"

#include "CaffeUtils.hpp"
#include "caffeConverter.hpp"

int caffe2MNNNet(const std::string prototxtFile, const std::string modelFile, const std::string bizCode,
                 std::unique_ptr<MNN::NetT>& netT) {
    caffe::NetParameter caffeProtxt;
    caffe::NetParameter caffeModel;
    bool succ = read_proto_from_text(prototxtFile.c_str(), &caffeProtxt);
    DCHECK(succ) << "read_proto_from_text failed";

    succ &= read_proto_from_binary(modelFile.c_str(), &caffeModel);
    DCHECK(succ) << "read_proto_from_binary failed";
    if (!succ) {
        MNN_ERROR("[ERROR] Model file is not caffe model.\n");
        return 1;
    }
    std::map<std::string, int> tensorName;

    // Load Parameters
    // MNN::NetT netT;
    // Add Extra Input
    // TODO Support shape
    if (caffeProtxt.input_size() > 0) {
        for (int v = 0; v < caffeProtxt.input_size(); ++v) {
            if (caffeProtxt.input_dim_size() <= 0) {
                continue;
            }
            MNN::OpT* op  = new MNN::OpT;
            op->name      = caffeProtxt.input(v);
            op->type      = MNN::OpType_Input;
            op->main.type = MNN::OpParameter_Input;
            auto inputT   = new MNN::InputT;
            for (int i = 0; i < caffeProtxt.input_dim_size(); ++i) {
                inputT->dims.push_back(caffeProtxt.input_dim(i));
            }
            op->main.value = inputT;
            op->outputIndexes.push_back(v);

            netT->oplists.emplace_back(op);
            netT->tensorName.push_back(op->name);
            tensorName.insert(std::make_pair(op->name, v));
        }
    }
    if (caffeProtxt.input_shape_size() > 0) {
        for (int v = 0; v < caffeProtxt.input_shape_size(); ++v) {
            MNN::OpT* op  = new MNN::OpT;
            op->name      = caffeProtxt.input(v);
            op->type      = MNN::OpType_Input;
            op->main.type = MNN::OpParameter_Input;
            auto inputT   = new MNN::InputT;
            auto shape    = caffeProtxt.input_shape(v);
            for (int i = 0; i < shape.dim_size(); ++i) {
                inputT->dims.push_back(shape.dim(i));
            }
            op->main.value = inputT;
            op->outputIndexes.push_back(v);

            netT->oplists.emplace_back(op);
            netT->tensorName.push_back(op->name);
            tensorName.insert(std::make_pair(op->name, v));
        }
    }

    // Compute TensorCount
    {
        for (int l = 0; l < caffeProtxt.layer_size(); ++l) {
            auto& layer = caffeProtxt.layer(l);
            for (int t = 0; t < layer.top_size(); ++t) {
                auto name = layer.top(t);
                if (tensorName.find(name) == tensorName.end()) {
                    tensorName.insert(std::make_pair(layer.top(t), tensorName.size()));
                    netT->tensorName.push_back(name);
                }
            }
        }
    }

    for (int l = 0; l < caffeProtxt.layer_size(); ++l) {
        MNN::OpT* op = new MNN::OpT;
        auto& layer  = caffeProtxt.layer(l);
        op->name     = layer.name();
        // Input Output
        for (int t = 0; t < layer.top_size(); ++t) {
            op->outputIndexes.emplace_back(tensorName.find(layer.top(t))->second);
        }

        for (int t = 0; t < layer.bottom_size(); ++t) {
            op->inputIndexes.emplace_back(tensorName.find(layer.bottom(t))->second);
        }

        auto creator = OpConverterSuit::get()->search(layer.type());
        if (nullptr == creator) {
            LG << "Don't support type [ " << layer.type().c_str() << " ], for " << layer.name().c_str();
            delete op;
            break;
        }
        const caffe::LayerParameter* layerP = nullptr;
        for (int v = 0; v < caffeModel.layer_size(); ++v) {
            auto& l = caffeModel.layer(v);
            if (l.name() == layer.name()) {
                layerP = &l;
                break;
            }
        }
        if (NULL == layerP) {
            layerP = &layer;
        }
        op->type      = creator->opType();
        op->main.type = creator->type();

        creator->run(op, layer, *layerP);

        netT->oplists.emplace_back(op);
    }
    netT->sourceType = MNN::NetSource_CAFFE;
    netT->bizCode    = bizCode;

    return 0;
}
