//
//  PipelineBuilder.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>

#include "PipelineBuilder.hpp"
#include "MNN/MNNDefine.h"
#include <sstream>
#include "rapidjson/document.h"
#include "cli.hpp"
#include "commonKit.hpp"

namespace compression {

PipelineBuilder::PipelineBuilder(const std::string& filename)
    : filename_(filename) {}

Pipeline PipelineBuilder::Build() const {
    Pipeline pipeline;
    if (!filename_.empty()) {
        MNN::Compression::Pipeline proto;
        std::string jsonSuffix = "json";
        std::string suffix = filename_.substr(filename_.find_last_of('.') + 1);
        if (jsonSuffix.compare(suffix) != 0) { // protobuf.bin
            std::fstream input(filename_.c_str(), std::ios::in | std::ios::binary);
            if (!proto.ParseFromIstream(&input)) {
                MNN_ERROR("Failed to parse compression pipeline proto.\n");
            } else {
                ParsePipeline(proto, &pipeline);
            }
        } else {
            CommonKit::json2protobuf(filename_.c_str(), nullptr, &proto);
            ParsePipeline(proto, &pipeline);
        }
    }
    return std::move(pipeline);
}

bool PipelineBuilder::ParsePipeline(const MNN::Compression::Pipeline& proto,
                                    Pipeline* pipeline) const {
    for (const auto& algo : proto.algo()) {
        Progress progress;
        progress.type = algo.type();
        switch (progress.type) {
            case CompressionAlgo::QUANTIZE: {
                ParseQuantization(algo.quant_params(),
                                  &(progress.quant_params));
                break;
            }
            case CompressionAlgo::PRUNE:
            default: {
                MNN_ERROR("Unsupported compression type: %d.\n", progress.type);
            }
        }
        pipeline->progress_.push_back(std::move(progress));
    }
    return true;
}

Quantization::TensorParams PipelineBuilder::ParseActivationQuantization(
        const LayerQuantizeParams::ActivationParams& proto) const {
    Quantization::TensorParams tensor_params;
    tensor_params.nbit = proto.bits();
    int size = proto.scales().size();
    tensor_params.scale.resize(size);
    for (int i = 0; i < size; ++i) {
        tensor_params.scale[i] = proto.scales(i);
    }
    tensor_params.zero_point = proto.zero_point();
    tensor_params.clamp_min = proto.clamp_min();
    tensor_params.clamp_max = proto.clamp_max();
    return std::move(tensor_params);
}

Quantization::TensorParams PipelineBuilder::ParseWeightQuantization(
        const LayerQuantizeParams::WeightParams& proto) const {
    Quantization::TensorParams tensor_params;
    tensor_params.nbit = proto.bits();
    int size = proto.scales().size();
    tensor_params.scale.resize(size);
    for (int i = 0; i < size; ++i) {
        tensor_params.scale[i] = proto.scales(i);
    }
    tensor_params.zero_point = 0.f;
    return std::move(tensor_params);
}

bool PipelineBuilder::ParseQuantization(
        const MNN::Compression::QuantizeParams& proto,
        Quantization* quant_params) const {
    quant_params->round_mode = proto.round_mode();
    for (const auto& layer_proto : proto.layer()) {
        auto method = layer_proto.method();
        for (const auto& input : layer_proto.input()) {
            const std::string& tensor_name = input.name();
            Quantization::TensorParams tensor_params =
                ParseActivationQuantization(input);
            tensor_params.method = method;
            quant_params->tensors[tensor_name].push_back(tensor_params);
        }
        int length = 0;
        for (const auto& weight : layer_proto.weight()) {
            const std::string& tensor_name = weight.name();
            Quantization::TensorParams tensor_params =
                ParseWeightQuantization(weight);
            // TODO(): FIXME
            // quant_params->tensors[tensor_name].push_back(tensor_params);
            length = tensor_params.scale.size();
        }
        for (const auto& output : layer_proto.output()) {
            const std::string& tensor_name = output.name();
            Quantization::TensorParams tensor_params =
                ParseActivationQuantization(output);
            if (tensor_params.scale.size() != length) {
                MNN_ERROR("Output(%s) scale and weight scale length are "
                           "mismatch, (%d vs %d).\n", tensor_name.c_str(),
                           int(tensor_params.scale.size()), length);
                MNN_ASSERT(false);
            }
            tensor_params.method = method;
            quant_params->tensors[tensor_name].push_back(tensor_params);
        }
    }
    return true;
}

};
