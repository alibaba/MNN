//
//  PipelineBuilder.hpp
//  MNNConverter
//
//  Created by MNN on 2020/07/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_COMPRESSION_PIPELINE_HPP_
#define MNN_CONVERTER_COMPRESSION_PIPELINE_HPP_

#include "quantization.hpp"
#include "MNN_compression.pb.h"

typedef MNN::Compression::CompressionAlgo CompressionAlgo;
typedef MNN::Compression::LayerQuantizeParams LayerQuantizeParams;

namespace compression {

class PipelineBuilder;

typedef struct Progress {
    CompressionAlgo::CompressionType type;

    // TODO(): Support prune.
    Quantization quant_params;
} Progress;

class Pipeline {
public:
    Pipeline() = default;
    virtual ~Pipeline() = default;

    const std::vector<Progress>& progress() const {
        return progress_;
    }

private:
    friend class PipelineBuilder;
    std::vector<Progress> progress_;
};

class PipelineBuilder {
public:
    PipelineBuilder() = delete;

    explicit PipelineBuilder(const std::string& filename);

    virtual ~PipelineBuilder() = default;

    Pipeline Build() const;

private:
    bool ParsePipeline(const MNN::Compression::Pipeline& proto,
                       Pipeline* pipeline) const;

    bool ParseQuantization(const MNN::Compression::QuantizeParams& proto,
                           Quantization* quant_params) const;

    Quantization::TensorParams ParseWeightQuantization(
        const LayerQuantizeParams::WeightParams& proto) const;

    Quantization::TensorParams ParseActivationQuantization(
        const LayerQuantizeParams::ActivationParams& proto) const;

private:
    std::string filename_ = "";
};

};

#endif  // MNN_CONVERTER_COMPRESSION_PIPELINE_HPP_
