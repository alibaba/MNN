//
//  OpConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "Tensor_generated.h"
#include <MNN/MNNDefine.h>
#include <stdlib.h>
#include <vector>
#include <string>

OpConverterSuit* OpConverterSuit::global = nullptr;
class DefaultCaffeOpConverter : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) override {
        dstOp->main.value = new MNN::ExtraT;
        dstOp->main.AsExtra()->engine = "Caffe";
        dstOp->main.AsExtra()->type = parameters.type();

        if (parameters.type() == "Power") {
            std::unique_ptr<MNN::AttributeT> attr1(new MNN::AttributeT);
            attr1->key = "scale";
            attr1->f = parameters.power_param().scale();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr1));

            std::unique_ptr<MNN::AttributeT> attr2(new MNN::AttributeT);
            attr2->key = "shift";
            attr2->f = parameters.power_param().shift();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr2));

            std::unique_ptr<MNN::AttributeT> attr3(new MNN::AttributeT);
            attr3->key = "power";
            attr3->f = parameters.power_param().power();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr3));
        }

        if (parameters.type() == "Exp") {
            std::unique_ptr<MNN::AttributeT> attr1(new MNN::AttributeT);
            attr1->key = "base";
            attr1->f = parameters.exp_param().base();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr1));

            std::unique_ptr<MNN::AttributeT> attr2(new MNN::AttributeT);
            attr2->key = "scale";
            attr2->f = parameters.exp_param().scale();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr2));

            std::unique_ptr<MNN::AttributeT> attr3(new MNN::AttributeT);
            attr3->key = "shift";
            attr3->f = parameters.exp_param().shift();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr3));
        }

        if (parameters.type() == "Log") {
            std::unique_ptr<MNN::AttributeT> attr1(new MNN::AttributeT);
            attr1->key = "base";
            attr1->f = parameters.log_param().base();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr1));

            std::unique_ptr<MNN::AttributeT> attr2(new MNN::AttributeT);
            attr2->key = "scale";
            attr2->f = parameters.log_param().scale();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr2));

            std::unique_ptr<MNN::AttributeT> attr3(new MNN::AttributeT);
            attr3->key = "shift";
            attr3->f = parameters.log_param().shift();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr3));
        }

        if (parameters.type() == "MVN") {
            std::unique_ptr<MNN::AttributeT> attr1(new MNN::AttributeT);
            attr1->key = "across_channels";
            attr1->b = parameters.mvn_param().across_channels();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr1));

            std::unique_ptr<MNN::AttributeT> attr2(new MNN::AttributeT);
            attr2->key = "eps";
            attr2->f = parameters.mvn_param().eps();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr2));

            std::unique_ptr<MNN::AttributeT> attr3(new MNN::AttributeT);
            attr3->key = "normalize_variance";
            attr3->b = parameters.mvn_param().normalize_variance();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr3));
        }

        if (parameters.type() == "Bias") {
            std::unique_ptr<MNN::AttributeT> attr1(new MNN::AttributeT);
            attr1->key = "axis";
            attr1->i = parameters.bias_param().axis();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr1));

            std::unique_ptr<MNN::AttributeT> attr2(new MNN::AttributeT);
            attr2->key = "num_axes";
            attr2->i = parameters.bias_param().num_axes();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr2));

            if (weight.blobs_size() != 0) {
                MNN_ASSERT(weight.blobs_size() == 1);
                std::unique_ptr<MNN::AttributeT> attr3(new MNN::AttributeT);
                attr3->key = "bias";
                auto shapeSize = weight.blobs(0).shape().dim_size();
                std::vector<int> biasShape;
                int biasSize = 1;
                for (int i = 0; i < shapeSize; i++) {
                    biasShape.emplace_back(weight.blobs(0).shape().dim(i));
                    biasSize *= biasShape[i];
                }
                attr3->tensor.reset(new MNN::BlobT);
                attr3->tensor->dims = biasShape;
                attr3->tensor->dataFormat = MNN::MNN_DATA_FORMAT::MNN_DATA_FORMAT_NCHW;
                attr3->tensor->float32s.clear();
                for (int i = 0; i < biasSize; i++) {
                    attr3->tensor->float32s.emplace_back(weight.blobs(0).data(i));
                }
                attr3->i = biasSize;
                dstOp->main.AsExtra()->attr.emplace_back(std::move(attr3));
            }
        }

        if (parameters.type() == "Embed") {
            std::unique_ptr<MNN::AttributeT> attr1(new MNN::AttributeT);
            attr1->key = "num_output";
            attr1->i = parameters.embed_param().num_output();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr1));

            std::unique_ptr<MNN::AttributeT> attr2(new MNN::AttributeT);
            attr2->key = "input_dim";
            attr2->i = parameters.embed_param().input_dim();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr2));

            std::unique_ptr<MNN::AttributeT> attr3(new MNN::AttributeT);
            attr3->key = "bias_term";
            attr3->b = parameters.embed_param().bias_term();
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr3));

            std::unique_ptr<MNN::AttributeT> attr4(new MNN::AttributeT);
            attr4->key = "weights";
            auto shapeSize = weight.blobs(0).shape().dim_size();
            std::vector<int> weightsShape;
            int weightsSize = 1;
            for (int i = 0; i < shapeSize; i++) {
                weightsShape.emplace_back(weight.blobs(0).shape().dim(i));
                weightsSize *= weightsShape[i];
            }
            attr4->tensor.reset(new MNN::BlobT);
            attr4->tensor->dims = weightsShape;
            attr4->tensor->dataFormat = MNN::MNN_DATA_FORMAT::MNN_DATA_FORMAT_NCHW;
            attr4->tensor->float32s.clear();
            for (int i = 0; i < weightsSize; i++) {
                attr4->tensor->float32s.emplace_back(weight.blobs(0).data(i));
            }
            attr4->i = weightsSize;
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr4));

            if (parameters.embed_param().bias_term()) {
                std::unique_ptr<MNN::AttributeT> attr5(new MNN::AttributeT);
                attr5->key = "bias";
                auto shapeSize = weight.blobs(1).shape().dim_size();
                std::vector<int> biasShape;
                int biasSize = 1;
                for (int i = 0; i < shapeSize; i++) {
                    biasShape.emplace_back(weight.blobs(1).shape().dim(i));
                    biasSize *= biasShape[i];
                }
                attr5->tensor.reset(new MNN::BlobT);
                attr5->tensor->dims = biasShape;
                attr5->tensor->dataFormat = MNN::MNN_DATA_FORMAT::MNN_DATA_FORMAT_NCHW;
                attr5->tensor->float32s.clear();
                for (int i = 0; i < biasSize; i++) {
                    attr5->tensor->float32s.emplace_back(weight.blobs(1).data(i));
                }
                attr5->i = biasSize;
                dstOp->main.AsExtra()->attr.emplace_back(std::move(attr5));
            }
        }

        if (parameters.type() == "Reduction") {
            std::string opType;
            if (parameters.reduction_param().operation() == caffe::ReductionParameter_ReductionOp_SUM) {
                opType = "SUM";
            }
            if (parameters.reduction_param().operation() == caffe::ReductionParameter_ReductionOp_MEAN) {
                opType = "MEAN";
            }
            if (parameters.reduction_param().operation() == caffe::ReductionParameter_ReductionOp_ASUM) {
                opType = "ASUM";
            }
            if (parameters.reduction_param().operation() == caffe::ReductionParameter_ReductionOp_SUMSQ) {
                opType = "SUMSQ";
            }

            std::unique_ptr<MNN::AttributeT> attr1(new MNN::AttributeT);
            attr1->key = opType;
            auto reductionDim = parameters.reduction_param().axis();
            if (reductionDim < 0) {
                reductionDim += 4; // only support at most 4 dimensions
            }
            attr1->i = reductionDim;
            dstOp->main.AsExtra()->attr.emplace_back(std::move(attr1));
        }
    }
    virtual MNN::OpParameter type() override {
        return MNN::OpParameter_Extra;
    }
    virtual MNN::OpType opType() override {
        return MNN::OpType_Extra;
    }
    
private:
};

OpConverter* OpConverterSuit::search(const std::string& name) {
    auto iter = mTests.find(name);
    if (iter == mTests.end()) {
        static DefaultCaffeOpConverter converter;
        return &converter;
    }
    return iter->second;
}

OpConverterSuit* OpConverterSuit::get() {
    if (global == nullptr)
        global = new OpConverterSuit;
    return global;
}

OpConverterSuit::~OpConverterSuit() {
    for (auto& iter : mTests) {
        delete iter.second;
    }
    mTests.clear();
}

void OpConverterSuit::insert(OpConverter* t, const char* name) {
    mTests.insert(std::make_pair(name, t));
}
