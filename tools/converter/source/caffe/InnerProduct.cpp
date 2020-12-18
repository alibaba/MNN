//
//  InnerProduct.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class InnerProductCommon : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        auto innerproduct                       = new MNN::InnerProductT;
        dstOp->main.value                       = innerproduct;
        auto& l                                 = parameters;
        const caffe::InnerProductParameter& par = l.inner_product_param();
        innerproduct->outputCount               = par.num_output();
        innerproduct->axis                      = 1;
        if (par.has_axis()) {
            innerproduct->axis = par.axis();
        }
        innerproduct->transpose = false;
        if (par.has_transpose()) {
            innerproduct->transpose = par.transpose();
        }
    }
    InnerProductCommon() {
    }
    virtual ~InnerProductCommon() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_InnerProduct;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_InnerProduct;
    }
};

class InnerProduct : public InnerProductCommon {
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        InnerProductCommon::run(dstOp, parameters, weight);
        auto innerproduct                       = dstOp->main.AsInnerProduct();
        const caffe::InnerProductParameter& par = parameters.inner_product_param();
        const caffe::LayerParameter* v0w        = &weight;
        DCHECK(v0w->blobs_size() >= 1) << "caffemodel error!";
        innerproduct->biasTerm = par.bias_term();
        innerproduct->bias.resize(par.num_output());
        ::memset(innerproduct->bias.data(), 0, innerproduct->bias.size() * sizeof(float));
        if (par.bias_term()) {
            ::memcpy(innerproduct->bias.data(), v0w->blobs(1).data().data(), par.num_output() * sizeof(float));
        }
        const caffe::BlobProto& WeightBlob = v0w->blobs(0);
        innerproduct->weightSize           = WeightBlob.data_size();
        innerproduct->weight.resize(innerproduct->weightSize);
        if (innerproduct->transpose) {
            const float* src = WeightBlob.data().data();
            float *dst       = innerproduct->weight.data();
            int outputCount  = innerproduct->outputCount;
            int srcCount     = innerproduct->weightSize / outputCount;
            for (int i = 0; i < outputCount; i++) {
                for (int j = 0; j < srcCount; j++) {
                    dst[i * srcCount + j] = src[i + j * outputCount];
                }
            }
            innerproduct->transpose = false;
        } else {
            ::memcpy(innerproduct->weight.data(), WeightBlob.data().data(), sizeof(float) * innerproduct->weightSize);
        }
    }
};

static OpConverterRegister<InnerProduct> a("InnerProduct");
