//
//  BatchNormalScale.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"
using namespace MNN;

class BatchNormal : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        auto bn           = new BatchNormT;
        dstOp->main.value = bn;
        auto& l           = parameters;
        auto w            = &weight;
        // blob0:mean blob1:slope blob2:scale_factor
        const caffe::LayerParameter* w0 = (const caffe::LayerParameter*)w;
        DCHECK(w0->blobs_size() >= 2) << "Batchnorm blob ERROR! ==> " << parameters.name();
        const caffe::BlobProto& mean_blob                 = w0->blobs(0);
        const caffe::BlobProto& var_blob                  = w0->blobs(1);
        const caffe::BatchNormParameter& batch_norm_param = l.batch_norm_param();
        float eps                                         = batch_norm_param.eps();

        bn->channels = mean_blob.data_size();
        std::vector<float> ones(mean_blob.data_size(), 1.f);
        bn->slopeData = ones;
        bn->varData.resize(var_blob.data_size());
        bn->meanData.resize(mean_blob.data_size());
        bn->epsilon = eps;

        int blob_cnt = w0->blobs_size();
        if (blob_cnt < 3) {
            memcpy(bn->meanData.data(), mean_blob.data().data(), sizeof(float) * mean_blob.data_size());
            float tmp;
            for (int j = 0; j < var_blob.data_size(); j++) {
                tmp            = var_blob.data().data()[j];
                bn->varData[j] = tmp;
            }
        } else {
            auto scale_factor_div = w0->blobs(2).data().data()[0];
            float scale_factor = 0.0f;
            if (scale_factor_div != 0.0f) {
                scale_factor = 1.0f / scale_factor_div;
            }
            // pre-multiply scale_factor to mean and variance
            float tmp;
            for (int j = 0; j < mean_blob.data_size(); j++) {
                tmp             = mean_blob.data().data()[j] * scale_factor;
                bn->meanData[j] = tmp;
            }
            for (int j = 0; j < var_blob.data_size(); j++) {
                tmp            = var_blob.data().data()[j] * scale_factor;
                bn->varData[j] = tmp;
            }
        }
        bn->biasData = std::vector<float>(mean_blob.data_size(), 0.0f);
    }
    BatchNormal() {
    }
    virtual ~BatchNormal() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_BatchNorm;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_BatchNorm;
    }
};

static OpConverterRegister<BatchNormal> a("BatchNorm");

class CuDNNBatchNorm : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        auto bn           = new BatchNormT;
        dstOp->main.value = bn;
        auto& l           = parameters;
        auto w0           = &weight;
        DCHECK(w0->blobs_size() >= 2) << "caffemodel error!";
        const caffe::BlobProto& mean_blob                 = w0->blobs(0);
        const caffe::BlobProto& var_blob                  = w0->blobs(1);
        const caffe::BatchNormParameter& batch_norm_param = l.batch_norm_param();
        float eps                                         = batch_norm_param.eps();
        int blob_cnt                                      = w0->blobs_size();

        bn->channels = mean_blob.data_size();
        // mean
        bn->meanData.resize(mean_blob.data_size());
        memcpy(bn->meanData.data(), mean_blob.data().data(), mean_blob.data_size() * sizeof(float));
        // var
        bn->varData.resize(var_blob.data_size());
        memcpy(bn->varData.data(), var_blob.data().data(), var_blob.data_size() * sizeof(float));
        bn->epsilon = eps;
        // slope
        if (blob_cnt < 3) {
            bn->slopeData.resize(bn->varData.size());
            for (int i = 0; i < bn->varData.size(); i++) {
                bn->slopeData[i] = 1.0f;
            }
        } else {
            const caffe::BlobProto& scale_blob = w0->blobs(2);
            bn->slopeData.resize(scale_blob.data_size());
            memcpy(bn->slopeData.data(), scale_blob.data().data(), scale_blob.data_size() * sizeof(float));
        }
        // bias
        if (blob_cnt < 4) {
            bn->biasData.resize(mean_blob.data_size());
            for (int i = 0; i < bn->biasData.size(); i++) {
                bn->biasData[i] = 0.0f;
            }
        } else {
            const caffe::BlobProto& bias_blob = w0->blobs(3);
            bn->biasData.resize(bias_blob.data_size());
            memcpy(bn->biasData.data(), bias_blob.data().data(), bias_blob.data_size() * sizeof(float));
        }
    }
    CuDNNBatchNorm() {
    }
    virtual ~CuDNNBatchNorm() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_BatchNorm;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_BatchNorm;
    }
};

static OpConverterRegister<CuDNNBatchNorm> b("CuDNNBatchNorm");

class ScaleNode : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        auto sc           = new ScaleT;
        dstOp->main.value = sc;

        auto w                          = &weight;
        auto& l                         = parameters;
        const caffe::LayerParameter* w0 = (const caffe::LayerParameter*)w;
        DCHECK(w0->blobs_size() >= 1) << "caffemodel error!";
        const caffe::BlobProto& weight_blob      = w0->blobs(0);
        const caffe::ScaleParameter& scale_param = l.scale_param();
        sc->scaleData.resize(weight_blob.data_size());
        auto bias_term = scale_param.bias_term();
        sc->biasData   = std::vector<float>(weight_blob.data_size(), 0.0f);
        sc->channels   = weight_blob.data_size();

        const caffe::BlobProto& blob = w0->blobs(0);
        memcpy(sc->scaleData.data(), blob.data().data(), sizeof(float) * weight_blob.data_size());
        if (!bias_term) {
            return;
        }
        const caffe::BlobProto bias = w0->blobs(1);
        memcpy(sc->biasData.data(), bias.data().data(), sizeof(float) * bias.data_size());
    }

    ScaleNode() {
    }
    virtual ~ScaleNode() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Scale;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Scale;
    }
};
static OpConverterRegister<ScaleNode> _a("Scale");
