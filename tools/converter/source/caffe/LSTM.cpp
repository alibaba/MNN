//
//  LSTM.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class LSTM : public OpConverter {
public:
    void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    virtual MNN::OpType opType() {
        return MNN::OpType_LSTM;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_LSTM;
    }
};

void LSTM::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    MNN::LSTMT* lstmt = new MNN::LSTMT;
    dstOp->main.value = lstmt;

    auto lstmcaffe           = parameters.lstm_param();
    lstmt->outputCount       = lstmcaffe.num_output();
    lstmt->clippingThreshold = lstmcaffe.clipping_threshold();

    int SizeI = 0, SizeH = 0;
    // blob[0] weight_i blob[1] weight_h blob[2] bias
    auto w      = &weight;
    int blobCnt = ((caffe::LayerParameter*)w)->blobs().size();
    if (blobCnt >= 1) {
        const caffe::BlobProto& wi = ((caffe::LayerParameter*)w)->blobs(0);
        SizeI                      = wi.data_size();
        if (SizeI > 0) {
            lstmt->weightI = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
            lstmt->weightI->dims.push_back(SizeI);
            lstmt->weightI->float32s.resize(SizeI);

            memcpy(lstmt->weightI->float32s.data(), wi.data().data(), sizeof(float) * SizeI);
        }
    }
    if (blobCnt >= 2) {
        const caffe::BlobProto& wh = ((caffe::LayerParameter*)w)->blobs(1);
        SizeH                      = wh.data_size();
        if (SizeH > 0) {
            lstmt->weightH = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
            lstmt->weightH->dims.push_back(SizeH);
            lstmt->weightH->float32s.resize(SizeH);

            memcpy(lstmt->weightH->float32s.data(), wh.data().data(), sizeof(float) * SizeH);
        }
    }
    if (blobCnt >= 3) {
        const caffe::BlobProto& b = ((caffe::LayerParameter*)w)->blobs(2);
        int biasCnt               = b.data_size();
        if (biasCnt > 0) {
            lstmt->bias = std::unique_ptr<MNN::BlobT>(new MNN::BlobT);
            lstmt->bias->dims.push_back(biasCnt);
            lstmt->bias->float32s.resize(biasCnt);

            memcpy(lstmt->bias->float32s.data(), b.data().data(), sizeof(float) * biasCnt);
        }
    }
    lstmt->weightSize = SizeI > SizeH ? SizeH : SizeI;
}

static OpConverterRegister<LSTM> a("Lstm");
static OpConverterRegister<LSTM> _a("OCRLSTM");
static OpConverterRegister<LSTM> _sa("OCRLSTMQ");
static OpConverterRegister<LSTM> __b("CuDNNLstmForward");
