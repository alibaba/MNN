//
//  Convolution.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/OpCommonUtils.hpp"
#include "OpConverter.hpp"
#include "logkit.h"
using namespace std;

class ConvolutionCommon : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        auto convolution2D = new MNN::Convolution2DT;
        DCHECK(weight.blobs_size() >= 1) << "Convolution weight blob ERROR! ==> " << parameters.name();
        dstOp->main.value = convolution2D;

        convolution2D->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
        auto& common          = convolution2D->common;

        auto& convProto = parameters.convolution_param();
        common->group   = convProto.has_group() ? convProto.group() : 1;

        auto& p             = convProto;
        common->outputCount = p.num_output();

        auto& weightBlob      = weight.blobs(0);
        if (weightBlob.has_shape()) {
            // get weight information from weight Blob shape(caffe proto v2)
            DCHECK(weightBlob.shape().dim_size() == 4) << "Conv Weight Dimension ERROR!";
            common->inputCount = weightBlob.shape().dim(0) * weightBlob.shape().dim(1) / p.num_output() * common->group;
        } else {
            // get shape information from Blob parameters(caffe proto v1)
            common->inputCount = weightBlob.num() * weightBlob.channels() / p.num_output() * common->group;
        }
        // kernelsize
        int kernelSize[3];
        int dilation[3];
        const int MAX_DIM = 3;
        kernelSize[2] = kernelSize[1] = kernelSize[0] = 1;
        if (p.kernel_size_size() == 1) {
            kernelSize[0] = p.kernel_size(0);
            kernelSize[1] = p.kernel_size(0);
            kernelSize[2] = p.kernel_size(0);
        } else if (p.kernel_size_size() > MAX_DIM) {
            for (int i = 0; i < MAX_DIM; i++) {
                kernelSize[i] = p.kernel_size(p.kernel_size_size() - MAX_DIM);
            }
        } else {
            for (int i = 0; i < p.kernel_size_size(); i++) {
                kernelSize[i] = p.kernel_size(i);
            }
        }
        if (p.has_kernel_h())
            kernelSize[1] = p.kernel_h();
        if (p.has_kernel_w())
            kernelSize[0] = p.kernel_w();

        common->kernelX = (kernelSize[0]);
        common->kernelY = (kernelSize[1]);

        // dilation
        dilation[2] = dilation[1] = dilation[0] = 1;
        if (p.dilation_size() == 1) {
            dilation[0] = p.dilation(0);
            dilation[1] = p.dilation(0);
            dilation[2] = p.dilation(0);
        } else if (p.dilation_size() > MAX_DIM) {
            for (int i = 0; i < MAX_DIM; i++) {
                dilation[i] = p.dilation(p.dilation_size() - MAX_DIM);
            }
        } else {
            for (int i = 0; i < p.dilation_size(); i++) {
                dilation[i] = p.dilation(i);
            }
        }
        common->dilateX = (dilation[0]);
        common->dilateY = (dilation[1]);

        // stride
        int stride[3];
        int pad[3];
        stride[2] = stride[1] = stride[0] = 1;
        if (p.stride_size() == 1) {
            stride[0] = p.stride(0);
            stride[1] = p.stride(0);
            stride[2] = p.stride(0);
        } else if (p.stride_size() > MAX_DIM) {
            for (int i = 0; i < MAX_DIM; i++) {
                stride[i] = p.stride(p.stride_size() - MAX_DIM);
            }
        } else {
            for (int i = 0; i < p.stride_size(); i++) {
                stride[i] = p.stride(i);
            }
        }
        if (p.has_stride_h())
            stride[1] = p.stride_h();
        if (p.has_stride_w())
            stride[0] = p.stride_w();
        common->strideX = stride[0];
        common->strideY = stride[1];
        // pad
        pad[0] = pad[1] = pad[2] = 0;
        if (p.pad_size() == 1) {
            pad[0] = p.pad(0);
            pad[1] = p.pad(0);
            pad[2] = p.pad(0);
        } else if (p.pad_size() > MAX_DIM) {
            for (int i = 0; i < MAX_DIM; i++)
                pad[i] = p.pad(p.pad_size() - MAX_DIM);
        } else {
            for (int i = 0; i < p.pad_size(); i++)
                pad[i] = p.pad(i);
        }
        if (p.has_pad_h())
            pad[1] = p.pad_h();
        if (p.has_pad_w())
            pad[0] = p.pad_w();

        common->padX    = pad[0];
        common->padY    = pad[1];
        common->padMode = MNN::PadMode_CAFFE;
    }
    ConvolutionCommon() {
    }
    virtual ~ConvolutionCommon() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Convolution;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Convolution2D;
    }
};

class Convolution : public ConvolutionCommon {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        ConvolutionCommon::run(dstOp, parameters, weight);
        auto weightBlob = weight.blobs(0);

        auto convolution2D = dstOp->main.AsConvolution2D();
        int size           = 1;
        if (weightBlob.has_shape()) {
            for (int i = 0; i < weightBlob.shape().dim_size(); ++i) {
                size *= weightBlob.shape().dim(i);
            }
        } else {
            size = weightBlob.num() * weightBlob.channels() * weightBlob.height() * weightBlob.width();
        }
        std::vector<float> weightData;
        weightData.resize(size);
        for (int i = 0; i < size; ++i) {
            weightData[i] = weightBlob.data(i);
        }
        convolution2D->weight = weightData;

        auto& convProto = parameters.convolution_param();
        std::vector<float> biasData(convProto.num_output(), 0.0f);
        if (convProto.bias_term() && weight.blobs_size() >= 2) {
            for (int i = 0; i < biasData.size(); ++i) {
                biasData[i] = weight.blobs(1).data(i);
            }
        }
        convolution2D->bias = biasData;
    }
};

static OpConverterRegister<Convolution> a("Convolution");
static OpConverterRegister<Convolution> ___aC("CuDNNGroupedConvolutionForward");

class Deconvolution : public Convolution {
public:
    virtual MNN::OpType opType() {
        return MNN::OpType_Deconvolution;
    }
};
static OpConverterRegister<Deconvolution> _a("Deconvolution");


class ConvolutionDepthwise : public ConvolutionCommon {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        ConvolutionCommon::run(dstOp, parameters, weight);
        auto weightBlob = weight.blobs(0);

        auto convolution2D = dstOp->main.AsConvolution2D();
        convolution2D->common->group = convolution2D->common->outputCount;
        convolution2D->common->inputCount = convolution2D->common->outputCount;
        int size           = 1;
        if (weightBlob.has_shape()) {
            for (int i = 0; i < weightBlob.shape().dim_size(); ++i) {
                size *= weightBlob.shape().dim(i);
            }
        } else {
            size = weightBlob.num() * weightBlob.channels() * weightBlob.height() * weightBlob.width();
        }

        std::vector<float> weightData;
        weightData.resize(size);
        for (int i = 0; i < size; ++i) {
            weightData[i] = weightBlob.data(i);
        }
        convolution2D->weight = weightData;

        auto& convProto = parameters.convolution_param();
        std::vector<float> biasData(convProto.num_output(), 0.0f);
        if (convProto.bias_term() && weight.blobs_size() >= 2) {
            for (int i = 0; i < biasData.size(); ++i) {
                biasData[i] = weight.blobs(1).data(i);
            }
        }
        convolution2D->bias = biasData;
    }
};

static OpConverterRegister<ConvolutionDepthwise> ab("ConvolutionDepthwise");
static OpConverterRegister<ConvolutionDepthwise> ab2("DepthwiseConv");
