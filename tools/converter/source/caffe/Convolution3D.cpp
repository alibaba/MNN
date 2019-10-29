//
//  Convolution3D.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cstdint>
#include <vector>
#include "OpConverter.hpp"
#include "logkit.h"
using namespace std;

class Convolution3DConverter : public OpConverter {
public:
    Convolution3DConverter() {
    }

    virtual ~Convolution3DConverter() {
    }

    virtual MNN::OpType opType() {
        return MNN::OpType_Convolution3D;
    }

    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Convolution3D;
    }

    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        auto convolution3D = new MNN::Convolution3DT;
        DCHECK(weight.blobs_size() >= 1) << "Convolution3D weight blob ERROR! ==> " << parameters.name();
        dstOp->main.value = convolution3D;

        convolution3D->common = std::unique_ptr<MNN::Convolution3DCommonT>(new MNN::Convolution3DCommonT);
        auto& common          = convolution3D->common;

        common->padMode = MNN::PadMode_CAFFE;
        common->relu = common->relu6 = false;

        auto& convProto = parameters.convolution3d_param();
        { // group must be equal to 1
            const int group = convProto.has_group() ? convProto.group() : 1;
            DCHECK(group == 1) << "Convolution3D not support group convolution";
        }

        { // kernel_size, kernel_depth
            const int kernel_depth = convProto.kernel_depth();
            const int kernel_size  = convProto.kernel_size();
            common->kernels        = std::vector<int32_t>({kernel_depth, kernel_size, kernel_size});
        }
        { // stride, temporal_stride
            const int stride          = convProto.stride();
            const int temporal_stride = convProto.temporal_stride();
            common->strides           = std::vector<int32_t>({temporal_stride, stride, stride});
        }
        { // pad, temporal_pad
            const int pad          = convProto.pad();
            const int temporal_pad = convProto.temporal_pad();
            common->pads           = std::vector<int32_t>({temporal_pad, pad, pad});
        }
        common->dilates = std::vector<int32_t>({1, 1, 1});

        { // set kernel weight data
            auto& weightBlob = weight.blobs(0);
            DCHECK(weightBlob.shape().dim_size() == 5) << "Conv3D Weight Dimension ERROR!";
            common->outputCount = convProto.num_output();
            DCHECK(weightBlob.has_shape()) << "Caffemodel ERROR!";
            common->inputCount = weightBlob.shape().dim(1);

            int size = 1;
            for (int i = 0; i < weightBlob.shape().dim_size(); ++i) {
                size *= weightBlob.shape().dim(i);
            }
            std::vector<float> weightData;
            weightData.resize(size);
            for (int i = 0; i < size; ++i) {
                weightData[i] = weightBlob.data(i);
            }
            convolution3D->weight = weightData;
        }

        { // set bias data
            std::vector<float> biasData(convProto.num_output(), 0.0f);
            if (convProto.bias_term() && weight.blobs_size() >= 2) {
                for (int i = 0; i < biasData.size(); ++i) {
                    biasData[i] = weight.blobs(1).data(i);
                }
            }
            convolution3D->bias = biasData;
        }
    }
};

// https://github.com/facebook/C3D/blob/master/C3D-v1.1/src/caffe/proto/caffe.proto
static OpConverterRegister<Convolution3DConverter> a("Convolution3D");
