//
//  Pool.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <vector>
#include "OpConverter.hpp"
#include "logkit.h"

class Pool : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Pool() {
    }
    virtual ~Pool() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Pooling;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Pool;
    }
};

void Pool::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    const caffe::PoolingParameter& p = parameters.pooling_param();
    auto pool                        = new MNN::PoolT;
    dstOp->main.value                = pool;
    auto poolingType                 = p.pool();
    if (poolingType == caffe::PoolingParameter::MAX) {
        pool->type = MNN::PoolType_MAXPOOL;
    } else if (poolingType == caffe::PoolingParameter::AVE) {
        pool->type = MNN::PoolType_AVEPOOL;
    } else {
        DLOG(FATAL) << "Pool type not support! ==> " << parameters.name();
    }

    // orinal NCHW, our whc
    int kernelSize[3];
    kernelSize[2] = kernelSize[1] = kernelSize[0] = 1;
    if (p.has_kernel_size())
        kernelSize[2] = kernelSize[1] = kernelSize[0] = p.kernel_size();
    if (p.has_kernel_w())
        kernelSize[0] = p.kernel_w();
    if (p.has_kernel_h())
        kernelSize[1] = p.kernel_h();

    pool->kernelY = (kernelSize[1]);
    pool->kernelX = (kernelSize[0]);

    int stride[3];
    int pad[3];
    int isGlobal = 0;
    stride[2] = stride[1] = stride[0] = 1;

    if (p.has_stride())
        stride[2] = stride[1] = stride[0] = p.stride();
    if (p.has_stride_w())
        stride[0] = p.stride_w();
    if (p.has_stride_h())
        stride[1] = p.stride_h();
    pool->strideY = (stride[1]);
    pool->strideX = (stride[0]);

    pad[2] = pad[1] = pad[0] = 0;
    if (p.has_pad())
        pad[2] = pad[1] = pad[0] = p.pad();
    if (p.has_pad_w())
        pad[0] = p.pad_w();
    if (p.has_pad_h())
        pad[1] = p.pad_h();
    pool->padY = pad[1];
    pool->padX = pad[0];

    isGlobal       = p.has_global_pooling() ? p.global_pooling() : 0;
    pool->isGlobal = isGlobal;
    pool->padType  = MNN::PoolPadType_CAFFE;
}
static OpConverterRegister<Pool> a("Pooling");

class Pool3D : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
        const caffe::Pooling3DParameter& p = parameters.pooling3d_param();
        auto pool3d                        = new MNN::Pool3DT;
        dstOp->main.value                  = pool3d;
        auto poolingType                   = p.pool();
        if (poolingType == caffe::Pooling3DParameter::MAX) {
            pool3d->type = MNN::PoolType_MAXPOOL;
        } else if (poolingType == caffe::Pooling3DParameter::AVE) {
            pool3d->type = MNN::PoolType_AVEPOOL;
        } else {
            DLOG(FATAL) << "Pool type not support! ==> " << parameters.name();
        }
        {
            const int kernel_size = p.kernel_size();
            const int kernel_depth = p.kernel_depth();
            pool3d->kernels = std::vector<int>({kernel_depth, kernel_size, kernel_size});
        }
        {
            const int stride = p.stride();
            const int temporal_stride = p.temporal_stride();
            pool3d->strides = std::vector<int>({temporal_stride, stride, stride});
        }
        {
            const int pad = p.pad();
            const int temporal_pad = p.has_temporal_pad() ? p.temporal_pad() : 0;
            pool3d->pads = std::vector<int>({temporal_pad, pad, pad});
        }
        pool3d->padType = MNN::PoolPadType_CAFFE;
    }
    Pool3D() {
    }
    virtual ~Pool3D() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Pooling3D;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Pool3D;
    }
};

static OpConverterRegister<Pool3D> b("Pooling3D");
