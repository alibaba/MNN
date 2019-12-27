//
//  NeuralNetWorkOp.cpp
//  MNN
//
//  Created by MNN on 2019/06/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <map>
#include <numeric>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/MNNDefine.h>
#include "MNN_generated.h"
#include "Utils.hpp"
namespace MNN {
namespace Express {
static PadMode _convertPadMode(PaddingMode mode) {
    switch (mode) {
        case CAFFE:
            return PadMode_CAFFE;
        case VALID:
            return PadMode_VALID;
        case SAME:
            return PadMode_SAME;
        default:
            break;
    }
    return PadMode_CAFFE;
}
static PoolPadType _convertPoollingPadMode(PaddingMode mode) {
    switch (mode) {
        case CAFFE:
            return PoolPadType_CAFFE;
        case VALID:
            return PoolPadType_VALID;
        case SAME:
            return PoolPadType_SAME;
        default:
            break;
    }
    return PoolPadType_CAFFE;
}

VARP _Input(INTS dims, Dimensionformat format, halide_type_t type) {
    Variable::Info info;
    info.dim = std::move(dims);
    info.order = format;
    info.type = type;
    info.ptr = nullptr;
    return (Variable::create(Expr::create(std::move(info))));
}
VARP _Scalar(const void* ptr, halide_type_t type) {
    Variable::Info info;
    info.dim = {};
    info.order = NHWC;
    info.type = type;
    info.ptr = (void*)ptr;
    return (Variable::create(Expr::create(std::move(info))));
}
VARP _Const(const void* ptr, INTS dims, Dimensionformat format, halide_type_t type) {
    Variable::Info info;
    info.dim = std::move(dims);
    info.order = format;
    info.type = type;
    info.ptr = (void*)ptr;
    return (Variable::create(Expr::create(std::move(info))));
}

VARP _Const(float value, INTS dims, Dimensionformat format) {
    auto size                          = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    std::vector<float> values;
    values.resize(size);
    for (int i = 0; i < size; ++i) {
        values[i] = value;
    }
    Variable::Info info;
    info.dim = std::move(dims);
    info.order = format;
    info.type = halide_type_of<float>();
    info.ptr = (void*)values.data();
    return (Variable::create(Expr::create(std::move(info))));
}

VARP _TrainableParam(const void* ptr, INTS dims, Dimensionformat format, halide_type_t type) {
    auto v = _Const(ptr, dims, format, type);
    v.fix(VARP::TRAINABLE);
    return v;
}
VARP _TrainableParam(float value, INTS dims, Dimensionformat format) {
    auto v = _Const(value, dims, format);
    v.fix(VARP::TRAINABLE);
    return v;
}

VARP _Conv(VARP weight, VARP bias, VARP x, PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Convolution;
    auto shape   = weight->getInfo();
    if (NHWC == shape->order) {
        weight = _Transpose(weight, {0, 3, 1, 2});
        shape  = weight->getInfo();
    }
    auto channel    = std::vector<int>{shape->dim[1], shape->dim[0]};
    auto kernelSize = std::vector<int>{shape->dim[3], shape->dim[2]};
    if (1 == channel[1] && channel[0] == group) {
        convOp->type = OpType_ConvolutionDepthwise;
        channel[1] = group;
    }
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    conv2D->common->padX        = pads[0];
    conv2D->common->padY        = pads[1];
    conv2D->common->padMode     = _convertPadMode(pad);
    conv2D->common->strideX     = stride[0];
    conv2D->common->strideY     = stride[1];
    conv2D->common->group       = group;
    conv2D->common->outputCount = channel[1];
    conv2D->common->inputCount  = channel[0];
    conv2D->common->dilateX     = dilate[0];
    conv2D->common->dilateY     = dilate[1];
    conv2D->common->kernelX     = kernelSize[0];
    conv2D->common->kernelY     = kernelSize[1];
    return (Variable::create(Expr::create(convOp.get(), {x, weight, bias})));
}
VARP _Conv(std::vector<float>&& weight, std::vector<float>&& bias, VARP x, INTS channel, INTS kernelSize,
           PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Convolution;
    if (channel[0] == channel[1] && channel[0] == group) {
        convOp->type = OpType_ConvolutionDepthwise;
    }
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    conv2D->common->padMode     = _convertPadMode(pad);
    conv2D->common->padX        = pads[0];
    conv2D->common->padY        = pads[1];
    conv2D->common->strideX     = stride[0];
    conv2D->common->strideY     = stride[1];
    conv2D->common->group       = group;
    conv2D->common->outputCount = channel[1];
    conv2D->common->inputCount  = channel[0];
    conv2D->common->dilateX     = dilate[0];
    conv2D->common->dilateY     = dilate[1];
    conv2D->common->kernelX     = kernelSize[0];
    conv2D->common->kernelY     = kernelSize[1];
    MNN_ASSERT(weight.size() == channel[1] * (channel[0] / group) * kernelSize[0] * kernelSize[1]);
    conv2D->weight = std::move(weight);
    MNN_ASSERT(bias.size() == channel[1]);
    conv2D->bias = std::move(bias);
    return (Variable::create(Expr::create(convOp.get(), {x})));
}

VARP _Conv(float weight, float bias, VARP x, INTS channel, INTS kernelSize, PaddingMode pad, INTS stride, INTS dilate,
           int group) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type = OpType_Convolution;
    if (channel[0] == channel[1] && channel[0] == group) {
        convOp->type = OpType_ConvolutionDepthwise;
    }
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    conv2D->common->padMode     = _convertPadMode(pad);
    conv2D->common->strideX     = stride[0];
    conv2D->common->strideY     = stride[1];
    conv2D->common->group       = group;
    conv2D->common->outputCount = channel[1];
    conv2D->common->inputCount  = channel[0];
    conv2D->common->dilateX     = dilate[0];
    conv2D->common->dilateY     = dilate[1];
    conv2D->common->kernelX     = kernelSize[0];
    conv2D->common->kernelY     = kernelSize[1];
    conv2D->weight.resize(channel[1] * (channel[0] / group) * kernelSize[0] * kernelSize[1]);
    std::fill(conv2D->weight.begin(), conv2D->weight.end(), weight);
    conv2D->bias.resize(channel[1]);
    std::fill(conv2D->bias.begin(), conv2D->bias.end(), bias);
    return (Variable::create(Expr::create(convOp.get(), {x})));
}

VARP _Deconv(VARP weight, VARP bias, VARP x, PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type    = OpType_Deconvolution;
    auto shape      = weight->getInfo();
    auto channel    = std::vector<int>{shape->dim[1], shape->dim[0]};
    auto kernelSize = std::vector<int>{shape->dim[3], shape->dim[2]};
    if (1 == channel[1] && channel[0] == group) {
        convOp->type = OpType_DeconvolutionDepthwise;
        channel[1] = group;
    }
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    static std::map<PaddingMode, PadMode> padmap{
        {CAFFE, PadMode_CAFFE},
        {VALID, PadMode_VALID},
        {SAME, PadMode_SAME},
    };
    conv2D->common->padX        = pads[0];
    conv2D->common->padY        = pads[1];
    conv2D->common->padMode     = padmap[pad];
    conv2D->common->strideX     = stride[0];
    conv2D->common->strideY     = stride[1];
    conv2D->common->group       = group;
    conv2D->common->outputCount = channel[0];
    conv2D->common->inputCount  = channel[1];
    conv2D->common->dilateX     = dilate[0];
    conv2D->common->dilateY     = dilate[1];
    conv2D->common->kernelX     = kernelSize[0];
    conv2D->common->kernelY     = kernelSize[1];
    if (nullptr != bias) {
        return (Variable::create(Expr::create(std::move(convOp), {x, weight, bias})));
    }
    return (Variable::create(Expr::create(std::move(convOp), {x, weight})));
}

static VARP _Pool(VARP x, INTS kernel, INTS stride, PoolType type, PaddingMode pad, INTS pads) {
    std::unique_ptr<OpT> pool(new OpT);
    pool->type       = OpType_Pooling;
    pool->main.type  = OpParameter_Pool;
    pool->main.value = new PoolT;
    if (kernel[0] == -1 && kernel[1] == -1) {
        pool->main.AsPool()->isGlobal = true;
    }
    pool->main.AsPool()->padX = 0;
    pool->main.AsPool()->padY = 0;
    if (pads.size() >= 2) {
        pool->main.AsPool()->padX = pads[0];
        pool->main.AsPool()->padY = pads[1];
    }
    pool->main.AsPool()->padType = _convertPoollingPadMode(pad);
    pool->main.AsPool()->kernelX = kernel[0];
    pool->main.AsPool()->kernelY = kernel[1];
    pool->main.AsPool()->strideX = stride[0];
    pool->main.AsPool()->strideY = stride[1];
    pool->main.AsPool()->type    = type;
    return (Variable::create(Expr::create(pool.get(), {x})));
}

VARP _AvePool(VARP x, INTS kernel, INTS stride, PaddingMode pad, INTS pads) {
    return _Pool(x, kernel, stride, PoolType_AVEPOOL, pad, pads);
}

VARP _MaxPool(VARP x, INTS kernel, INTS stride, PaddingMode pad, INTS pads) {
    return _Pool(x, kernel, stride, PoolType_MAXPOOL, pad, pads);
}
VARP _Reshape(VARP x, INTS dim, Dimensionformat format) {
    std::unique_ptr<OpT> reshape(new OpT);
    reshape->type                      = OpType_Reshape;
    reshape->main.type                 = OpParameter_Reshape;
    reshape->main.value                = new ReshapeT;
    reshape->main.AsReshape()->dims    = dim;
    reshape->main.AsReshape()->dimType = (MNN_DATA_FORMAT)Utils::convertFormat(format);
    return (Variable::create(Expr::create(reshape.get(), {x})));
}
VARP _Reshape(VARP x, VARP shape) {
    std::unique_ptr<OpT> reshape(new OpT);
    reshape->type                      = OpType_Reshape;
    reshape->main.type                 = OpParameter_Reshape;
    reshape->main.value                = new ReshapeT;
    reshape->main.AsReshape()->dimType = MNN_DATA_FORMAT_NCHW;
    return (Variable::create(Expr::create(reshape.get(), {x, shape})));
}
VARP _Scale(VARP x, int channels, std::vector<float>&& scales, std::vector<float>&& bias) {
    std::unique_ptr<OpT> scale(new OpT);
    scale->type                      = OpType_Scale;
    scale->main.type                 = OpParameter_Scale;
    scale->main.value                = new ScaleT;
    scale->main.AsScale()->channels  = channels;
    scale->main.AsScale()->scaleData = std::move(scales);
    scale->main.AsScale()->biasData  = std::move(bias);
    return (Variable::create(Expr::create(std::move(scale), {x})));
}
VARP _Relu(VARP x, float slope) {
    std::unique_ptr<OpT> relu(new OpT);
    relu->type                 = OpType_ReLU;
    relu->main.type            = OpParameter_Relu;
    relu->main.value           = new ReluT;
    relu->main.AsRelu()->slope = slope;
    return (Variable::create(Expr::create(relu.get(), {x})));
}
VARP _Relu6(VARP x) {
    std::unique_ptr<OpT> relu(new OpT);
    relu->type = OpType_ReLU6;
    return (Variable::create(Expr::create(relu.get(), {x})));
}
VARP _PRelu(VARP x, std::vector<float>&& slopes) {
    std::unique_ptr<OpT> prelu(new OpT);
    prelu->type                       = OpType_PReLU;
    prelu->main.type                  = OpParameter_PRelu;
    prelu->main.value                 = new PReluT;
    prelu->main.AsPRelu()->slope      = slopes;
    prelu->main.AsPRelu()->slopeCount = slopes.size();
    return (Variable::create(Expr::create(prelu.get(), {x})));
}

VARP _Softmax(VARP x, int axis) {
    std::unique_ptr<OpT> softmax(new OpT);
    softmax->type                = OpType_Softmax;
    softmax->main.type           = OpParameter_Axis;
    softmax->main.value          = new AxisT;
    softmax->main.AsAxis()->axis = axis;
    return (Variable::create(Expr::create(softmax.get(), {x})));
}

VARP _Softplus(VARP x) {
    return _Log(_Add(_Exp(x), _Const(1)));
}

VARP _Softsign(VARP x) {
    return _Divide(x, _Add(_Abs(x), _Const(1)));
}

VARP _Concat(VARPS xs, int axis) {
    std::unique_ptr<OpT> concat(new OpT);
    concat->type                = OpType_Concat;
    concat->main.type           = OpParameter_Axis;
    concat->main.value          = new AxisT;
    concat->main.AsAxis()->axis = axis;
    return (Variable::create(Expr::create(concat.get(), xs)));
}

VARP _Convert(VARP x, Dimensionformat dest) {
    std::unique_ptr<OpT> convert(new OpT);
    if (nullptr != x->getInfo()) {
        auto source = x->getInfo()->order;
        if (source == dest) {
            return x;
        }
    }
    convert->type                               = OpType_ConvertTensor;
    convert->main.type                          = OpParameter_TensorConvertInfo;
    convert->main.value                         = new TensorConvertInfoT;
    convert->main.AsTensorConvertInfo()->dest   = (MNN_DATA_FORMAT)Utils::convertFormat(dest);
    return (Variable::create(Expr::create(convert.get(), {x})));
}

std::vector<VARP> _Split(VARP x, INTS points, int axis) {
    MNN_ASSERT(points.size() >= 1);
    std::unique_ptr<OpT> op(new OpT);
    op->type                        = OpType_Slice;
    op->main.type                   = OpParameter_Slice;
    op->main.value                  = new SliceT;
    op->main.AsSlice()->axis        = axis;
    op->main.AsSlice()->sourceType  = NetSource_TENSORFLOW;
    op->main.AsSlice()->slicePoints = points;

    int slices = points.size() == 1 ? points[0] : (int)points.size();
    EXPRP expr = Expr::create(std::move(op), {x}, slices);
    std::vector<VARP> res;
    for (int i = 0; i < slices; ++i) {
        res.emplace_back(Variable::create(expr, i));
    }
    return res;
}

VARP _Slice(VARP x, VARP starts, VARP sizes) {
    std::unique_ptr<OpT> slice(new OpT);
    slice->type = OpType_SliceTf;
    return (Variable::create(Expr::create(slice.get(), {x, starts, sizes})));
}

VARP _StridedSlice(VARP x, VARP begin, VARP end, VARP strided, halide_type_t type, int32_t beginMask,
                   int32_t endMask, int32_t ellipsisMask, int32_t newAxisMask, int32_t shrinkAxisMask) {
    std::unique_ptr<OpT> op(new OpT);
    op->type                        = OpType_StridedSlice;
    op->main.type                   = OpParameter_StridedSliceParam;
    op->main.value                  = new StridedSliceParamT;

    op->main.AsStridedSliceParam()->T              = (MNN::DataType)Utils::convertDataType(type);;
    op->main.AsStridedSliceParam()->beginMask      = beginMask;
    op->main.AsStridedSliceParam()->endMask        = endMask;
    op->main.AsStridedSliceParam()->ellipsisMask   = ellipsisMask;
    op->main.AsStridedSliceParam()->newAxisMask    = newAxisMask;
    op->main.AsStridedSliceParam()->shrinkAxisMask = shrinkAxisMask;
    return (Variable::create(Expr::create(op.get(), {x, begin, end, strided})));
}

VARP _Transpose(VARP x, INTS perm) {
    auto permVar = _Const((const void*)perm.data(), {static_cast<int>(perm.size())}, NHWC, halide_type_of<int>());
    return _Transpose(x, permVar);
}
VARP _Transpose(VARP x, VARP perm) {
    std::unique_ptr<OpT> transpose(new OpT);
    transpose->type                      = OpType_Transpose;
    transpose->main.type                 = OpParameter_Transpose;
    transpose->main.value                = new TransposeT;
    transpose->main.AsTranspose()->Tperm = DataType_DT_INT32;
    return (Variable::create(Expr::create(std::move(transpose), {x, perm})));
}

VARP _ChannelShuffle(VARP x, int group) {
    x = _Convert(x, NHWC);
    x = _Reshape(x, {0, 0, 0, group, -1}, NHWC);
    x = _Transpose(x, {0, 1, 2, 4, 3});
    x = _Reshape(x, {0, 0, 0, -1}, NHWC);
    x = _Convert(x, NC4HW4);
    return x;
}
VARP _ReverseSequence(VARP x, VARP y, int batchDim, int seqDim) {
    std::unique_ptr<OpT> op(new OpT);
    op->type                                    = OpType_ReverseSequence;
    op->main.type                               = OpParameter_ReverseSequenceParam;
    op->main.value                              = new ReverseSequenceParamT;
    op->main.AsReverseSequenceParam()->batchDim = batchDim;
    op->main.AsReverseSequenceParam()->seqDim   = seqDim;
    return (Variable::create(Expr::create(op.get(), {x, y})));
}
VARP _ChangeInputFormat(VARP x, Dimensionformat requireInput) {
    if (nullptr == x || nullptr == x->getInfo()) {
        return nullptr;
    }
    if (x->getInfo()->order == requireInput) {
        return x;
    }
    auto input   = _Input(x->getInfo()->dim, requireInput, x->getInfo()->type);
    auto convert = _Convert(input, x->getInfo()->order);
    Variable::replace(x, convert);
    return input;
}

VARP _Clone(VARP source, bool deepCopy) {
    if (nullptr == source || nullptr == source->expr().first) {
        return nullptr;
    }
    if (!deepCopy) {
        return Variable::create(source->expr().first, source->expr().second);
    }
    auto info      = source->getInfo();
    auto sourcePtr = source->readMap<void>();
    if (nullptr == info || nullptr == sourcePtr) {
        MNN_ERROR("Source Buffer Not Available\n");
        return nullptr;
    }
    auto inputVar = _Input(info->dim, info->order, info->type);
    auto destPtr  = inputVar->writeMap<void>();
    if (nullptr == destPtr) {
        MNN_ERROR("Alloc Buffer Error\n");
        return nullptr;
    }
    ::memcpy(destPtr, sourcePtr, info->size * info->type.bytes());
    return inputVar;
}
VARP _Conv2DBackPropFilter(VARP weight, VARP input, VARP inputGrad, PaddingMode pad, INTS stride, INTS dilate,
                           int group, INTS pads) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type       = OpType_Conv2DBackPropFilter;
    auto shape         = weight->getInfo();
    auto channel       = std::vector<int>{shape->dim[1], shape->dim[0]};
    auto kernelSize    = std::vector<int>{shape->dim[3], shape->dim[2]};
    convOp->main.type  = OpParameter_Convolution2D;
    convOp->main.value = new Convolution2DT;
    auto conv2D        = convOp->main.AsConvolution2D();
    conv2D->common.reset(new Convolution2DCommonT);
    conv2D->common->padX        = pads[0];
    conv2D->common->padY        = pads[1];
    conv2D->common->padMode     = _convertPadMode(pad);
    conv2D->common->strideX     = stride[0];
    conv2D->common->strideY     = stride[1];
    conv2D->common->group       = group;
    conv2D->common->outputCount = channel[1];
    conv2D->common->inputCount  = channel[0];
    conv2D->common->dilateX     = dilate[0];
    conv2D->common->dilateY     = dilate[1];
    conv2D->common->kernelX     = kernelSize[0];
    conv2D->common->kernelY     = kernelSize[1];
    INTS weightDims             = {channel[1], channel[0] / group, kernelSize[1], kernelSize[0]};

    return Variable::create(Expr::create(std::move(convOp), {weight, input, inputGrad}));
}

VARP _PoolGrad(VARP originInput, VARP originOutput, VARP inputGrad, INTS kernel, INTS stride, PoolingMode type,
               PaddingMode pad, INTS pads) {
    std::unique_ptr<OpT> pool(new OpT);
    pool->type       = OpType_PoolGrad;
    pool->main.type  = OpParameter_Pool;
    pool->main.value = new PoolT;
    if (kernel[0] == -1 && kernel[1] == -1) {
        pool->main.AsPool()->isGlobal = true;
    }
    pool->main.AsPool()->padX = 0;
    pool->main.AsPool()->padY = 0;
    if (pads.size() >= 2) {
        pool->main.AsPool()->padX = pads[0];
        pool->main.AsPool()->padY = pads[1];
    }
    pool->main.AsPool()->padType = _convertPoollingPadMode(pad);
    pool->main.AsPool()->kernelX = kernel[0];
    pool->main.AsPool()->kernelY = kernel[1];
    pool->main.AsPool()->strideX = stride[0];
    pool->main.AsPool()->strideY = stride[1];
    pool->main.AsPool()->type    = (PoolType)type;
    return (Variable::create(Expr::create(std::move(pool), {originInput, originOutput, inputGrad})));
}

VARP _Crop(VARP x, VARP s, int axis, INTS offset) {
    std::unique_ptr<OpT> crop(new OpT);
    crop->type                  = OpType_Crop;
    crop->main.type             = OpParameter_Crop;
    crop->main.value            = new CropT;
    crop->main.AsCrop()->axis   = axis;
    crop->main.AsCrop()->offset = offset;
    return (Variable::create(Expr::create(std::move(crop), {x, s})));
}
VARP _Resize(VARP x, float xScale, float yScale) {
    std::unique_ptr<OpT> resize(new OpT);
    resize->type                    = OpType_Resize;
    resize->main.type               = OpParameter_Resize;
    resize->main.value              = new ResizeT;
    resize->main.AsResize()->xScale = xScale;
    resize->main.AsResize()->yScale = yScale;
    return (Variable::create(Expr::create(std::move(resize), {x})));
}
VARP _Pad(VARP x, VARP pads, PadValueMode mode) {
    std::unique_ptr<OpT> pad(new OpT);
    pad->type       = OpType_Padding;
    pad->main.type  = OpParameter_PadParam;
    pad->main.value = new PadParamT;
    switch (mode) {
        case CONSTANT:
            pad->main.AsPadParam()->mode = MNN::PadValueMode_CONSTANT;
            break;
        case SYMMETRIC:
            pad->main.AsPadParam()->mode = MNN::PadValueMode_SYMMETRIC;
            break;
        case REFLECT:
            pad->main.AsPadParam()->mode = MNN::PadValueMode_REFLECT;
            break;
        default:
            pad->main.AsPadParam()->mode = MNN::PadValueMode_CONSTANT;
            break;
    }
    return (Variable::create(Expr::create(std::move(pad), {x, pads})));
}
VARP _ExpandDims(VARP x, int axis) {
    std::unique_ptr<OpT> expand(new OpT);
    expand->type                      = OpType_ExpandDims;
    expand->main.type                 = OpParameter_ExpandDims;
    expand->main.value                = new ExpandDimsT;
    expand->main.AsExpandDims()->axis = axis;
    return (Variable::create(Expr::create(std::move(expand), {x})));
}
VARP _ExpandDims(VARP x, VARP axis) {
    std::unique_ptr<OpT> expand(new OpT);
    expand->type       = OpType_ExpandDims;
    expand->main.type  = OpParameter_ExpandDims;
    expand->main.value = new ExpandDimsT;
    return (Variable::create(Expr::create(std::move(expand), {x, axis})));
}

VARP _Shape(VARP x) {
    std::unique_ptr<OpT> shape(new OpT);
    shape->type = OpType_Shape;
    return (Variable::create(Expr::create(std::move(shape), {x})));
}
/*Stacks a list of rank-R variables into one rank-(R+1) variable.
Packs the list of variables in `values` into a ariable with rank one higher than each variable in values,
by packing them along the axis dimension. 
Given a list of length N of variables of shape (A, B, C);
if axis == 0 then the output variable will have the shape (N, A, B, C). 
if axis == 1 then the output variable will have the shape (A, N, B, C). Etc.
Args:
values: A list of variable objects with the same shape and type.
axis: An int. The axis to stack along. Defaults to the first dimension. Negative values wrap around, 
so the valid range is [-(R+1), R+1).
Returns:
output: A stacked variable with the same type as `values`.
*/
VARP _Stack(VARPS values, int axis) {
    std::unique_ptr<OpT> pack(new OpT);
    pack->type                         = OpType_Pack;
    MNN_ASSERT(values.size()>0);
    auto info_first = values[0]->getInfo();
    MNN_ASSERT(nullptr != info_first);
    pack->main.type                    = OpParameter_PackParam;
    pack->main.value                   = new PackParamT;
    pack->main.AsPackParam()->dataType = (MNN::DataType)Utils::convertDataType(info_first->type);
    pack->main.AsPackParam()->axis     = axis;
    return (Variable::create(Expr::create(std::move(pack), values)));
}
VARP _CropAndResize(VARP image, VARP boxes, VARP indexes, VARP sizes, float extrapolation, InterpolationMethod method) {
    std::unique_ptr<OpT> car(new OpT);
    car->type                                       = OpType_CropAndResize;
    car->main.type                                  = OpParameter_CropAndResize;
    car->main.value                                 = new CropAndResizeT;
    car->main.AsCropAndResize()->extrapolationValue = extrapolation;
    switch (method) {
        case NEAREST:
            car->main.AsCropAndResize()->method = CropAndResizeMethod_NEAREST;
            break;
        case BILINEAR:
        default:
            car->main.AsCropAndResize()->method = CropAndResizeMethod_BILINEAR;
            break;
    }
    return (Variable::create(Expr::create(std::move(car), {image, boxes, indexes, sizes})));
}
VARP _Fill(VARP s, VARP v) {
    std::unique_ptr<OpT> fill(new OpT);
    fill->type       = OpType_Fill;
    fill->main.type  = OpParameter_Fill;
    fill->main.value = new FillT;
    return (Variable::create(Expr::create(std::move(fill), {s, v})));
}
VARP _Tile(VARP x, VARP mul) {
    std::unique_ptr<OpT> tile(new OpT);
    tile->type = OpType_Tile;
    return (Variable::create(Expr::create(std::move(tile), {x, mul})));
}
VARP _Gather(VARP embedding, VARP indices) {
    std::unique_ptr<OpT> gather(new OpT);
    gather->type       = OpType_Gather;
    gather->main.value = new GatherT;
    return (Variable::create(Expr::create(std::move(gather), {embedding, indices})));
}
VARP _GatherV2(VARP params, VARP indices, VARP axis) {
    std::unique_ptr<OpT> gather(new OpT);
    gather->type       = OpType_GatherV2;
    gather->main.value = new GatherV2T;
    if (axis.get()) {
        return (Variable::create(Expr::create(std::move(gather), {params, indices, axis})));
    } else {
        return (Variable::create(Expr::create(std::move(gather), {params, indices})));
    }
}

VARP _Squeeze(VARP x, INTS axes) {
    std::unique_ptr<OpT> squeeze(new OpT);
    squeeze->type             = OpType_Squeeze;
    auto squeezeParam         = new SqueezeParamT;
    squeezeParam->squeezeDims = axes;
    squeeze->main.type        = OpParameter_SqueezeParam;
    squeeze->main.value       = squeezeParam;
    return Variable::create(Expr::create(std::move(squeeze), {x}));
}

VARP _Unsqueeze(VARP x, INTS axes) {
    std::unique_ptr<OpT> squeeze(new OpT);
    squeeze->type             = OpType_Unsqueeze;
    auto squeezeParam         = new SqueezeParamT;
    squeezeParam->squeezeDims = axes;
    squeeze->main.type        = OpParameter_SqueezeParam;
    squeeze->main.value       = squeezeParam;
    return Variable::create(Expr::create(std::move(squeeze), {x}));
}
/*Computes exponential linear: alpha * (exp(features) - 1) if < 0, features otherwise.
features: A variable of type Halide_Type_Float
alpha: Alpha factor (positive float)
Returns:
A variable. Has the same type as features.
*/
VARP _Elu(VARP features, float alpha) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_ELU;
    auto eluParam = new ELUT;
    op->main.type = OpParameter_ELU;
    eluParam->alpha = alpha;
    op->main.value = eluParam;
    return (Variable::create(Expr::create(std::move(op), {features})));
}
/*Computes the size of the variable
Args:
input: A variable of type Halide_Type_Float or Halide_Type_Int
Returns:
A variable. The shape is (), and type is Halide_Type_Int
*/
VARP _Size(VARP input) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_Size;
    return (Variable::create(Expr::create(std::move(op), {input})));
}

/*Computes scaled exponential linear: scale * alpha * (exp(features) - 1) if < 0, scale * features otherwise.
Args:
features: A variable of type Halide_Type_Float
scale: Scaling factor (positive float)
alpha: Alpha factor (positive float)
Returns:
A variable. Has the same type as features.
*/
VARP _Selu(VARP features, float scale, float alpha) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_Selu;
    auto seluParam = new SeluT;
    op->main.type = OpParameter_Selu;
    seluParam->scale = scale;
    seluParam->alpha = alpha;
    op->main.value = seluParam;
    return (Variable::create(Expr::create(std::move(op), {features})));

}
/*Gather slices from params into a variable with shape specified by indices.
Args:
params: A variable. The variables from which to gather values.
indices: A variable. Must be one of the following types: Halide_Type_Int.
Returns:
A variable. Has the same type as params.
*/
VARP _GatherND(VARP params, VARP indices) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_GatherND;
    return (Variable::create(Expr::create(std::move(op), {params, indices})));
}

/*BatchToSpace for N-D variables
This operation reshapes the "batch" dimension 0 into M + 1 dimensions of shape block_shape + [batch], 
interleaves these blocks back into the grid defined by the spatial dimensions [1, ..., M], 
to obtain a result with the same rank as the input. 
The spatial dimensions of this intermediate result are then optionally cropped according to crops to 
produce the output. This is the reverse of SpaceToBatch. See below for a precise description.
Arguments:
input: must be 4-D with NC4HW4 format. N-D with shape input_shape = [batch] + spatial_shape + remaining_shape, where spatial_shape has M dimensions.
block_shape: 1-D with shape [M], all values must be >= 1.
crops: 2-D with shape [M, 2], all values must be >= 0. crops[i] = [crop_start, crop_end] specifies the amount to crop from input dimension i + 1, 
which corresponds to spatial dimension i. It is required that crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1].
This operation is equivalent to the following steps:
Reshape input to reshaped of shape: [block_shape[0], ..., block_shape[M-1], batch / prod(block_shape), 
input_shape[1], ..., input_shape[N-1]]
Permute dimensions of reshaped to produce permuted of shape 
[batch / prod(block_shape),input_shape[1], block_shape[0], ..., input_shape[M], block_shape[M-1],input_shape[M+1], ..., input_shape[N-1]]
Reshape permuted to produce reshaped_permuted of shape 
[batch / prod(block_shape),input_shape[1] * block_shape[0], ..., input_shape[M] * block_shape[M-1],input_shape[M+1], ..., input_shape[N-1]]
Crop the start and end of dimensions [1, ..., M] of reshaped_permuted according to crops to produce the output of shape: 
[batch / prod(block_shape),input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1], ..., input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],input_shape[M+1], ..., input_shape[N-1]]
Some examples:
for the following input of shape [4, 1, 1, 3], block_shape = [2, 2], and crops = [[0, 0], [0, 0]]:
[[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
The output variable has shape [1, 2, 2, 3] and value:
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
Returns:
Output: The output variable
*/

VARP _BatchToSpaceND(VARP input, VARP block_shape, VARP crops) {
    std::unique_ptr<OpT> op(new OpT);
    std::unique_ptr<BlobT> blob_blockShape(new BlobT);
    std::unique_ptr<BlobT> blob_paddings(new BlobT);
    
    auto info_block_shape = block_shape->getInfo();
    auto info_crops = crops->getInfo();
    MNN_ASSERT(info_block_shape != nullptr);
    MNN_ASSERT(info_crops != nullptr);
    MNN_ASSERT(halide_type_int == info_block_shape->type.code);
    MNN_ASSERT(halide_type_int == info_crops->type.code);
  
    blob_blockShape->dims = info_block_shape->dim;
    blob_blockShape->dataFormat = (MNN_DATA_FORMAT)Utils::convertFormat(info_block_shape->order);
    blob_blockShape->dataType = (MNN::DataType)Utils::convertDataType(info_block_shape->type);
    auto data_block_shape = block_shape->readMap<int>();
    for (int i=0; i<info_block_shape->size; i++)
    {
        blob_blockShape->int32s.emplace_back(data_block_shape[i]);
    }
    blob_paddings->dims = info_crops->dim;
    blob_paddings->dataFormat = (MNN_DATA_FORMAT)Utils::convertFormat(info_crops->order);
    blob_paddings->dataType = (MNN::DataType)Utils::convertDataType(info_crops->type);
    auto data_crop = crops->readMap<int>();
    for (int i=0; i<info_crops->size; i++)
    {
        blob_paddings->int32s.emplace_back(data_crop[i]);
    }
    op->main.type                         = OpParameter_SpaceBatch;
    op->type                              = OpType_BatchToSpaceND;
    op->main.value                        = new SpaceBatchT;
    op->main.AsSpaceBatch()->blockShape = std::move(blob_blockShape);
    op->main.AsSpaceBatch()->padding = std::move(blob_paddings);
    return Variable::create(Expr::create(std::move(op), {input}));
}
/*Copies a variable setting everything outside a central band in each innermost matrix.
Arguments:
input: Rank k variable.
num_lower: Number of subdiagonals to keep. If negative, keep entire lower triangle.
num_upper: Number of superdiagonals to keep. If negative, keep entire upper triangle.
Returns:
Output: Rank k variable of the same shape as input. The extracted banded tensor.
*/
VARP _MatrixBandPart(VARP input, VARP num_lower, VARP num_upper) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_MatrixBandPart;
    auto lrnParam = new LRNT;
    op->main.type = OpParameter_NONE;
    return (Variable::create(Expr::create(std::move(op), {input, num_lower, num_upper})));
}
/*Calculates the mean and variance of x.
Args:
x: A variable. must be 4-D with NC4HW4 format.
axes: Array of ints. Axes along which to compute mean and variance. Ignored for this implementation: must be {2, 3}
shift: Not used in the current implementation. 
keepdims: produce moments with the same dimensionality as the input.  Ignored for this implementation: must be true.
Returns:
Two variable objects: mean and variance.
*/
std::vector<VARP> _Moments(VARP x, INTS axis, VARP shift, bool keepDims) {
    std::unique_ptr<OpT> op(new OpT);
    axis = {2, 3};
    keepDims = true;
    // if axis != {2,3} or keepDims != true, print warning. 
    // ignore shift.
    op->type       = OpType_Moments;
    auto momentsParam = new MomentsParamT;
    op->main.type = OpParameter_MomentsParam;
    momentsParam->dim = axis;
    momentsParam->keepDims = keepDims;
    momentsParam->dType = (MNN::DataType)Utils::convertDataType(x->getInfo()->type);
    op->main.value = momentsParam;
    EXPRP expr = Expr::create(std::move(op), {x}, 2);
    std::vector<VARP> res;
    res.emplace_back(Variable::create(expr, 0));
    res.emplace_back(Variable::create(expr, 1));
    return res;
}
/*Computes the difference between two lists of numbers or strings.
Given a list x and a list y, this operation returns a list out that represents all values that are in x but not in y. 
The returned list out is sorted in the same order that the numbers appear in x (duplicates are preserved). 
This operation also returns a list idx that represents the position of each out element in x. 
Arguments:
x: 1-D variable of type Halide_Type_Int. Values to keep. 
y: 1-D variable of type Halide_Type_Int. Values to remove.
Returns:
Output out: 1-D variable of type Halide_Type_Int. Values present in x but not in y.
*/
VARP _SetDiff1D(VARP x, VARP y) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_SetDiff1D;
    op->main.type = OpParameter_NONE;
    op->main.value = nullptr;
    return Variable::create(Expr::create(std::move(op), {x, y}));
}
/*Rearranges blocks of spatial data, into depth. 
More specifically, it outputs a copy of the input variable where values from the height and width dimensions are moved to the depth dimension. 
The block_size indicates the input block size.
Non-overlapping blocks of size block_size x block_size are rearranged into depth at each location.
The depth of the output variable is block_size * block_size * input_depth.
The Y, X coordinates within each block of the input become the high order component of the output channel index.
The input variable's height and width must be divisible by block_size
Args:
input: A variable.
block_size: An int that is >= 2. The size of the spatial block.
Returns:
A variable. Has the same type as input.
*/
VARP _SpaceToDepth(VARP input, int block_size) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_SpaceToDepth;
    auto param =  new DepthSpaceParamT;
    param->blockSize = block_size;
    op->main.type = OpParameter_DepthSpaceParam;
    op->main.value = param;
    return Variable::create(Expr::create(std::move(op), {input}));
}

/*This operation divides "spatial" dimensions [1, ..., M] of the input into a grid of blocks of shape block_shape, 
and interleaves these blocks with the "batch" dimension 
such that in the output, the spatial dimensions [1, ..., M] correspond to the position within the grid,
and the batch dimension combines both the position within a spatial block and the original batch position.
Prior to division into blocks, the spatial dimensions of the input are optionally zero padded according to paddings. 
See below for a precise description.
Args:
input: A variable. must be 4-D with NC4HW4 format. N-D with shape input_shape = [batch] + spatial_shape + remaining_shape, where spatial_shape has M dimensions.
block_shape: A variable. Must be one of the following types: int32, int64. 1-D with shape [M], all values must be >= 1.
paddings: A variable. Must be one of the following types: int32, int64. 2-D with shape [M, 2], all values must be >= 0. paddings[i] = [pad_start, pad_end] specifies the padding for input dimension i + 1, which corresponds to spatial dimension i. It is required that block_shape[i] divides input_shape[i + 1] + pad_start + pad_end.
Returns:
A variable. Has the same type as input.
*/
VARP _SpaceToBatchND(VARP input, VARP block_shape, VARP paddings) {
    std::unique_ptr<OpT> op(new OpT);
    std::unique_ptr<BlobT> blob_blockShape(new BlobT);
    std::unique_ptr<BlobT> blob_paddings(new BlobT);
    op->type       = OpType_SpaceToBatchND;
    auto param =  new SpaceBatchT;
    auto info_block_shape = block_shape->getInfo();
    auto info_paddings = paddings->getInfo();
    MNN_ASSERT(info_block_shape != nullptr);
    MNN_ASSERT(info_paddings != nullptr);
    MNN_ASSERT(halide_type_int == info_block_shape->type.code);
    MNN_ASSERT(halide_type_int == info_paddings->type.code);
  
    blob_blockShape->dims = info_block_shape->dim;
    blob_blockShape->dataFormat = (MNN::MNN_DATA_FORMAT)Utils::convertFormat(info_block_shape->order);
    blob_blockShape->dataType = (MNN::DataType)Utils::convertDataType(info_block_shape->type);
    auto data_block_shape = block_shape->readMap<int>();
    for (int i=0; i<info_block_shape->size; i++)
    {
        blob_blockShape->int32s.emplace_back(data_block_shape[i]);
    }
    blob_paddings->dims = info_paddings->dim;
    blob_paddings->dataFormat = (MNN::MNN_DATA_FORMAT)Utils::convertFormat(info_paddings->order);
    blob_paddings->dataType = (MNN::DataType)Utils::convertDataType(info_paddings->type);
    auto data_paddings = paddings->readMap<int>();
    for (int i=0; i<info_paddings->size; i++)
    {
        blob_paddings->int32s.emplace_back(data_paddings[i]);
    }
    param->blockShape = std::move(blob_blockShape);
    param->padding = std::move(blob_paddings);
    op->main.type = OpParameter_SpaceBatch;
    op->main.value = param;
    return Variable::create(Expr::create(std::move(op), {input}));
}
/*Creates a variable with all elements set to zero.
Args:
input: A variable.
Returns:
A variable with all elements set to zero.
*/

VARP _ZerosLike(VARP input) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_ZerosLike;
    op->main.type = OpParameter_NONE;
    op->main.value = nullptr;
    return Variable::create(Expr::create(std::move(op), {input}));
}
/*Unpacks the given dimension of a rank-R tensor into rank-(R-1) variable.
For example, given a variable of shape (A, B, C, D);
If axis == 0 then the i'th variable in output is the slice value[i, :, :, :] and each variable in output will have shape (B, C, D). 
(Note that the dimension unpacked along is gone, unlike split).
If axis == 1 then the i'th variable in output is the slice value[:, i, :, :] and each variable in output will have shape (A, C, D). 
Args:
value: A rank R > 0 variable to be unstacked.
num: An int. The length of the dimension axis. Automatically inferred if None (the default).
axis: An int. The axis to unstack along. Defaults to the first dimension. Negative values wrap around, so the valid range is [-R, R).
Returns:
The list of variable objects unstacked from value.
*/
std::vector <VARP> _Unstack(VARP value, int axis) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_Unpack;
    auto info_value = value->getInfo();
    MNN_ASSERT(info_value != nullptr);
    auto dims = info_value->dim;
    auto dimsize = dims.size();
    MNN_ASSERT(dimsize > 1);
    axis = axis % dimsize;
    if(axis < 0) {
        axis += dimsize;
    }
    auto size = dims[axis];
    MNN_ASSERT(size > 0);
    auto axisParam = new AxisT;
    axisParam->axis = axis;
    op->main.type = OpParameter_Axis;
    op->main.value = axisParam;
    EXPRP expr = Expr::create(std::move(op), {value}, size);
    std::vector<VARP> res;
    for (int i = 0; i < size; ++i) {
        res.emplace_back(Variable::create(expr, i));
    }
    return res;   
}

/*Returns the rank of a variable.
Returns a 0-D int32 variable representing the rank of input.
Note: The rank of a variable is not the same as the rank of a matrix. 
It's the number of indices required to uniquely select each element of the variable. 
It's also known as "order", "degree", or "ndims."
Args:
input: A variable.
Returns:
A 0-D variable of type Halide_Type_Int
*/
VARP _Rank(VARP input) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_Rank;
    op->main.type = OpParameter_NONE;
    op->main.value = nullptr;
    return Variable::create(Expr::create(std::move(op), {input}));
}
/*Creates a sequence of numbers.
Args:
start: A 0-D variable (scalar). 
limit: A 0-D variable (scalar). 
delta: A 0-D variable (scalar). 
*/
VARP _Range(VARP start, VARP limit, VARP delta) {
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_Range;
    auto rangeParam = new RangeT;
    rangeParam->Tidx = (MNN::DataType)Utils::convertDataType(start->getInfo()->type);
    op->main.type = OpParameter_Range;
    op->main.value = rangeParam;
    return Variable::create(Expr::create(std::move(op), {start, limit, delta}));
}


VARP _Interp(VARPS xs, float widthScale, float heightScale, int outputWidth, int outputHeight, int resizeType, bool alignCorners) {
    std::unique_ptr<OpT> interp(new OpT);
    interp->type        = OpType_Interp;
    auto param          = new InterpT;
    param->widthScale   = widthScale;
    param->heightScale  = heightScale;
    param->outputWidth  = outputWidth;
    param->outputHeight = outputHeight;
    param->resizeType   = resizeType;
    param->alignCorners = alignCorners;
    interp->main.value  = param;
    interp->main.type   = OpParameter_Interp;
    return Variable::create(Expr::create(std::move(interp), xs));
}

} // namespace Express
} // namespace MNN
