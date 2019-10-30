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
#include "ExprCreator.hpp"
#include "MNNDefine.h"
#include "MNN_generated.h"
#include "Utils.hpp"
namespace MNN {
namespace Express {
static MNN_DATA_FORMAT _convertFormat(Dimensionformat format) {
    switch (format) {
        case NCHW:
            return MNN_DATA_FORMAT_NCHW;
        case NHWC:
            return MNN_DATA_FORMAT_NHWC;
        case NC4HW4:
            return MNN_DATA_FORMAT_NC4HW4;
        default:
            break;
    }
    return MNN_DATA_FORMAT_UNKNOWN;
}
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
    std::unique_ptr<OpT> input(new OpT);
    input->type                    = OpType_Input;
    input->main.type               = OpParameter_Input;
    input->main.value              = new InputT;
    input->main.AsInput()->dtype   = (MNN::DataType)Utils::convertDataType(type);
    MNN_ASSERT(input->main.AsInput()->dtype != DataType_DT_INVALID);
    input->main.AsInput()->dims    = dims;
    input->main.AsInput()->dformat = (MNN_DATA_FORMAT)Utils::convertFormat(format);
    return (Variable::create(Expr::create(input.get(), {})));
}
VARP _Const(const void* ptr, INTS dims, Dimensionformat format, halide_type_t type) {
    MNN_ASSERT(type.code == halide_type_float || type.code == halide_type_int);
    auto blob        = new BlobT;
    blob->dataFormat = (MNN_DATA_FORMAT)Utils::convertFormat(format);
    blob->dims       = dims;
    int length       = 1;
    for (int i = 0; i < dims.size(); ++i) {
        length *= dims[i];
    }
    if (type.code == halide_type_float) {
        blob->dataType = DataType_DT_FLOAT;
        blob->float32s.resize(length);
        ::memcpy(blob->float32s.data(), ptr, length * sizeof(float));
    } else {
        blob->dataType = DataType_DT_INT32;
        blob->int32s.resize(length);
        ::memcpy(blob->int32s.data(), ptr, length * sizeof(int));
    }
    std::unique_ptr<OpT> op(new OpT);
    op->type       = OpType_Const;
    op->main.type  = OpParameter_Blob;
    op->main.value = blob;
    return (Variable::create(Expr::create(op.get(), {})));
}

VARP _Const(float value, INTS dims, Dimensionformat format) {
    std::unique_ptr<OpT> constOp(new OpT);
    constOp->type                      = OpType_Const;
    constOp->main.type                 = OpParameter_Blob;
    constOp->main.value                = new BlobT;
    constOp->main.AsBlob()->dataType   = DataType_DT_FLOAT;
    constOp->main.AsBlob()->dataFormat = (MNN_DATA_FORMAT)Utils::convertFormat(format);
    constOp->main.AsBlob()->dims       = dims;
    auto size                          = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    constOp->main.AsBlob()->float32s.resize(size);
    for (int i = 0; i < size; ++i) {
        constOp->main.AsBlob()->float32s[i] = value;
    }
    return (Variable::create(Expr::create(constOp.get(), {})));
}
VARP _Conv(VARP weight, VARP bias, VARP x, PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads) {
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type    = OpType_Convolution;
    auto shape      = weight->getInfo();
    if (NHWC == shape->order) {
        weight = _Transpose(weight, {0, 3, 1, 2});
        shape = weight->getInfo();
    }
    auto channel    = std::vector<int>{shape->dim[1], shape->dim[0]};
    auto kernelSize = std::vector<int>{shape->dim[3], shape->dim[2]};
    if (channel[0] == channel[1] && channel[0] == group) {
        convOp->type = OpType_ConvolutionDepthwise;
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
    INTS weightDims             = {channel[1], channel[0] / group, kernelSize[1], kernelSize[0]};
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

VARP _Deconv(VARP weight, VARP bias, VARP x, PaddingMode pad, INTS stride,
             INTS dilate, int group, INTS pads){
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type    = OpType_Deconvolution;
    auto shape      = weight->getInfo();
    auto channel    = std::vector<int>{shape->dim[1], shape->dim[0]};
    auto kernelSize = std::vector<int>{shape->dim[3], shape->dim[2]};
    if (channel[0] == channel[1] && channel[0] == group) {
        convOp->type = OpType_DeconvolutionDepthwise;
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
    INTS weightDims             = {channel[1], channel[0] / group, kernelSize[1], kernelSize[0]};
    return (Variable::create(Expr::create(std::move(convOp), {x, weight, bias})));
}

static VARP _Pool(VARP x, INTS kernel, INTS stride, PoolType type, PaddingMode pad, INTS pads) {
    std::unique_ptr<OpT> pool(new OpT);
    pool->type       = OpType_Pooling;
    pool->main.type  = OpParameter_Pool;
    pool->main.value = new PoolT;
    if (kernel[0] == -1 && kernel[1] == -1) {
        pool->main.AsPool()->isGlobal = true;
    }
    pool->main.AsPool()->padX    = 0;
    pool->main.AsPool()->padY    = 0;
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
    reshape->main.AsReshape()->dimType = _convertFormat(format);
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

VARP _Softmax(VARP x, int axis) {
    std::unique_ptr<OpT> softmax(new OpT);
    softmax->type                = OpType_Softmax;
    softmax->main.type           = OpParameter_Axis;
    softmax->main.value          = new AxisT;
    softmax->main.AsAxis()->axis = axis;
    return (Variable::create(Expr::create(softmax.get(), {x})));
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
    if (nullptr == x->getInfo()) {
        return x;
    }
    auto source = x->getInfo()->order;
    if (source == dest) {
        return x;
    }
    convert->type                               = OpType_ConvertTensor;
    convert->main.type                          = OpParameter_TensorConvertInfo;
    convert->main.value                         = new TensorConvertInfoT;
    convert->main.AsTensorConvertInfo()->source = (MNN_DATA_FORMAT)Utils::convertFormat(source);
    convert->main.AsTensorConvertInfo()->dest   = (MNN_DATA_FORMAT)Utils::convertFormat(dest);
    return (Variable::create(Expr::create(convert.get(), {x})));
}

std::vector<VARP> _Slice(VARP x, INTS points, int axis) {
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
    op->type                        = OpType_ReverseSequence;
    op->main.type                   = OpParameter_ReverseSequenceParam;
    op->main.value                  = new ReverseSequenceParamT;
    op->main.AsReverseSequenceParam()->batchDim  = batchDim;
    op->main.AsReverseSequenceParam()->seqDim = seqDim;
    return (Variable::create(Expr::create(op.get(), {x, y})));
}
VARP _ChangeInputFormat(VARP x, Dimensionformat requireInput) {
    if (nullptr == x || nullptr == x->getInfo()) {
        return nullptr;
    }
    if (x->getInfo()->order == requireInput) {
        return x;
    }
    auto input = _Input(x->getInfo()->dim, requireInput, x->getInfo()->type);
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
    auto info = source->getInfo();
    auto sourcePtr = source->readMap<void>();
    if (nullptr == info || nullptr == sourcePtr) {
        MNN_ERROR("Source Buffer Not Available\n");
        return nullptr;
    }
    auto inputVar = _Input(info->dim, info->order, info->type);
    auto destPtr = inputVar->writeMap<void>();
    if (nullptr == destPtr) {
        MNN_ERROR("Alloc Buffer Error\n");
        return nullptr;
    }
    ::memcpy(destPtr, sourcePtr, info->size * info->type.bytes());
    return inputVar;
}
VARP _Conv2DBackPropFilter(VARP weight, VARP input, VARP inputGrad, PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads){
    std::unique_ptr<OpT> convOp(new OpT);
    convOp->type    = OpType_Conv2DBackPropFilter;
    auto shape      = weight->getInfo();
    auto channel    = std::vector<int>{shape->dim[1], shape->dim[0]};
    auto kernelSize = std::vector<int>{shape->dim[3], shape->dim[2]};
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

VARP _PoolGrad(VARP originInput, VARP originOutput, VARP inputGrad, INTS kernel, INTS stride, PoolingMode type, PaddingMode pad, INTS pads){
    std::unique_ptr<OpT> pool(new OpT);
    pool->type       = OpType_PoolGrad;
    pool->main.type  = OpParameter_Pool;
    pool->main.value = new PoolT;
    if (kernel[0] == -1 && kernel[1] == -1) {
        pool->main.AsPool()->isGlobal = true;
    }
    pool->main.AsPool()->padX    = 0;
    pool->main.AsPool()->padY    = 0;
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
VARP _Pad(VARP x, VARP pads) {
    std::unique_ptr<OpT> pad(new OpT);
    pad->type      = OpType_Padding;
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

VARP _Pack(VARPS xs, halide_type_t dtype, int axis) {
    std::unique_ptr<OpT> pack(new OpT);
    pack->type                         = OpType_Pack;
    pack->main.type                    = OpParameter_PackParam;
    pack->main.value                   = new PackParamT;
    pack->main.AsPackParam()->dataType = (MNN::DataType)Utils::convertDataType(dtype);
    pack->main.AsPackParam()->axis     = axis;
    return (Variable::create(Expr::create(std::move(pack), xs)));
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
} // namespace Express
} // namespace MNN
