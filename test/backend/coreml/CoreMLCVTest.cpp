#include "MNNTestSuite.h"
#if defined(MNN_COREML_ENABLED) && MNN_COREML_ENABLED
#include "MNN_generated.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "core/Backend.hpp"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <atomic>
#include <thread>
#include <map>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
namespace {
using namespace MNN;
static std::vector<float> data(int n, float scale = .2f) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = std::sin(i * .73f) * scale;
    }
    return v;
}
struct CVCase {
    std::string name;
    std::vector<uint8_t> bytes;
    int expectedFallbacks = 0;
    bool checkDispatchOnly = false;
};
static void save(std::vector<CVCase>& cases, const std::string& name, std::unique_ptr<OpT> op,
                 std::vector<std::vector<int>> shapes = {{1, 4, 5, 6}},
                 MNN_DATA_FORMAT format = MNN_DATA_FORMAT_NC4HW4) {
    NetT net;
    net.sourceType = NetSource_CAFFE;
    net.bizCode = "coreml-cv-regression";
    for (int i = 0; i < shapes.size(); ++i) {
        std::unique_ptr<OpT> input(new OpT);
        input->type = OpType_Input;
        input->name = "input" + std::to_string(i);
        input->outputIndexes = {i};
        input->main.type = OpParameter_Input;
        input->main.value = new InputT;
        auto p = input->main.AsInput();
        p->dims = shapes[i];
        p->dtype = (name == "Binary_DIV_int" || name == "Binary_LOGICALOR" || name.find("Reduction_ANY_") == 0 ||
                    name.find("Reduction_ALL_") == 0 ||
                    ((name.find("GatherND_") == 0 || name.find("GatherElements_") == 0) && i == 1))
                       ? DataType_DT_INT32 : DataType_DT_FLOAT;
        p->dformat = format;
        net.tensorName.push_back(input->name);
        net.oplists.push_back(std::move(input));
        op->inputIndexes.push_back(i);
    }
    op->outputIndexes = {int(shapes.size())};
    op->name = name;
    net.tensorName.push_back("output");
    net.outputName = {"output"};
    net.oplists.push_back(std::move(op));
    net.tensorNumber = net.tensorName.size();
    flatbuffers::FlatBufferBuilder fb;
    fb.Finish(Net::Pack(fb, &net));
    CVCase item;
    item.name = name;
    item.bytes.assign(fb.GetBufferPointer(), fb.GetBufferPointer() + fb.GetSize());
    cases.push_back(std::move(item));
}
static std::unique_ptr<OpT> make(OpType type, OpParameter param) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = type;
    op->main.type = param;
    return op;
}
static std::vector<CVCase> makeCases() {
    std::vector<CVCase> dir;
    for (auto type :
         {OpType_Convolution, OpType_ConvolutionDepthwise, OpType_Deconvolution, OpType_DeconvolutionDepthwise}) {
        for (int kernel : {1, 3}) {
            auto op = make(type, OpParameter_Convolution2D);
            op->main.value = new Convolution2DT;
            auto p = op->main.AsConvolution2D();
            p->common.reset(new Convolution2DCommonT);
            auto c = p->common.get();
            c->inputCount = c->outputCount = 4;
            c->kernelX = c->kernelY = kernel;
            c->group = (type == OpType_ConvolutionDepthwise || type == OpType_DeconvolutionDepthwise) ? 4 : 1;
            c->padMode = PadMode_VALID;
            c->strideX = c->strideY = 1;
            c->dilateX = c->dilateY = 1;
            p->weight = data(4 * 4 / c->group * kernel * kernel);
            p->bias = data(4, .1f);
            save(dir, std::string(EnumNameOpType(type)) + "_k" + std::to_string(kernel), std::move(op));
        }
    }
    // Rank-five Raster bindings used to make the ANE compiler repeatedly
    // decompose an identity reshape, exhausting memory even for 32 elements.
    for (int kernel : {1, 2}) {
        for (int pad : {0, 1}) {
            auto op = make(OpType_Convolution3D, OpParameter_Convolution3D);
            op->main.value = new Convolution3DT;
            auto p = op->main.AsConvolution3D();
            p->common.reset(new Convolution3DCommonT);
            auto c = p->common.get();
            c->inputCount = 2;
            c->outputCount = 3;
            c->group = 1;
            c->kernels = {kernel, kernel, kernel};
            c->strides = c->dilates = {1, 1, 1};
            c->padMode = PadMode_CAFFE;
            c->pads = {pad, pad, pad};
            p->weight = data(3 * 2 * kernel * kernel * kernel, .1f);
            p->bias = data(3, .1f);
            save(dir, "Conv3D_k" + std::to_string(kernel) + "_p" + std::to_string(pad), std::move(op),
                 {{2, 2, 2, 2, 2}});
        }
    }
    for (auto type : {OpType_ReLU, OpType_ReLU6, OpType_ELU, OpType_PReLU, OpType_Sigmoid, OpType_Softmax}) {
        std::unique_ptr<OpT> op(new OpT);
        op->type = type;
        if (type == OpType_ReLU) {
            op->main.type = OpParameter_Relu;
            op->main.value = new ReluT;
            op->main.AsRelu()->slope = .1f;
        }
        if (type == OpType_ReLU6) {
            op->main.type = OpParameter_Relu6;
            op->main.value = new Relu6T;
            op->main.AsRelu6()->minValue = 0;
            op->main.AsRelu6()->maxValue = 1;
        }
        if (type == OpType_ELU) {
            op->main.type = OpParameter_ELU;
            op->main.value = new ELUT;
            op->main.AsELU()->alpha = .5f;
        }
        if (type == OpType_PReLU) {
            op->main.type = OpParameter_PRelu;
            op->main.value = new PReluT;
            op->main.AsPRelu()->slopeCount = 4;
            op->main.AsPRelu()->slope = {.1f, .2f, .3f, .4f};
        }
        if (type == OpType_Softmax) {
            op->main.type = OpParameter_Axis;
            op->main.value = new AxisT;
            op->main.AsAxis()->axis = 1;
        }
        save(dir, EnumNameOpType(type), std::move(op));
    }
    for (auto t : {EltwiseType_SUM, EltwiseType_SUB, EltwiseType_PROD, EltwiseType_MAXIMUM}) {
        auto op = make(OpType_Eltwise, OpParameter_Eltwise);
        op->main.value = new EltwiseT;
        op->main.AsEltwise()->type = t;
        save(dir, std::string("Eltwise_") + EnumNameEltwiseType(t), std::move(op), {{1, 4, 5, 6}, {1, 4, 5, 6}});
    }
    for (auto type : {OpType_MatMul, OpType_BatchMatMul}) {
        for (int trans = 0; trans < 2; ++trans) {
            std::unique_ptr<OpT> op(new OpT);
            op->type = type;
            if (type == OpType_MatMul) {
                op->main.type = OpParameter_MatMul;
                op->main.value = new MatMulT;
                op->main.AsMatMul()->transposeB = trans;
            } else {
                op->main.type = OpParameter_BatchMatMulParam;
                op->main.value = new BatchMatMulParamT;
                op->main.AsBatchMatMulParam()->adjY = trans;
            }
            std::vector<int> a = {3, 4}, b = trans ? std::vector<int>{5, 4} : std::vector<int>{4, 5};
            if (type == OpType_BatchMatMul) {
                a.insert(a.begin(), 2);
                b.insert(b.begin(), 2);
            }
            save(dir, std::string(EnumNameOpType(type)) + "_tb" + std::to_string(trans), std::move(op), {a, b},
                 MNN_DATA_FORMAT_NCHW);
        }
    }
    for (auto type : {PoolType_MAXPOOL, PoolType_AVEPOOL}) {
        for (int mode = 0; mode < 3; ++mode) {
            auto op = make(OpType_Pooling, OpParameter_Pool);
            op->main.value = new PoolT;
            auto p = op->main.AsPool();
            p->type = type;
            p->kernelX = p->kernelY = 3;
            p->strideX = p->strideY = 1;
            p->padType = mode == 0 ? PoolPadType_VALID : PoolPadType_CAFFE;
            p->ceilModel = false;
            p->isGlobal = mode == 2;
            p->padX = p->padY = mode == 1 ? 1 : 0;
            save(dir, std::string("Pooling_") + EnumNamePoolType(type) + "_" + std::to_string(mode), std::move(op));
        }
    }
    {
        auto op = make(OpType_Scale, OpParameter_Scale);
        op->main.value = new ScaleT;
        auto p = op->main.AsScale();
        p->channels = 4;
        p->scaleData = {.5f, .7f, .9f, 1.1f};
        p->biasData = {.1f, .2f, .3f, .4f};
        save(dir, "Scale", std::move(op));
    }
    for (auto t :
         {UnaryOpOperation_ABS,    UnaryOpOperation_ACOS,  UnaryOpOperation_ACOSH,         UnaryOpOperation_ASIN,
          UnaryOpOperation_ASINH,  UnaryOpOperation_ATAN,  UnaryOpOperation_ATANH,         UnaryOpOperation_CEIL,
          UnaryOpOperation_COS,    UnaryOpOperation_COSH,  UnaryOpOperation_ERF,           UnaryOpOperation_EXP,
          UnaryOpOperation_FLOOR,  UnaryOpOperation_GELU,  UnaryOpOperation_GELU_STANDARD, UnaryOpOperation_HARDSWISH,
          UnaryOpOperation_LOG,    UnaryOpOperation_LOG1P, UnaryOpOperation_NEG,           UnaryOpOperation_RECIPROCAL,
          UnaryOpOperation_ROUND,  UnaryOpOperation_RSQRT, UnaryOpOperation_SIGMOID,       UnaryOpOperation_SIGN,
          UnaryOpOperation_SILU,   UnaryOpOperation_SIN,   UnaryOpOperation_SINH,          UnaryOpOperation_SQRT,
          UnaryOpOperation_SQUARE, UnaryOpOperation_TAN,   UnaryOpOperation_TANH,          UnaryOpOperation_BNLL,
          UnaryOpOperation_EXPM1,  UnaryOpOperation_ERFC}) {
        auto op = make(OpType_UnaryOp, OpParameter_UnaryOp);
        op->main.value = new UnaryOpT;
        op->main.AsUnaryOp()->opType = t;
        op->main.AsUnaryOp()->T = DataType_DT_FLOAT;
        save(dir, std::string("UnaryOp_") + EnumNameUnaryOpOperation(t), std::move(op));
    }
    for (auto type : {OpType_ArgMax, OpType_ArgMin}) {
        for (int axis : {0, 1}) {
            auto op = make(type, OpParameter_ArgMax);
            op->main.value = new ArgMaxT;
            op->main.AsArgMax()->axis = axis;
            op->main.AsArgMax()->topK = 1;
            save(dir, std::string(EnumNameOpType(type)) + std::to_string(axis), std::move(op), {{3, 5}},
                 MNN_DATA_FORMAT_NCHW);
        }
    }
    for (auto type :
         {ReductionType_SUM, ReductionType_MEAN, ReductionType_MAXIMUM, ReductionType_MINIMUM, ReductionType_PROD}) {
        auto op = make(OpType_Reduction, OpParameter_ReductionParam);
        op->main.value = new ReductionParamT;
        op->main.AsReductionParam()->operation = type;
        op->main.AsReductionParam()->dim = {1};
        op->main.AsReductionParam()->keepDims = true;
        save(dir, std::string("Reduction_") + EnumNameReductionType(type), std::move(op), {{2, 3, 4}},
             MNN_DATA_FORMAT_NCHW);
    }
    for (auto type : {ReductionType_ANY, ReductionType_ALL}) {
        for (int axis : {1, -1}) {
            for (int keep : {0, 1}) {
                auto op = make(OpType_Reduction, OpParameter_ReductionParam);
                op->main.value = new ReductionParamT;
                op->main.AsReductionParam()->operation = type;
                op->main.AsReductionParam()->dim = {axis};
                op->main.AsReductionParam()->keepDims = keep;
                save(dir, std::string("Reduction_") + EnumNameReductionType(type) + "_" +
                              std::to_string(axis) + "_" + std::to_string(keep),
                     std::move(op), {{2, 3, 4}}, MNN_DATA_FORMAT_NCHW);
            }
        }
    }
    {
        auto op = make(OpType_LayerNorm, OpParameter_LayerNorm);
        op->main.value = new LayerNormT;
        auto p = op->main.AsLayerNorm();
        p->axis = {-1};
        p->epsilon = 1e-5f;
        p->gamma = {.5f, .75f, 1.f, 1.25f};
        p->beta = {.1f, .2f, .3f, .4f};
        save(dir, "LayerNorm", std::move(op), {{2, 3, 4}}, MNN_DATA_FORMAT_NCHW);
    }
    for (int type : {1, 2, 3, 4}) {
        auto op = make(OpType_Interp, OpParameter_Interp);
        op->main.value = new InterpT;
        auto p = op->main.AsInterp();
        p->resizeType = type;
        p->outputWidth = 8;
        p->outputHeight = 7;
        p->ctm = CoordinateTransformationMode_HalfPixels;
        save(dir, "Interp_" + std::to_string(type), std::move(op));
    }
    {
        auto op = make(OpType_Permute, OpParameter_Permute);
        op->main.value = new PermuteT;
        op->main.AsPermute()->dims = {0, 2, 3, 1};
        save(dir, "Raster_transpose", std::move(op), {{2, 3, 4, 5}}, MNN_DATA_FORMAT_NCHW);
    }
    for (auto type :
         {BinaryOpOperation_ADD, BinaryOpOperation_SUB, BinaryOpOperation_MUL, BinaryOpOperation_DIV,
          BinaryOpOperation_REALDIV, BinaryOpOperation_POW, BinaryOpOperation_MINIMUM, BinaryOpOperation_MAXIMUM,
          BinaryOpOperation_GREATER, BinaryOpOperation_GREATER_EQUAL, BinaryOpOperation_LESS,
          BinaryOpOperation_LESS_EQUAL, BinaryOpOperation_EQUAL, BinaryOpOperation_NOTEQUAL, BinaryOpOperation_FLOORDIV,
          BinaryOpOperation_MOD, BinaryOpOperation_ATAN2, BinaryOpOperation_LOGICALOR}) {
        auto op = make(OpType_BinaryOp, OpParameter_BinaryOp);
        op->main.value = new BinaryOpT;
        op->main.AsBinaryOp()->opType = type;
        op->main.AsBinaryOp()->T = type == BinaryOpOperation_LOGICALOR ? DataType_DT_INT32 : DataType_DT_FLOAT;
        save(dir, std::string("Binary_") + EnumNameBinaryOpOperation(type), std::move(op), {{2, 3, 4}, {4}},
             MNN_DATA_FORMAT_NCHW);
    }
    {
        auto op = make(OpType_BinaryOp, OpParameter_BinaryOp);
        op->main.value = new BinaryOpT;
        op->main.AsBinaryOp()->opType = BinaryOpOperation_DIV;
        op->main.AsBinaryOp()->T = DataType_DT_INT32;
        save(dir, "Binary_DIV_int", std::move(op), {{2, 3, 4}, {4}}, MNN_DATA_FORMAT_NCHW);
    }
    {
        auto op = make(OpType_Convolution, OpParameter_Convolution2D);
        op->main.value = new Convolution2DT;
        auto p = op->main.AsConvolution2D();
        p->common.reset(new Convolution2DCommonT);
        auto c = p->common.get();
        c->inputCount = 256;
        c->outputCount = 16;
        c->kernelX = c->kernelY = 3;
        c->strideX = c->strideY = c->dilateX = c->dilateY = c->group = 1;
        c->padMode = PadMode_SAME;
        p->weight.resize(16 * 256 * 9);
        for (int i = 0; i < p->weight.size(); ++i) {
            p->weight[i] = float(i % 255) / 255000.f;
        }
        p->bias.assign(16, .75f);
        save(dir, "Convolution_long_accumulation", std::move(op), {{1, 256, 8, 8}});
    }
    for (auto type : {PoolType_MAXPOOL, PoolType_AVEPOOL}) {
        auto op = make(OpType_Pooling, OpParameter_Pool);
        op->main.value = new PoolT;
        auto p = op->main.AsPool();
        p->type = type;
        p->kernelY = 3;
        p->kernelX = 1;
        p->strideY = p->strideX = 1;
        p->padY = 1;
        p->padType = PoolPadType_CAFFE;
        p->ceilModel = false;
        save(dir, std::string("Pooling_1d_") + EnumNamePoolType(type), std::move(op), {{1, 4, 6}});
    }
    for (int channels : {3, 4, 7}) {
        for (int size : {3, 5}) {
            auto lrn = make(OpType_LRN, OpParameter_LRN);
            lrn->main.value = new LRNT;
            lrn->main.AsLRN()->regionType = 0;
            lrn->main.AsLRN()->localSize = size;
            lrn->main.AsLRN()->alpha = 0.5f;
            lrn->main.AsLRN()->beta = 0.75f;
            lrn->main.AsLRN()->bias = 1.0f;
            save(dir, "LRN_" + std::to_string(channels) + "_" + std::to_string(size), std::move(lrn),
                 {{1, channels, 4, 5}}, MNN_DATA_FORMAT_NC4HW4);
        }
    }
    for (auto mode : {SampleMode_BILINEAR}) {
        for (auto padding : {BorderMode_ZEROS, BorderMode_CLAMP}) {
            for (int align : {0, 1}) {
                auto grid = make(OpType_GridSample, OpParameter_GridSample);
                grid->main.value = new GridSampleT;
                grid->main.AsGridSample()->mode = mode;
                grid->main.AsGridSample()->paddingMode = padding;
                grid->main.AsGridSample()->alignCorners = align;
                save(dir, "GridSample_" + std::to_string(int(mode)) + "_" + std::to_string(int(padding)) + "_" +
                              std::to_string(align), std::move(grid), {{1, 3, 4, 5}, {1, 2, 3, 2}},
                     MNN_DATA_FORMAT_NCHW);
                std::unique_ptr<NetT> net(GetNet(dir.back().bytes.data())->UnPack());
                net->oplists[0]->main.AsInput()->dformat = MNN_DATA_FORMAT_NC4HW4;
                flatbuffers::FlatBufferBuilder fb;
                fb.Finish(Net::Pack(fb, net.get()));
                dir.back().bytes.assign(fb.GetBufferPointer(), fb.GetBufferPointer() + fb.GetSize());
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        auto grid = make(OpType_GridSample, OpParameter_GridSample);
        grid->main.value = new GridSampleT;
        grid->main.AsGridSample()->mode = i == 0 ? SampleMode_NEAREST : SampleMode_BILINEAR;
        grid->main.AsGridSample()->paddingMode = i == 0 ? BorderMode_ZEROS : BorderMode_REFLECTION;
        grid->main.AsGridSample()->alignCorners = true;
        save(dir, "GridSample_expected_fallback_" + std::to_string(i), std::move(grid),
             {{1, 3, 4, 5}, {1, 2, 3, 2}}, MNN_DATA_FORMAT_NCHW);
        std::unique_ptr<NetT> net(GetNet(dir.back().bytes.data())->UnPack());
        net->oplists[0]->main.AsInput()->dformat = MNN_DATA_FORMAT_NC4HW4;
        flatbuffers::FlatBufferBuilder fb;
        fb.Finish(Net::Pack(fb, net.get()));
        dir.back().bytes.assign(fb.GetBufferPointer(), fb.GetBufferPointer() + fb.GetSize());
        dir.back().expectedFallbacks = 1;
    }
    save(dir, "GatherND_native", make(OpType_GatherND, OpParameter_NONE), {{3, 4}, {2, 1}}, MNN_DATA_FORMAT_NCHW);
    save(dir, "GatherElements_native", make(OpType_GatherElements, OpParameter_NONE), {{3, 4}, {2, 4}}, MNN_DATA_FORMAT_NCHW);
    save(dir, "GatherND_depth2", make(OpType_GatherND, OpParameter_NONE), {{3, 4, 5}, {2, 2}}, MNN_DATA_FORMAT_NCHW);

    save(dir, "ReLU6_default", make(OpType_ReLU6, OpParameter_NONE));
    auto appendNet = [&](NetT& net, const std::string& name) {
        flatbuffers::FlatBufferBuilder fb;
        net.tensorNumber = net.tensorName.size();
        fb.Finish(Net::Pack(fb, &net));
        CVCase item;
        item.name = name;
        item.bytes.assign(fb.GetBufferPointer(), fb.GetBufferPointer() + fb.GetSize());
        dir.push_back(std::move(item));
    };
    for (auto bounds : std::vector<std::vector<int>>{{-1, 0}, {0, -1}, {0, 0}, {1, 2}, {-1, -1}}) {
        std::unique_ptr<NetT> net(GetNet(dir[0].bytes.data())->UnPack());
        net->oplists.resize(1);
        auto input = net->oplists[0]->main.AsInput();
        input->dims = {2, 3, 4};
        input->dformat = MNN_DATA_FORMAT_NCHW;
        net->tensorName = {"input0", "output", "lower", "upper"};
        net->outputName = {"output"};
        for (int i = 0; i < 2; ++i) {
            auto constant = make(OpType_Const, OpParameter_Blob);
            constant->name = net->tensorName[2 + i];
            constant->outputIndexes = {2 + i};
            constant->main.value = new BlobT;
            auto blob = constant->main.AsBlob();
            blob->dims = {1};
            blob->dataType = DataType_DT_INT32;
            blob->int32s = {bounds[i]};
            net->oplists.push_back(std::move(constant));
        }
        auto band = make(OpType_MatrixBandPart, OpParameter_NONE);
        band->name = "band";
        band->inputIndexes = {0, 2, 3};
        band->outputIndexes = {1};
        net->oplists.push_back(std::move(band));
        appendNet(*net, "MatrixBandPart_" + std::to_string(bounds[0]) + "_" + std::to_string(bounds[1]));
    }
    for (int axis : {0, 1, 2}) {
        for (int integer = 0; integer < 2; ++integer) {
            std::unique_ptr<NetT> net(GetNet(dir[0].bytes.data())->UnPack());
            net->oplists.resize(1);
            auto input = net->oplists[0]->main.AsInput();
            input->dims = {2, 3, 4};
            input->dformat = MNN_DATA_FORMAT_NCHW;
            input->dtype = integer ? DataType_DT_INT32 : DataType_DT_FLOAT;
            net->tensorName = {"input0", "output", "axis"};
            net->outputName = {"output"};
            auto constant = make(OpType_Const, OpParameter_Blob);
            constant->name = "axis";
            constant->outputIndexes = {2};
            constant->main.value = new BlobT;
            constant->main.AsBlob()->dataType = DataType_DT_INT32;
            constant->main.AsBlob()->dims = {1};
            constant->main.AsBlob()->int32s = {axis};
            net->oplists.push_back(std::move(constant));
            auto reverse = make(OpType_Reverse, OpParameter_NONE);
            reverse->name = "reverse";
            reverse->inputIndexes = {0, 2};
            reverse->outputIndexes = {1};
            net->oplists.push_back(std::move(reverse));
            appendNet(*net, std::string("Reverse_") + (integer ? "int_" : "float_") + std::to_string(axis));
        }
    }
    for (int axis : {0, 1, -1}) {
        for (int integer = 0; integer < 2; ++integer) {
            std::unique_ptr<NetT> net(GetNet(dir[0].bytes.data())->UnPack());
            net->oplists.resize(1);
            auto input = net->oplists[0]->main.AsInput();
            input->dims = {2, 3};
            input->dformat = MNN_DATA_FORMAT_NCHW;
            input->dtype = DataType_DT_INT32;
            net->tensorName = {"input0", "output", "depth", "on", "off"};
            net->outputName = {"output"};
            for (int i = 0; i < 3; ++i) {
                auto constant = make(OpType_Const, OpParameter_Blob);
                constant->name = net->tensorName[2 + i];
                constant->outputIndexes = {2 + i};
                constant->main.value = new BlobT;
                auto blob = constant->main.AsBlob();
                blob->dims = {1};
                blob->dataType = (i == 0 || integer) ? DataType_DT_INT32 : DataType_DT_FLOAT;
                if (i == 0) {
                    blob->int32s = {4};
                } else if (integer) {
                    blob->int32s = {i == 1 ? 3 : -2};
                } else {
                    blob->float32s = {i == 1 ? 2.0f : -0.5f};
                }
                net->oplists.push_back(std::move(constant));
            }
            auto oneHot = make(OpType_OneHot, OpParameter_OneHotParam);
            oneHot->name = "one_hot";
            oneHot->inputIndexes = {0, 2, 3, 4};
            oneHot->outputIndexes = {1};
            oneHot->main.value = new OneHotParamT;
            oneHot->main.AsOneHotParam()->axis = axis;
            oneHot->main.AsOneHotParam()->dType = integer ? DataType_DT_INT32 : DataType_DT_FLOAT;
            net->oplists.push_back(std::move(oneHot));
            appendNet(*net, std::string("OneHot_") + (integer ? "int_" : "float_") + std::to_string(axis));
        }
    }
    for (int axis : {1, -1}) {
        auto source = std::find_if(dir.begin(), dir.end(), [](const CVCase& item) { return item.name == "GatherElements_native"; });
        std::unique_ptr<NetT> net(GetNet(source->bytes.data())->UnPack());
        net->oplists[1]->main.AsInput()->dims = {3, 2};
        net->oplists[2]->inputIndexes.push_back(3);
        auto constant = make(OpType_Const, OpParameter_Blob);
        constant->name = "axis";
        constant->outputIndexes = {3};
        constant->main.value = new BlobT;
        constant->main.AsBlob()->dataType = DataType_DT_INT32;
        constant->main.AsBlob()->dims = {1};
        constant->main.AsBlob()->int32s = {axis};
        net->tensorName.push_back("axis");
        net->oplists.insert(net->oplists.begin() + 2, std::move(constant));
        appendNet(*net, std::string("GatherElements_axis_") + std::to_string(axis));
    }
    for (int axis : {1, -1}) {
        for (int flags = 0; flags < 4; ++flags) {
            for (int integer = 0; integer < 2; ++integer) {
                std::unique_ptr<NetT> net(GetNet(dir[0].bytes.data())->UnPack());
                net->oplists.resize(1);
                auto input = net->oplists[0]->main.AsInput();
                input->dims = {2, 3, 4};
                input->dformat = MNN_DATA_FORMAT_NCHW;
                input->dtype = integer ? DataType_DT_INT32 : DataType_DT_FLOAT;
                net->tensorName = {"input0", "output", "axis"};
                net->outputName = {"output"};
                auto axisOp = make(OpType_Const, OpParameter_Blob);
                axisOp->name = "axis";
                axisOp->outputIndexes = {2};
                axisOp->main.value = new BlobT;
                axisOp->main.AsBlob()->dataType = DataType_DT_INT32;
                axisOp->main.AsBlob()->dims = {1};
                axisOp->main.AsBlob()->int32s = {axis};
                net->oplists.push_back(std::move(axisOp));
                auto sum = make(OpType_CumSum, OpParameter_CumSum);
                sum->name = "cumsum";
                sum->inputIndexes = {0, 2};
                sum->outputIndexes = {1};
                sum->main.value = new CumSumT;
                sum->main.AsCumSum()->exclusive = flags & 1;
                sum->main.AsCumSum()->reverse = flags & 2;
                net->oplists.push_back(std::move(sum));
                appendNet(*net, std::string("CumSum_") + (integer ? "int_" : "float_") +
                               std::to_string(axis) + "_" + std::to_string(flags));
            }
        }
    }
    {
        std::unique_ptr<NetT> net(GetNet(dir[0].bytes.data())->UnPack());
        net->oplists[0]->main.AsInput()->dims = {1, 5, 6, 4};
        net->oplists[0]->main.AsInput()->dformat = MNN_DATA_FORMAT_NHWC;
        appendNet(*net, "Convolution_NHWC");
    }
    {
        auto source = std::find_if(dir.begin(), dir.end(), [](const CVCase& item) { return item.name == "Interp_1"; });
        std::unique_ptr<NetT> net(GetNet(source->bytes.data())->UnPack());
        auto param = net->oplists[1]->main.AsInterp();
        param->outputHeight = 10;
        param->outputWidth = 12;
        param->ctm = CoordinateTransformationMode_Asymmetric;
        net->oplists[1]->inputIndexes.push_back(2);
        auto constant = make(OpType_Const, OpParameter_Blob);
        constant->name = "scales";
        constant->outputIndexes = {2};
        constant->main.value = new BlobT;
        constant->main.AsBlob()->dataType = DataType_DT_FLOAT;
        constant->main.AsBlob()->dims = {4};
        constant->main.AsBlob()->float32s = {1, 1, 2, 2};
        net->oplists.insert(net->oplists.begin() + 1, std::move(constant));
        net->tensorName.push_back("scales");
        appendNet(*net, "Interp_FP16_scales");
    }
    {
        std::unique_ptr<NetT> net(GetNet(dir[0].bytes.data())->UnPack());
        net->oplists.resize(1);
        const std::vector<std::vector<int>> params = {{0, 0, 1, 0}, {1, 4, 5, 6}, {1, 1, 2, 2}};
        for (int i = 0; i < 3; ++i) {
            auto constant = make(OpType_Const, OpParameter_Blob);
            constant->name = "index" + std::to_string(i);
            constant->outputIndexes = {2 + i};
            constant->main.value = new BlobT;
            constant->main.AsBlob()->dataType = DataType_DT_INT32;
            constant->main.AsBlob()->dims = {4};
            constant->main.AsBlob()->int32s = params[i];
            net->tensorName.push_back(constant->name);
            net->oplists.push_back(std::move(constant));
        }
        auto slice = make(OpType_StridedSlice, OpParameter_StridedSliceParam);
        slice->name = "slice";
        slice->inputIndexes = {0, 2, 3, 4};
        slice->outputIndexes = {1};
        slice->main.value = new StridedSliceParamT;
        slice->main.AsStridedSliceParam()->fromType = 0;
        net->oplists.push_back(std::move(slice));
        appendNet(*net, "StridedSlice_TensorFlow");
    }
    auto baseNet = [&]() {
        std::unique_ptr<NetT> net(GetNet(dir[0].bytes.data())->UnPack());
        net->oplists.resize(1);
        net->oplists[0]->main.AsInput()->dims = {2, 3};
        net->oplists[0]->main.AsInput()->dformat = MNN_DATA_FORMAT_NCHW;
        return net;
    };
    auto intConst = [](NetT& net, const std::string& name, const std::vector<int>& values) {
        int index = net.tensorName.size();
        net.tensorName.push_back(name);
        auto op = make(OpType_Const, OpParameter_Blob);
        op->name = name;
        op->outputIndexes = {index};
        op->main.value = new BlobT;
        auto blob = op->main.AsBlob();
        blob->dataType = DataType_DT_INT32;
        blob->dims = {int(values.size())};
        blob->int32s = values;
        net.oplists.push_back(std::move(op));
        return index;
    };
    for (auto dtype : {DataType_DT_INT64, DataType_DT_BOOL}) {
        auto op = make(OpType_Cast, OpParameter_CastParam);
        op->main.value = new CastParamT;
        op->main.AsCastParam()->dstT = dtype;
        save(dir, std::string("Cast_") + EnumNameDataType(dtype), std::move(op), {{2, 3}}, MNN_DATA_FORMAT_NCHW);
    }
    for (bool scalar : {false, true}) {
        auto op = make(OpType_BinaryOp, OpParameter_BinaryOp);
        op->main.value = new BinaryOpT;
        op->main.AsBinaryOp()->opType = BinaryOpOperation_ADD;
        save(dir, scalar ? "Binary_scalar" : "Binary_vector", std::move(op),
             {scalar ? std::vector<int>{} : std::vector<int>{3}, {2, 3}}, MNN_DATA_FORMAT_NCHW);
    }
    for (int alignment : {0, 1}) {
        auto net = baseNet();
        net->oplists[0]->main.AsInput()->dims = {3};
        auto shape = intConst(*net, "shape", alignment ? std::vector<int>{3, 2} : std::vector<int>{2, 3});
        auto op = make(OpType_BroadcastTo, OpParameter_Axis);
        op->main.value = new AxisT;
        op->main.AsAxis()->axis = alignment;
        op->inputIndexes = {0, shape};
        op->outputIndexes = {1};
        net->oplists.push_back(std::move(op));
        appendNet(*net, "Broadcast_rank_" + std::to_string(alignment));
    }
    for (auto mode : {PadValueMode_CONSTANT, PadValueMode_REFLECT, PadValueMode_SYMMETRIC}) {
        auto net = baseNet();
        auto padding = intConst(*net, "padding", {1, 0, 0, 1});
        auto op = make(OpType_Padding, OpParameter_PadParam);
        op->main.value = new PadParamT;
        op->main.AsPadParam()->mode = mode;
        op->inputIndexes = {0, padding};
        if (mode == PadValueMode_CONSTANT) {
            int fill = intConst(*net, "fill", {5});
            auto blob = net->oplists.back()->main.AsBlob();
            blob->dataType = DataType_DT_FLOAT;
            blob->int32s.clear();
            blob->float32s = {5.f};
            op->inputIndexes.push_back(fill);
        }
        op->outputIndexes = {1};
        net->oplists.push_back(std::move(op));
        appendNet(*net, std::string("Padding_") + EnumNamePadValueMode(mode));
    }
    for (bool withAxis : {false, true}) {
        auto net = baseNet();
        auto k = intConst(*net, "k", {2});
        auto op = make(OpType_TopKV2, OpParameter_TopKV2);
        op->main.value = new TopKV2T;
        op->main.AsTopKV2()->largest = true;
        op->inputIndexes = {0, k};
        if (withAxis) {
            op->inputIndexes.push_back(intConst(*net, "axis", {-1}));
        }
        int indices = net->tensorName.size();
        net->tensorName.push_back("indices");
        net->outputName.push_back("indices");
        op->outputIndexes = {1, indices};
        net->oplists.push_back(std::move(op));
        appendNet(*net, withAxis ? "TopK_three_inputs" : "TopK_two_inputs");
    }
    {
        auto net = baseNet();
        int axis = intConst(*net, "axis", {0});
        auto op = make(OpType_Reduction, OpParameter_ReductionParam);
        op->main.value = new ReductionParamT;
        op->main.AsReductionParam()->operation = ReductionType_SUM;
        op->main.AsReductionParam()->dim = {1};
        op->main.AsReductionParam()->keepDims = true;
        op->inputIndexes = {0, axis};
        op->outputIndexes = {1};
        net->oplists.push_back(std::move(op));
        appendNet(*net, "Reduction_dim_precedence");
    }
    {
        auto net = baseNet();
        flatbuffers::FlatBufferBuilder fb;
        auto input = Op::Pack(fb, net->oplists[0].get());
        auto axes = fb.CreateVector(std::vector<int>{});
        auto param = CreateReductionParam(fb, ReductionType_SUM, axes, 0.f, true);
        auto op = CreateOp(fb, OpType_Reduction, fb.CreateString("empty_axes"),
                           fb.CreateVector(std::vector<int>{0}), fb.CreateVector(std::vector<int>{1}),
                           OpParameter_ReductionParam, param.Union());
        auto ops = fb.CreateVector(std::vector<flatbuffers::Offset<Op>>{input, op});
        auto names = fb.CreateVectorOfStrings(std::vector<std::string>{"input0", "output"});
        auto outputs = fb.CreateVectorOfStrings(std::vector<std::string>{"output"});
        fb.Finish(CreateNet(fb, ops, names, 0, NetSource_CAFFE, 0, outputs, 2));
        CVCase item;
        item.name = "Reduction_empty_axes";
        item.bytes.assign(fb.GetBufferPointer(), fb.GetBufferPointer() + fb.GetSize());
        dir.push_back(std::move(item));
    }
    for (bool grouped : {false, true}) {
        auto op = make(OpType_LayerNorm, OpParameter_LayerNorm);
        op->main.value = new LayerNormT;
        auto param = op->main.AsLayerNorm();
        param->useRMSNorm = true;
        param->group = grouped ? 2 : 1;
        param->axis = grouped ? std::vector<int>{1} : std::vector<int>{2, 3};
        param->gamma = data(6, .5f);
        param->beta = data(6, .1f);
        save(dir, grouped ? "RMS_group" : "RMS_multi_axis", std::move(op),
             grouped ? std::vector<std::vector<int>>{{2, 12}} : std::vector<std::vector<int>>{{2, 4, 2, 3}},
             MNN_DATA_FORMAT_NCHW);
    }
    for (bool scale : {false, true}) {
        auto op = make(scale ? OpType_Scale : OpType_PReLU, scale ? OpParameter_Scale : OpParameter_PRelu);
        if (scale) {
            op->main.value = new ScaleT;
            op->main.AsScale()->channels = 4;
            op->main.AsScale()->scaleData = {.1f, .2f, .3f, .4f};
            op->main.AsScale()->biasData = {.4f, .3f, .2f, .1f};
        } else {
            op->main.value = new PReluT;
            op->main.AsPRelu()->slopeCount = 4;
            op->main.AsPRelu()->slope = {.1f, .2f, .3f, .4f};
        }
        save(dir, scale ? "Scale_rank1" : "PReLU_rank1", std::move(op), {{4}}, MNN_DATA_FORMAT_NCHW);
    }
    for (auto kind : {UnaryOpOperation_EXPM1, UnaryOpOperation_LOG1P, UnaryOpOperation_ERFC}) {
        auto op = make(OpType_UnaryOp, OpParameter_UnaryOp);
        op->main.value = new UnaryOpT;
        op->main.AsUnaryOp()->opType = kind;
        save(dir, std::string("Unary_edge_") + EnumNameUnaryOpOperation(kind), std::move(op), {{1, 8}},
             MNN_DATA_FORMAT_NCHW);
    }
    for (int variant = 0; variant < 2; ++variant) {
        std::unique_ptr<NetT> net(GetNet(dir[1].bytes.data())->UnPack());
        auto p = net->oplists.back()->main.AsConvolution2D();
        if (variant == 0) {
            p->common->group = 2;
            p->common->inputCount = 2;
            p->weight = data(4 * 2 * 3 * 3);
        } else {
            p->common->padMode = PadMode_CAFFE;
            p->common->pads = {1, 0, 0, 2};
        }
        appendNet(*net, variant == 0 ? "Conv_group_inputCount" : "Conv_asymmetric");
    }
    for (int variant = 0; variant < 3; ++variant) {
        auto source = std::find_if(dir.begin(), dir.end(), [](const CVCase& c) { return c.name == "Conv3D_k2_p0"; });
        std::unique_ptr<NetT> net(GetNet(source->bytes.data())->UnPack());
        auto p = net->oplists.back()->main.AsConvolution3D();
        if (variant == 0) {
            p->common->outputCount = 4;
            p->common->group = 2;
            p->weight = data(4 * 1 * 8, .1f);
            p->bias = data(4, .1f);
        } else if (variant == 1) {
            p->common->pads = {1, 0, 0, 0, 1, 0};
        } else {
            p->common->relu6 = true;
            p->bias = {-1.f, .1f, 8.f};
        }
        appendNet(*net, "Conv3D_edge_" + std::to_string(variant));
    }
    for (int axis : {0, 10000, 1}) {
        auto op = make(OpType_ArgMax, OpParameter_ArgMax);
        op->main.value = new ArgMaxT;
        op->main.AsArgMax()->axis = axis;
        op->main.AsArgMax()->topK = 1;
        save(dir, "ArgMax_fallback_" + std::to_string(axis), std::move(op), {{1, 2050, 1, 1}});
        dir.back().expectedFallbacks = 1;
        dir.back().checkDispatchOnly = axis == 10000;
    }
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
    for (int mask = 0; mask < 4; ++mask) {
        auto op = make(OpType_Attention, OpParameter_AttentionParam);
        op->main.value = new AttentionParamT;
        op->main.AsAttentionParam()->kv_cache = false;
        std::vector<std::vector<int>> shapes = {{1, 3, 2, 4}, {1, 3, 2, 4}, {1, 3, 2, 4}};
        if (mask > 0) {
            shapes.push_back(mask == 1   ? std::vector<int>{}
                             : mask == 2 ? std::vector<int>{1}
                                         : std::vector<int>{1, 1, 3, 3});
        }
        save(dir, "Attention_mask_" + std::to_string(mask), std::move(op), shapes, MNN_DATA_FORMAT_NCHW);
    }
#endif
    return dir;
}

static bool runCase(const CVCase& item) {
    MNN_PRINT("CoreML CV START %s\n", item.name.c_str());
    fflush(stdout);
    // DIV is a legacy alias absent from the CPU dispatcher; REALDIV supplies the same reference semantics.
    std::unique_ptr<NetT> referenceNet(GetNet(item.bytes.data())->UnPack());
    for (auto& op : referenceNet->oplists) {
        if (op->type == OpType_BinaryOp && op->main.AsBinaryOp()->opType == BinaryOpOperation_DIV) {
            op->main.AsBinaryOp()->opType = BinaryOpOperation_REALDIV;
        }
    }
    if (item.name == "Reduction_dim_precedence") {
        // The CPU execution reads the second input first; use the canonical dim-only graph as oracle.
        referenceNet->oplists.back()->inputIndexes.resize(1);
    }
    if (item.name == "Conv3D_edge_0") {
        // CPU GeometryConv3D assumes group=1; expand grouped weights into a block-diagonal reference.
        auto param = referenceNet->oplists.back()->main.AsConvolution3D();
        auto grouped = param->weight;
        param->weight.assign(4 * 2 * 8, 0.f);
        for (int oc = 0; oc < 4; ++oc) {
            std::copy_n(grouped.data() + oc * 8, 8, param->weight.data() + (oc * 2 + oc / 2) * 8);
        }
        param->common->group = 1;
    }
    if (item.name == "Reduction_empty_axes") {
        // Native object packing erases the distinction between an absent and an empty dim vector.
        auto& op = referenceNet->oplists.back();
        op->type = OpType_Reshape;
        op->main.Reset();
        op->main.type = OpParameter_Reshape;
        op->main.value = new ReshapeT;
        op->main.AsReshape()->dims = {2, 3};
    }
    flatbuffers::FlatBufferBuilder referenceBuffer;
    referenceBuffer.Finish(Net::Pack(referenceBuffer, referenceNet.get()));
    std::shared_ptr<Interpreter> cpu(
        Interpreter::createFromBuffer(referenceBuffer.GetBufferPointer(), referenceBuffer.GetSize()),
        Interpreter::destroy);
    std::shared_ptr<Interpreter> mil(Interpreter::createFromBuffer(item.bytes.data(), item.bytes.size()),
                                     Interpreter::destroy);
    if (!cpu || !mil) {
        return false;
    }
    BackendConfig cpuConfig, milConfig;
    cpuConfig.precision = BackendConfig::Precision_High;
    milConfig.precision = BackendConfig::Precision_Low;
    ScheduleConfig config;
    config.numThread = 1;
    config.type = MNN_FORWARD_CPU;
    config.backendConfig = &cpuConfig;
    auto cpuSession = cpu->createSession(config);
    config.type = MNN_FORWARD_NN;
    config.backendConfig = &milConfig;
    auto milSession = mil->createSession(config);
    if (!cpuSession || !milSession) {
        return false;
    }
    int cpuState = -1, milState = -1;
    cpu->getSessionInfo(cpuSession, Interpreter::RESIZE_STATUS, &cpuState);
    mil->getSessionInfo(milSession, Interpreter::RESIZE_STATUS, &milState);
    if (cpuState || milState) {
        return false;
    }
    if (item.checkDispatchOnly) {
        auto backend = mil->getBackend(milSession, mil->getSessionOutput(milSession, nullptr));
        MNN_PRINT("CoreML dispatch-only %s (no numerical claim)\n", item.name.c_str());
        return backend && backend->type() != MNN_FORWARD_NN;
    }
    for (int run = 0; run < 2; ++run) {
        if (run == 1 && item.name == "Graph_resize") {
            for (auto interpreter : {cpu.get(), mil.get()}) {
                auto session = interpreter == cpu.get() ? cpuSession : milSession;
                interpreter->resizeTensor(interpreter->getSessionInput(session, nullptr), {2, 4, 7, 6});
                interpreter->resizeSession(session);
                int state = -1;
                interpreter->getSessionInfo(session, Interpreter::RESIZE_STATUS, &state);
                if (state) {
                    return false;
                }
            }
        }
        std::map<std::string, std::shared_ptr<Tensor>> inputCopies;
        for (auto& entry : cpu->getSessionInputAll(cpuSession)) {
            auto target = mil->getSessionInput(milSession, entry.first.c_str());
            Tensor host(entry.second, Tensor::CAFFE);
            for (int i = 0; i < host.elementSize(); ++i) {
                if (host.getType() == halide_type_of<int>()) {
                    host.host<int>()[i] = ((i % 11) + 1) * (run + 1) * (i % 3 == 0 ? -1 : 1);
                } else {
                    host.host<float>()[i] =
                        float((i % 11) + 1) / 16.0f + run / 32.0f + (item.name == "UnaryOp_ACOSH" ? 1.f : 0.f);
                }
            }
            if ((item.name.find("GatherND_") == 0 || item.name.find("GatherElements_") == 0) &&
                entry.first == "input1") {
                for (int i = 0; i < host.elementSize(); ++i) {
                    host.host<int>()[i] = (i + run) % 3;
                }
            }
            if (item.name.find("OneHot_") == 0) {
                for (int i = 0; i < host.elementSize(); ++i) {
                    host.host<int>()[i] = (i + run) % 6 - 1;
                }
            }
            if (item.name.find("Reduction_ANY_") == 0 || item.name.find("Reduction_ALL_") == 0) {
                for (int i = 0; i < host.elementSize(); ++i) {
                    host.host<int>()[i] = (i + run) % 5 - 2;
                }
            }
            if (item.name.find("GridSample_") == 0 && entry.first == "input1") {
                const float points[] = {-1.25f, -1.0f, -0.75f, -0.5f, -0.125f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.25f};
                for (int i = 0; i < host.elementSize(); ++i) {
                    host.host<float>()[i] = points[(i + run * 3) % 11];
                }
            }
            if (item.name == "UnaryOp_ROUND") {
                const float edges[] = {-2.5f, -1.5f, -.5f, -.499755859375f, 0.f, .499755859375f, .5f, 1.5f, 2.5f};
                for (int i = 0; i < host.elementSize(); ++i) {
                    host.host<float>()[i] = edges[i % 9] + run * .25f;
                }
            }
            if (item.name == "ELU" || item.name == "PReLU" || item.name == "ReLU" || item.name == "ReLU6" ||
                item.name == "ReLU6_default" || item.name == "PReLU_rank1" || item.name.find("Cast_DT_") == 0) {
                for (int i = 0; i < host.elementSize(); ++i) {
                    host.host<float>()[i] = (i % 17 - 8) * .5f + run * .25f;
                }
            }
            if (item.name.find("Unary_edge_") == 0) {
                for (int i = 0; i < host.elementSize(); ++i) {
                    host.host<float>()[i] = item.name == "Unary_edge_ERFC" ? 2.5f + i * .125f : (i - 4) * .0001f;
                }
            }
            inputCopies[entry.first].reset(new Tensor(&host, Tensor::CAFFE));
            std::memcpy(inputCopies[entry.first]->host<void>(), host.host<void>(), host.size());
            if (!entry.second->copyFromHostTensor(&host) || !target->copyFromHostTensor(&host)) {
                return false;
            }
        }
        if (cpu->runSession(cpuSession) != NO_ERROR) {
            return false;
        }
        // The allocator may reuse input storage; each independent inference receives fresh input.
        for (bool callback : {false, true}) {
            for (const auto& input : inputCopies) {
                if (!mil->getSessionInput(milSession, input.first.c_str())->copyFromHostTensor(input.second.get())) {
                    return false;
                }
            }
            int milOps = 0, fallbackOps = 0;
            auto code =
                callback ? mil->runSessionWithCallBackInfo(
                               milSession, [](const std::vector<Tensor*>&, const OperatorInfo*) { return true; },
                               [&](const std::vector<Tensor*>& outputs, const OperatorInfo*) {
                                   auto backend = outputs.empty() ? nullptr : mil->getBackend(milSession, outputs[0]);
                                   if (backend && backend->type() == MNN_FORWARD_NN) {
                                       ++milOps;
                                   } else {
                                       ++fallbackOps;
                                   }
                                   return true;
                               })
                         : mil->runSession(milSession);
            if (code != NO_ERROR ||
                (callback && ((!item.expectedFallbacks && milOps == 0) || fallbackOps != item.expectedFallbacks))) {
                MNN_ERROR("CoreML CV %s: code=%d MIL=%d fallback=%d\n", item.name.c_str(), code, milOps, fallbackOps);
                return false;
            }
            for (auto& entry : cpu->getSessionOutputAll(cpuSession)) {
                Tensor reference(entry.second, Tensor::CAFFE);
                auto output = mil->getSessionOutput(milSession, entry.first.c_str());
                if (!output) {
                    return false;
                }
                Tensor actual(output, Tensor::CAFFE);
                if (reference.elementSize() != actual.elementSize() || reference.getType() != actual.getType()) {
                    return false;
                }
                if (!entry.second->copyToHostTensor(&reference) || !output->copyToHostTensor(&actual)) {
                    return false;
                }
                double difference = 0, energy = 0, maximum = 0, error = 0;
                for (int i = 0; i < reference.elementSize(); ++i) {
                    bool integer = reference.getType() == halide_type_of<int>();
                    double a = integer ? reference.host<int>()[i] : reference.host<float>()[i];
                    double b = integer ? actual.host<int>()[i] : actual.host<float>()[i];
                    if (!std::isfinite(a) || !std::isfinite(b) || (integer && a != b)) {
                        return false;
                    }
                    double d = a - b;
                    difference += d * d;
                    energy += a * a;
                    maximum = std::max(maximum, std::abs(a));
                    error = std::max(error, std::abs(d));
                }
                double relative = std::sqrt(difference / std::max(energy, 1e-20));
                if (relative > .01 || error / std::max(maximum, 1e-10) > .02) {
                    MNN_ERROR("CoreML CV %s run=%d callback=%d relative_l2=%g max_error=%g\n", item.name.c_str(), run,
                              callback, relative, error);
                    return false;
                }
            }
        }
    }

    if (item.expectedFallbacks) {
        MNN_PRINT("CoreML partition PASS %s (expected CPU operators=%d, not native coverage)\n", item.name.c_str(),
                  item.expectedFallbacks);
    } else {
        MNN_PRINT("CoreML CV PASS %s (MIL, zero CPU fallback, two inputs)\n", item.name.c_str());
    }
    return true;
}
class CoreMLCVTest : public MNNTestCase {
public:
    bool run(int) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_NN) {
            ++MNNTestSuite::get()->skipped;
            return true;
        }
        auto cases = makeCases();
        const char* filter = std::getenv("MNN_COREML_CASE");
        int matched = 0;
        for (const auto& item : cases) {
            if (filter && item.name != filter) {
                continue;
            }
            ++matched;
            if (const char* exportDir = std::getenv("MNN_COREML_EXPORT_CASE_DIR")) {
                std::ofstream out(std::string(exportDir) + "/" + item.name + ".mnn", std::ios::binary);
                out.write(reinterpret_cast<const char*>(item.bytes.data()), item.bytes.size());
                if (!out) {
                    return false;
                }
            }
            if (!runCase(item)) {
                return false;
            }
        }
        return matched > 0;
    }
};
MNNTestSuiteRegister(CoreMLCVTest, "backend/coreml/cv");
static bool checkCallbacks() {
    std::vector<CVCase> cases;
    auto neg = make(OpType_UnaryOp, OpParameter_UnaryOp);
    neg->main.value = new UnaryOpT;
    neg->main.AsUnaryOp()->opType = UnaryOpOperation_NEG;
    save(cases, "negate", std::move(neg), {{4}}, MNN_DATA_FORMAT_NCHW);
    std::unique_ptr<NetT> net(GetNet(cases[0].bytes.data())->UnPack());
    auto square = make(OpType_UnaryOp, OpParameter_UnaryOp);
    square->main.value = new UnaryOpT;
    square->main.AsUnaryOp()->opType = UnaryOpOperation_SQUARE;
    square->name = "square";
    square->inputIndexes = {1};
    square->outputIndexes = {2};
    net->oplists.push_back(std::move(square));
    net->tensorName.push_back("squared");
    net->outputName = {"squared"};
    net->tensorNumber = 3;
    flatbuffers::FlatBufferBuilder fb;
    fb.Finish(Net::Pack(fb, net.get()));
    std::shared_ptr<Interpreter> model(Interpreter::createFromBuffer(fb.GetBufferPointer(), fb.GetSize()),
                                       Interpreter::destroy);
    ScheduleConfig config;
    config.type = MNN_FORWARD_NN;
    auto session = model->createSession(config);
    if (!session) {
        return false;
    }
    auto input = model->getSessionInput(session, nullptr);
    Tensor host(input, Tensor::CAFFE);
    for (int i = 0; i < 4; ++i) {
        host.host<float>()[i] = i + 1;
    }
    input->copyFromHostTensor(&host);
    auto finalOutput = model->getSessionOutput(session, nullptr);
    bool valid = true;
    for (bool skip : {false, true}) {
        input->copyFromHostTensor(&host);
        int visited = 0;
        std::vector<float> skippedValues;
        auto code = model->runSessionWithCallBackInfo(
            session,
            [&](const std::vector<Tensor*>&, const OperatorInfo* info) {
                if (skip && info->name() == "square") {
                    Tensor previous(finalOutput, Tensor::CAFFE);
                    valid = valid && finalOutput->copyToHostTensor(&previous);
                    skippedValues.assign(previous.host<float>(), previous.host<float>() + 4);
                    return false;
                }
                return true;
            },
            [&](const std::vector<Tensor*>& outputs, const OperatorInfo* info) {
                ++visited;
                Tensor actual(outputs[0], Tensor::CAFFE);
                valid = valid && model->getBackend(session, outputs[0])->type() == MNN_FORWARD_NN &&
                        outputs[0]->copyToHostTensor(&actual);
                if (info->name() == "negate") {
                    for (int i = 0; i < 4; ++i) {
                        valid = valid && actual.host<float>()[i] == -(i + 1);
                        actual.host<float>()[i] = 4.f;
                    }
                    valid = valid && outputs[0]->copyFromHostTensor(&actual);
                } else {
                    // A skipped command must leave its output storage unchanged.
                    for (int i = 0; i < 4; ++i) {
                        valid = valid && actual.host<float>()[i] == (skip ? skippedValues[i] : 16.f);
                    }
                }
                return true;
            });
        if (code != NO_ERROR || visited != 2 || !valid) {
            MNN_PRINT("callback code=%d visited=%d valid=%d\n", code, visited, valid);
            return false;
        }
    }
    input->copyFromHostTensor(&host);
    if (model->runSession(session) != NO_ERROR) {
        return false;
    }
    auto output = model->getSessionOutput(session, nullptr);
    Tensor actual(output, Tensor::CAFFE);
    if (!output->copyToHostTensor(&actual)) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        valid = valid && actual.host<float>()[i] == (i + 1) * (i + 1);
    }
    MNN_PRINT("CoreML callback intermediate/override/skip and fused re-entry %s\n", valid ? "PASS" : "FAIL");
    return valid;
}
class CoreMLGraphTest : public MNNTestCase {
public:
    bool run(int) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_NN) {
            ++MNNTestSuite::get()->skipped;
            return true;
        }
        auto seed = makeCases()[0];
        for (int variant = 0; variant < 3; ++variant) {
            std::unique_ptr<NetT> net(GetNet(seed.bytes.data())->UnPack());
            auto unary = [&](int input, int output, UnaryOpOperation kind) {
                auto op = make(OpType_UnaryOp, OpParameter_UnaryOp);
                op->name = "unary" + std::to_string(output);
                op->inputIndexes = {input};
                op->outputIndexes = {output};
                op->main.value = new UnaryOpT;
                op->main.AsUnaryOp()->opType = kind;
                net->oplists.push_back(std::move(op));
            };
            unary(1, 2, UnaryOpOperation_SQUARE);
            unary(1, 3, variant == 2 ? UnaryOpOperation_ERFINV : UnaryOpOperation_NEG);
            auto sum = make(OpType_BinaryOp, OpParameter_BinaryOp);
            sum->name = "branch_sum";
            sum->inputIndexes = {2, 3};
            sum->outputIndexes = {4};
            sum->main.value = new BinaryOpT;
            sum->main.AsBinaryOp()->opType = BinaryOpOperation_ADD;
            net->oplists.push_back(std::move(sum));
            net->tensorName = {"input", "branch", "squared", "other", "output"};
            net->tensorNumber = 5;
            net->outputName = {"branch", "output"};
            flatbuffers::FlatBufferBuilder fb;
            fb.Finish(Net::Pack(fb, net.get()));
            CVCase item;
            item.name = variant == 0 ? "Graph_branch" : variant == 1 ? "Graph_resize" : "Graph_partition";
            item.bytes.assign(fb.GetBufferPointer(), fb.GetBufferPointer() + fb.GetSize());
            item.expectedFallbacks = variant == 2 ? 1 : 0;
            if (!runCase(item)) {
                return false;
            }
        }
        return checkCallbacks();
    }
};
MNNTestSuiteRegister(CoreMLGraphTest, "backend/coreml/graph");

class CoreMLModelTest : public MNNTestCase {
public:
    bool run(int) override {
        const char* path = std::getenv("MNN_COREML_TEST_MODEL");
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_NN || !path) {
            ++MNNTestSuite::get()->skipped;
            return true;
        }
        std::ifstream file(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
        if (bytes.empty()) {
            return false;
        }
        std::shared_ptr<Interpreter> cpu(Interpreter::createFromBuffer(bytes.data(), bytes.size()),
                                         Interpreter::destroy);
        BackendConfig cpuConfig, config;
        cpuConfig.precision = BackendConfig::Precision_High;
        config.precision = BackendConfig::Precision_Low;
        ScheduleConfig schedule;
        schedule.numThread = 4;
        schedule.backendConfig = &cpuConfig;
        auto session = cpu->createSession(schedule);
        if (!session) {
            return false;
        }
        std::vector<std::string> inputNames, outputNames;
        std::map<std::string, std::vector<float>> values;
        for (const auto& item : cpu->getSessionInputAll(session)) {
            inputNames.push_back(item.first);
            Tensor host(item.second, Tensor::CAFFE);
            auto& input = values[item.first];
            input.resize(host.elementSize());
            for (int i = 0; i < input.size(); ++i) {
                input[i] = std::sin(i * .017f) * .5f + .5f;
            }
        }
        const char* inputPath = std::getenv("MNN_COREML_TEST_INPUT");
        if (inputPath && std::string(inputPath) != "-") {
            if (inputNames.size() != 1) {
                return false;
            }
            std::ifstream input(inputPath);
            for (auto& value : values.begin()->second) {
                if (!(input >> value)) {
                    return false;
                }
            }
        }
        for (const auto& item : cpu->getSessionOutputAll(session)) {
            outputNames.push_back(item.first);
        }
        using namespace MNN::Express;
        auto executor = Executor::newExecutor(MNN_FORWARD_NN, config, 4);
        ExecutorScope scope(executor);
        schedule.type = MNN_FORWARD_NN;
        schedule.backendConfig = &config;
        std::shared_ptr<Executor::RuntimeManager> runtime(Executor::RuntimeManager::createRuntimeManager(schedule),
                                                          Executor::RuntimeManager::destroy);
        runtime->setMode(Interpreter::Session_Debug);
        int native = 0, fallback = 0;
        executor->setCallBack([](const std::vector<Tensor*>&, const OperatorInfo*) { return true; },
                              [&](const std::vector<Tensor*>& tensors, const OperatorInfo*) {
                                  auto backend = tensors.empty()
                                                     ? nullptr
                                                     : TensorUtils::getDescribeOrigin(tensors[0])->getBackend();
                                  if (backend && backend->type() == MNN_FORWARD_NN) {
                                      ++native;
                                  } else {
                                      ++fallback;
                                  }
                                  return true;
                              });
        std::shared_ptr<Module> module(Module::load(inputNames, outputNames,
                                                    reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(),
                                                    runtime),
                                       Module::destroy);
        if (!module) {
            return false;
        }
        std::vector<VARP> inputs;
        for (auto& name : inputNames) {
            auto tensor = cpu->getSessionInput(session, name.c_str());
            inputs.push_back(_Input(tensor->shape(), tensor->getDimensionType() == Tensor::TENSORFLOW ? NHWC : NCHW,
                                    halide_type_of<float>()));
        }
        for (int run = 0; run < 2; ++run) {
            for (int i = 0; i < inputNames.size(); ++i) {
                auto tensor = cpu->getSessionInput(session, inputNames[i].c_str());
                Tensor host(tensor,
                            tensor->getDimensionType() == Tensor::TENSORFLOW ? Tensor::TENSORFLOW : Tensor::CAFFE);
                auto& data = values[inputNames[i]];
                std::copy(data.begin(), data.end(), host.host<float>());
                if (!tensor->copyFromHostTensor(&host)) {
                    return false;
                }
                std::copy(data.begin(), data.end(), inputs[i]->writeMap<float>());
            }
            if (cpu->runSession(session) != NO_ERROR) {
                return false;
            }
            auto outputs = module->onForward(inputs);
            if (outputs.size() != outputNames.size()) {
                return false;
            }
            for (int i = 0; i < outputs.size(); ++i) {
                if (!outputs[i].get() || !outputs[i]->readMap<float>()) {
                    return false;
                }
                auto refTensor = cpu->getSessionOutput(session, outputNames[i].c_str());
                auto gotTensor = outputs[i]->getTensor();
                Tensor ref(refTensor, Tensor::CAFFE), got(gotTensor, Tensor::CAFFE);
                if (ref.shape() != got.shape() || ref.getType() != got.getType() ||
                    !refTensor->copyToHostTensor(&ref) || !gotTensor->copyToHostTensor(&got)) {
                    return false;
                }
                if (ref.getType() == halide_type_of<int>()) {
                    for (int j = 0; j < ref.elementSize(); ++j) {
                        if (ref.host<int>()[j] != got.host<int>()[j]) {
                            return false;
                        }
                    }
                    MNN_PRINT("CoreML module run=%d output=%s integer exact\n", run, outputNames[i].c_str());
                    continue;
                }
                double energy = 0, error = 0, peak = 0, maximum = 0, dot = 0, actualEnergy = 0;
                for (int j = 0; j < ref.elementSize(); ++j) {
                    double a = ref.host<float>()[j], b = got.host<float>()[j];
                    if (!std::isfinite(a) || !std::isfinite(b)) {
                        return false;
                    }
                    energy += a * a;
                    actualEnergy += b * b;
                    error += (a - b) * (a - b);
                    dot += a * b;
                    peak = std::max(peak, std::abs(a));
                    maximum = std::max(maximum, std::abs(a - b));
                }
                double l2 = std::sqrt(error / std::max(energy, 1e-20));
                double normalized = maximum / std::max(peak, 1e-10);
                double cosine =
                    energy == 0 && actualEnergy == 0 ? 1 : dot / std::sqrt(std::max(energy * actualEnergy, 1e-30));
                MNN_PRINT("CoreML module run=%d output=%s l2=%g max=%g cosine=%g\n", run, outputNames[i].c_str(), l2,
                          normalized, cosine);
                if (l2 > .05 || normalized > .05 || cosine < .999) {
                    return false;
                }
                if (std::getenv("CV_CLASSIFIER")) {
                    std::vector<int> expected(ref.elementSize()), actual(ref.elementSize());
                    for (int j = 0; j < expected.size(); ++j) {
                        expected[j] = actual[j] = j;
                    }
                    auto top = [](std::vector<int>& indices, const float* data) {
                        std::sort(indices.begin(), indices.end(),
                                  [&](int a, int b) { return data[a] == data[b] ? a < b : data[a] > data[b]; });
                        indices.resize(std::min<size_t>(5, indices.size()));
                    };
                    top(expected, ref.host<float>());
                    top(actual, got.host<float>());
                    if (expected[0] != actual[0]) {
                        return false;
                    }
                    std::sort(expected.begin(), expected.end());
                    std::sort(actual.begin(), actual.end());
                    if (expected != actual) {
                        return false;
                    }
                }
            }
            for (auto& input : values) {
                for (auto& value : input.second) {
                    value = value * .9f + .037f;
                }
            }
        }
        MNN_PRINT("CoreML MODULE_RESULT native=%d fallback=%d\n", native, fallback);
        return native > 0 && fallback == 0;
    }
};
MNNTestSuiteRegister(CoreMLModelTest, "backend/coreml/model");

class CoreMLCacheConcurrencyTest : public MNNTestCase {
public:
    bool run(int) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_NN) {
            ++MNNTestSuite::get()->skipped;
            return true;
        }
        auto cases = makeCases();
        // Initialize MNN's process-wide backend registry before starting workers.
        if (!runCase(cases[0])) {
            return false;
        }
        std::atomic<bool> success(true);
        std::vector<std::thread> workers;
        for (int worker = 0; worker < 4; ++worker) {
            workers.emplace_back([&, worker]() {
                for (int iteration = 0; iteration < 16; ++iteration) {
                    // Each call owns an independent runtime and MLModel, while
                    // the content-addressed compiled programs are shared.
                    if (!runCase(cases[(iteration + worker) % 8])) {
                        success.store(false);
                        return;
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        return success.load();
    }
};
MNNTestSuiteRegister(CoreMLCacheConcurrencyTest, "backend/coreml/cache_concurrency");
} // namespace
#endif
