//
//  NeuralNetWorkOp.hpp
//  MNN
//
//  Created by MNN on 2019/06/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

namespace MNN {
namespace Express {
enum PaddingMode {CAFFE, VALID, SAME};
enum PoolingMode {MAXPOOL, AVEPOOL};
MNN_EXPRESS_PUBLIC VARP _Input(INTS dims = {}, Dimensionformat format = NC4HW4, halide_type_t type = halide_type_of<float>());
MNN_EXPRESS_PUBLIC VARP _Clone(VARP source, bool deepCopy=false);

MNN_EXPRESS_PUBLIC VARP _Const(float value, INTS dims = {}, Dimensionformat format = NHWC);
MNN_EXPRESS_PUBLIC VARP _Const(const void* ptr, INTS dims = {}, Dimensionformat format = NHWC,
                       halide_type_t type = halide_type_of<float>());
MNN_EXPRESS_PUBLIC VARP _Conv(VARP weight, VARP bias, VARP x, PaddingMode pad = VALID, INTS stride = {1, 1},
                      INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0});

MNN_EXPRESS_PUBLIC VARP _Conv(float weight, float bias, VARP x, INTS channel, INTS kernelSize, PaddingMode pad = VALID,
                      INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1);
MNN_EXPRESS_PUBLIC VARP _Conv(std::vector<float>&& weight, std::vector<float>&& bias, VARP x, INTS channel, INTS kernelSize,
                      PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0});
MNN_EXPRESS_PUBLIC VARP _Deconv(VARP weight, VARP bias, VARP x, PaddingMode pad = VALID, INTS stride = {1, 1},
                                INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0});
MNN_EXPRESS_PUBLIC VARP _MaxPool(VARP x, INTS kernel, INTS stride, PaddingMode pad = VALID, INTS pads= {0, 0});
MNN_EXPRESS_PUBLIC VARP _AvePool(VARP x, INTS kernel, INTS stride, PaddingMode pad = VALID, INTS pads= {0, 0});
MNN_EXPRESS_PUBLIC VARP _Reshape(VARP x, INTS dim, Dimensionformat format);
MNN_EXPRESS_PUBLIC VARP _Reshape(VARP x, VARP shape);
MNN_EXPRESS_PUBLIC VARP _Scale(VARP x, int channels, std::vector<float>&& scales, std::vector<float>&& bias);

MNN_EXPRESS_PUBLIC VARP _Relu(VARP x, float slope = 0.0f);
MNN_EXPRESS_PUBLIC VARP _Relu6(VARP x);
MNN_EXPRESS_PUBLIC VARP _Softmax(VARP x, int axis);
MNN_EXPRESS_PUBLIC std::vector<VARP> _Slice(VARP x, INTS points, int axis);
MNN_EXPRESS_PUBLIC VARP _Concat(VARPS xs, int axis);
MNN_EXPRESS_PUBLIC VARP _Convert(VARP x, Dimensionformat dest);
MNN_EXPRESS_PUBLIC VARP _Transpose(VARP x, INTS perm);
MNN_EXPRESS_PUBLIC VARP _Transpose(VARP x, VARP perm);
MNN_EXPRESS_PUBLIC VARP _ChannelShuffle(VARP x, int group);
MNN_EXPRESS_PUBLIC VARP _ChangeInputFormat(VARP x, Dimensionformat requireInput);
MNN_EXPRESS_PUBLIC VARP _Conv2DBackPropFilter(VARP weight, VARP input, VARP inputGrad, PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0});
MNN_EXPRESS_PUBLIC VARP _PoolGrad(VARP originInput, VARP originOutput, VARP inputGrad, INTS kernel, INTS stride, PoolingMode type, PaddingMode pad = VALID, INTS pads= {0, 0});
// FIXME: move the api to Array Ops
MNN_EXPRESS_PUBLIC VARP _ReverseSequence(VARP x, VARP y, int batchDim, int seqDim);
// FIXME: move the api to Image Ops
MNN_EXPRESS_PUBLIC VARP _Crop(VARP x, VARP s, int axis, INTS offset);
MNN_EXPRESS_PUBLIC VARP _Resize(VARP x, float xScale, float yScale);
MNN_EXPRESS_PUBLIC VARP _Pad(VARP x, VARP pads);
MNN_EXPRESS_PUBLIC VARP _ExpandDims(VARP x, int axis);
MNN_EXPRESS_PUBLIC VARP _ExpandDims(VARP x, VARP axis);
    
MNN_EXPRESS_PUBLIC VARP _Pack(VARPS xs, halide_type_t dtype, int axis);
enum InterpolationMethod {BILINEAR, NEAREST};
MNN_EXPRESS_PUBLIC VARP _CropAndResize(VARP image, VARP boxes, VARP indexes, VARP sizes, float extrapolation, InterpolationMethod method);
MNN_EXPRESS_PUBLIC VARP _Fill(VARP s, VARP v);
MNN_EXPRESS_PUBLIC VARP _Tile(VARP x, VARP mul);
MNN_EXPRESS_PUBLIC VARP _GatherV2(VARP params, VARP indices, VARP axis = nullptr);
    
} // namespace Express
} // namespace MNN
