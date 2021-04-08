//
//  NeuralNetWorkOp.hpp
//  MNN
//
//  Created by MNN on 2019/06/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NeuralNetWorkOp_HPP
#define NeuralNetWorkOp_HPP

namespace MNN {
namespace Express {
enum PaddingMode {CAFFE, VALID, SAME};
enum PoolingMode {MAXPOOL, AVEPOOL};
enum PadValueMode {CONSTANT, REFLECT, SYMMETRIC};
MNN_PUBLIC VARP _Input(INTS shape = {}, Dimensionformat data_format = NC4HW4, halide_type_t dtype = halide_type_of<float>()) ;
MNN_PUBLIC VARP _Clone(VARP source, bool deepCopy = false);

MNN_PUBLIC VARP _Scalar(const void* ptr, halide_type_t type);

template <typename T>
VARP _Scalar(T value) {
    return _Scalar(&value, halide_type_of<T>());
}


MNN_PUBLIC VARP _Const(float value, INTS shape = {}, Dimensionformat format = NHWC);
MNN_PUBLIC VARP _Const(const void* ptr, INTS shape = {}, Dimensionformat format = NHWC,
                       halide_type_t type = halide_type_of<float>());
MNN_PUBLIC VARP _TrainableParam(float value, INTS dims, Dimensionformat format);
MNN_PUBLIC VARP _TrainableParam(const void* ptr, INTS dims, Dimensionformat format,
                                  halide_type_t type = halide_type_of<float>());
MNN_PUBLIC VARP _InnerProduct(std::vector<float>&& weight, std::vector<float>&& bias, VARP x, INTS outputShape);
MNN_PUBLIC VARP _Conv(VARP weight, VARP bias, VARP x, PaddingMode pad = VALID, INTS stride = {1, 1},
                      INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0});

MNN_PUBLIC VARP _Conv(float weight, float bias, VARP x, INTS channel, INTS kernelSize, PaddingMode pad = VALID,
                      INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1);
MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<float>&& bias, VARP x, INTS channel, INTS kernelSize,
                      PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0}, bool relu = false, bool relu6 = false, int nbits = 8);
MNN_PUBLIC VARP _Conv(std::vector<float>&& weight, std::vector<float>&& bias, VARP x, INTS channel, INTS kernelSize,
                      PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0}, bool relu = false, bool relu6 = false);
MNN_PUBLIC VARP _Deconv(VARP weight, VARP bias, VARP x, PaddingMode pad = VALID, INTS stride = {1, 1},
                                INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0});

MNN_PUBLIC VARP _Deconv(std::vector<float>&& weight, std::vector<float>&& bias, VARP x, INTS channel, INTS kernelSize,
PaddingMode pad, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0}, bool relu = false, bool relu6 = false);

MNN_PUBLIC VARP _MaxPool(VARP x, INTS kernel, INTS stride = {1, 1}, PaddingMode pad = VALID, INTS pads= {0, 0});
MNN_PUBLIC VARP _AvePool(VARP x, INTS kernel, INTS stride = {1, 1}, PaddingMode pad = VALID, INTS pads= {0, 0});
MNN_PUBLIC VARP _Reshape(VARP x, INTS shape, Dimensionformat original_format = NCHW);
MNN_PUBLIC VARP _Reshape(VARP x, VARP shape);
MNN_PUBLIC VARP _Scale(VARP x, int channels, std::vector<float>&& scales, std::vector<float>&& bias);

MNN_PUBLIC VARP _Relu(VARP x, float slope = 0.0f);
MNN_PUBLIC VARP _Relu6(VARP x, float minValue = 0.0f, float maxValue = 6.0f);
MNN_PUBLIC VARP _PRelu(VARP x, std::vector<float> &&slopes);
MNN_PUBLIC VARP _Softmax(VARP logits, int axis = -1);
MNN_PUBLIC VARP _Softplus(VARP features);
MNN_PUBLIC VARP _Softsign(VARP features);
MNN_PUBLIC std::vector<VARP> _Split(VARP value, INTS size_splits, int axis = 0);
MNN_PUBLIC VARP _Slice(VARP x, VARP starts, VARP sizes);
MNN_PUBLIC VARP _StridedSlice(VARP input, VARP begin, VARP end, VARP strided,
                                      int32_t beginMask, int32_t endMask, int32_t ellipsisMask,
                                      int32_t newAxisMask, int32_t shrinkAxisMask);
MNN_PUBLIC VARP _Concat(VARPS values, int axis);
MNN_PUBLIC VARP _Convert(VARP input, Dimensionformat format);
MNN_PUBLIC VARP _Transpose(VARP x, INTS perm);
MNN_PUBLIC VARP _Transpose(VARP x, VARP perm);
MNN_PUBLIC VARP _ChannelShuffle(VARP x, int group);
MNN_PUBLIC VARP _ChangeInputFormat(VARP input, Dimensionformat format);
MNN_PUBLIC VARP _Conv2DBackPropFilter(VARP input, VARP inputGrad, INTS kernelSize, PaddingMode pad = VALID, INTS stride = {1, 1}, INTS dilate = {1, 1}, int group = 1, INTS pads = {0, 0});
MNN_PUBLIC VARP _PoolGrad(VARP originInput, VARP originOutput, VARP inputGrad, INTS kernel, INTS stride, PoolingMode type, PaddingMode pad = VALID, INTS pads= {0, 0});
// FIXME: move the api to Array Ops
MNN_PUBLIC VARP _ReverseSequence(VARP x, VARP y, int batchDim, int seqDim);
// FIXME: move the api to Image Ops
MNN_PUBLIC VARP _Crop(VARP images, VARP size, int axis, INTS offset);
MNN_PUBLIC VARP _Resize(VARP images, float xScale, float yScale);
MNN_PUBLIC VARP _Pad(VARP x, VARP paddings, PadValueMode mode = CONSTANT);
MNN_PUBLIC VARP _ExpandDims(VARP input, int axis);
MNN_PUBLIC VARP _ExpandDims(VARP input, VARP axis);

MNN_PUBLIC VARP _Shape(VARP input, bool nchw = false);
MNN_PUBLIC VARP _Stack(VARPS values, int axis=0);
enum InterpolationMethod {BILINEAR, NEAREST};
MNN_PUBLIC VARP _CropAndResize(VARP image, VARP boxes, VARP box_ind, VARP crop_size, 
                                InterpolationMethod method, float extrapolation_value = 0.0);
MNN_PUBLIC VARP _Fill(VARP dims, VARP value);
MNN_PUBLIC VARP _Tile(VARP input, VARP multiples);
MNN_PUBLIC VARP _Gather(VARP params, VARP indices);
MNN_PUBLIC VARP _GatherV2(VARP params, VARP indices, VARP axis = nullptr);
MNN_PUBLIC VARP _Squeeze(VARP input, INTS axis = {});
MNN_PUBLIC VARP _Unsqueeze(VARP input, INTS axis = {});
MNN_PUBLIC VARP _BatchToSpaceND(VARP input, VARP block_shape, VARP crops);
MNN_PUBLIC VARP _GatherND(VARP params, VARP indices);
MNN_PUBLIC VARP _Selu(VARP features, float scale, float alpha);
MNN_PUBLIC VARP _Size(VARP input);
MNN_PUBLIC VARP _Elu(VARP features, float alpha=1.0);
MNN_PUBLIC VARP _Threshold(VARP features, float alpha=1.0);
MNN_PUBLIC VARP _MatrixBandPart(VARP input, VARP num_lower, VARP num_upper);
MNN_PUBLIC std::vector<VARP> _Moments(VARP x, INTS axis, VARP shift, bool keepDims);
MNN_PUBLIC VARP _SetDiff1D(VARP x, VARP y); 
MNN_PUBLIC VARP _SpaceToDepth(VARP input, int block_size);
MNN_PUBLIC VARP _SpaceToBatchND(VARP input, VARP block_shape, VARP paddings);
MNN_PUBLIC VARP _ZerosLike(VARP input);
MNN_PUBLIC std::vector<VARP> _Unstack(VARP value, int axis=0);
MNN_PUBLIC VARP _Rank(VARP input);
MNN_PUBLIC VARP _Range(VARP start, VARP limit, VARP delta);
MNN_PUBLIC VARP _DepthToSpace(VARP input, int block_size);
MNN_PUBLIC VARP _PriorBox(VARP feature, VARP image, 
                            std::vector<float> min_size, std::vector<float> max_size, std::vector<float>aspect_ratio, 
                            bool flip, bool clip, std::vector<float>variance,
                            unsigned int img_h, unsigned int img_w, float step_h, float step_w, float offset = 0.5);
MNN_PUBLIC VARP _Permute(VARP input, INTS dims);
MNN_PUBLIC VARP _DetectionOutput(VARP location, VARP confidence, VARP priorbox, 
                        unsigned int num_classes, bool share_location, int background_label_id, 
                        float nms_threshhold, int nms_topk, int code_type, 
                        bool variance_encoded_in_target,
                        int keep_top_k, float confidence_threshold, float visualize_threshold); 
MNN_PUBLIC  std::vector<VARP> _DetectionPostProcess(VARP encode_boxes, VARP class_predictions, VARP anchors, 
                        int num_classes, int max_detections, 
                        int max_class_per_detection, int detections_per_class, 
                        float nms_threshold, float iou_threshold, 
                        bool use_regular_nms, std::vector<float> centersize_encoding);
MNN_PUBLIC VARP _Interp(VARPS xs, float widthScale, float heightScale, int outputWidth, int outputHeight, int resizeType, bool alignCorners);

MNN_PUBLIC VARP _ZeroGrad(VARP x);

// Int8 Inference
MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<int>&& bias, std::vector<float>&& scale, VARP x, INTS channel, INTS kernelSize,
                      PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu, int nbits = 8);
MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<int>&& bias, std::vector<float>&& scale,
                      VARP x, INTS channel, INTS kernelSize,
                      PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu,
                      int8_t inputZeroPoint, int8_t outputZeroPoint,
                      int8_t minValue, int8_t maxValue, bool accumulateToInt16);
MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<float>&& bias, std::vector<float>&& weightScale,
                      VARP x, INTS channel, INTS kernelSize,
                      PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu,
                      float scaleIn, float scaleOut,
                      int8_t inputZeroPoint, int8_t outputZeroPoint,
                      int8_t minValue, int8_t maxValue, float weightClampValue, bool accumulateToInt16);
MNN_PUBLIC VARP _CosineSimilarity(VARP input0, VARP input1, VARP inputDim);

enum GridSamplePaddingMode {GRID_SAMPLE_PADDING_ZEROS, GRID_SAMPLE_PADDING_BORDER, GRID_SAMPLE_PADDING_REFLECTION};
MNN_PUBLIC VARP _GridSample(VARP input, VARP grid, InterpolationMethod mode=BILINEAR, GridSamplePaddingMode paddingMode=GRID_SAMPLE_PADDING_ZEROS, bool alignCorners=false);
MNN_PUBLIC VARP _FloatToInt8(VARP x, VARP scale, char minValue, char maxValue);
MNN_PUBLIC VARP _FloatToInt8(VARP x, VARP scale, int8_t minValue, int8_t maxValue, int8_t zeroPoint);
MNN_PUBLIC VARP _Int8ToFloat(VARP x, VARP scale);
MNN_PUBLIC VARP _Int8ToFloat(VARP x, VARP scale, int8_t zeroPoint);

MNN_PUBLIC VARP _Select(VARP select, VARP input0, VARP input1);
MNN_PUBLIC std::vector<VARP> _TopKV2(VARP input0, VARP input1);

} // namespace Express
} // namespace MNN

#endif /* NeuralNetWorkOp_HPP */
