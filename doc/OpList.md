MNN Op manual

> The specific parameters of Op can be found in *.fbs, and the special case is specified in the corresponding Op below.

[TOC]

## Concat
*Inputs*:
- `input0`: float32(NHWC|NC4HW4)
- `input1`: float32(NHWC|NC4HW4)
- `inputn`: float32(NHWC|NC4HW4)

*Outputs*:

- `output`: float32

*Backend*:

* - [x] CPU
* - [x] Metal
* - [x] Vulkan
* - [x] OpenCL

*Caffe op*:

- Concat

*Tensorflow op*:
- Concat
- ConcatV2

## Convolution
*Inputs*
- `input`: float32|int8(NC4HW4)

*Outputs*:
- `output`: float32|int8

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- Convolution
- CuDNNGroupedConvolutionForward

*Tensorflow op*:
- Conv2D

## Pooling
*Inputs*:
- `input`: float32(NC4HW4)

*Outpus*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- Pooling

*Tensorflow op*:
- MaxPool
- AvgPool

## ReLU
*Inputs*:
- `inputs`: float32(NHWC|NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- Relu

*Tensorflow op*:
- Relu

## Softmax
*Inputs*:
- `input`: float32(NC4HW4|NHWC)

*Outputs*:
- `outpus`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- Softmax

*Tensorflow op*:
- Softmax

## PReLU

*Inputs*:
- `input`: float32(NC4HW4)

*Outputs*:
- `outpus`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- PReLU

*Tensorflow op*:
- PRelu

## LRN
*Inputs*:
- `input`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- LRN
- CuDNNLRNCrossChannel

*Tensorflow op*:

- LRN

## Scale
*Inputs*:
- `input`: float32(NC4HW4|NHWC)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*
- Scale
- BatchNorm
- CuDNNBatchNorm

*Tensorflow op*:
- FusedBatchNorm

## Eltwise
*Inputs*:
- `input0`: float32(NC4HW4|NHWC)
- `input1`: float32(NC4HW4|NHWC)
- `intputn`: float32(NC4HW4|NHWC)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU: support Product, Sum, Max
- - [x] Metal: support Product, Sum, Max
- - [x] Vulkan: supports Product, Sum
- - [x] OpenCL: supports Product, Sum, Max

*Caffe op*:
- Eltwise

## ConvolutionDepthwise
*Inputs*:
- `input`: float32|int8(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- CuDNNGroupedConvolutionForward
- Convolution(group)

*Tensorflow op*:
- DepthwiseConv2dNative

## ArgMax
*Inputs*:
- `input`: float32(NC4HW4)

*Outpus*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [x] OpenCL

*Caffe op*:
- ArgMax

## Reshape
*Inputs*:
- `input`: float32(NC4HW4|NHWC)
- `shape`: int32(optional)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- Flatten
- Reshape

*Tensorflow*:
- Reshape

## Const
*Inputs*:
- 无

*Outputs*:
- `output`: float32|int32|int8

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Tensorflow op*:
- Const

## Binary
*Inputs*:
- `input0`: float32|int32
- `input1`: float32|int32

*Outputs*:
- `output`: float32|int32

*Backend*:
- - [x] CPU: supports Max, Min, Mul, Add, Sub, ReadDiv, Greater
- - [x] Metal
- - [x] Vulkan: supports float32 only; supports Mul, Add, Sub
- - [x] OpenCL: supports float32 only; supports Mul, Add, Sub, Max

*Caffe op*:

*Tensorflow op*:
- Mul
- Sub
- Add
- Maximum
- RealDiv
- Minimum
- Greater
- BiasAdd

## Deconvolution
*Inputs*:
- `input`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- Deconvolution

*Tensorflow*:
- Conv2DBackpropInput

## Crop
*Inputs*:
- `input0`: float32(NC4HW4)
- `input1`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU: supports Crop Spatial only
- - [x] Metal
- - [x] Vulkan: supports Crop Spatial only
- - [x] OpenCL: supports Crop Spatial only

*Limit*:
- axis >= 2

*Caffe op*:
- Crop

## DetectionOutput
*Inputs*:
- `input`: float32(NC4HW4)
- `confidence`: float32(NC4HW4)
- `priorbox`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Caffe op*:
- DetectionOutput

## ExpandDims
*Inputs*:
- `input`: float32(NHWC)
- `dims`: int32

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:
- ExpandDims

## Interp
*Inputs*:
- `input`: float32(NC4HW4)
- `shape`: int32(optional)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU: supports Nearstneighbor, bilinear, cubic
- - [x] Metal
- - [x] Vulkan: supports bilinear
- - [x] OpenCL: supports bilinear

*Caffe op*:
- Interp

*Tensorflow*:
- ResizeBilinear
- ResizeNearestNeighbor

## LSTM
*Inputs*:
- `input0`: float32(NC4HW4)
- `input1`: float32(NC4HW4, optional)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan: support one input only
- - [x] OpenCL

*Caffe op*:
- LSTM

## Normalize
*Inputs*:
- `input`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:

- - [x] CPU: unsupport acrossSpatial and channelShared
- - [x] Metal
- - [x] Vulkan: unsupport acrossSpatial and channelShared
- - [x] OpenCL

*Caffe op*:

- Normalize

## Permute
*Inputs*:
- `input`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- Permute

## PriorBox
*Inputs*:
- `input`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [x] OpenCL

*Caffe op*:
- PriorBox

## Proposal
*Inputs*:
- `input`: float32(NC4HW4)
- `bounding_box`: float32
- `image_size`: float32

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [ ] OpenCL

## Reduction
*Inputs*:
- `input`: float32|int32(NHWC)

*Outputs*:
- `output`: float32|int32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:
- Min
- Max
- Mean
- Sum
- Prod

## ROIPooling
*Inputs*:
- `input`: float32(NC4HW4)
- `roi`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- ROIPooling

## Sigmoid
*Inputs*:
- `input`: float32(NHWC|NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- Sigmoid

*Tensorflow op*:
- Sigmoid

## Slice
*Inputs*:
- `input`: float32(NHWC|NC4HW4)

*Outputs*:
- `output0`: float32
- `output1`: float32
- `outputn`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan: supports NC4HW4 input with axis = 1 only
- - [x] OpenCL

*Caffe op*:
- Slice

*Tensorflow op*:
- Split

## Squeeze
*Inputs*:
- `input`: float32|int8|int32(NHWC)

*Outputs*:
- `output`: float32|int8|int32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Tensorflow op*:
- Squeeze

## TanH
*Inputs*:
- `input`: float32(NHWC|NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- TanH

*Tensorflow op*:
- TanH

## Tile
*Inputs*:
- `input`: float32(NHWC)
- `multipliers`: int32(NHWC)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [x] OpenCL

*Tensorflow op*:
- Tile

## UnaryOp
*Inputs*:
- `input`: float32(NHWC|NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU: supports Rsqrt, Square, Neg, Exp, Sqrt, Abs, Ceil
- - [x] Metal: supports Rsqrt, Square, Neg, Exp, Sqrt, Abs, Ceil
- - [x] Vulkan: supports Rsqrt, Exp, Sqrt, Abs
- - [x] OpenCL: supports Rsqrt, Exp, Sqrt, Abs

*Caffe op*:
- UnaryOp

*Tensorflow op*:
- Rsqrt
- Square
- Exp
- Neg
- Abs
- Ceil
- Sqrt

## SpatialProduct
*Inputs*:
- `input0`: float32(NC4HW4)
- `input1`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Caffe op*:
- SpatialProduct

## DeconvolutionDepthwise
*Inputs*:
- `input`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

## Resize
*Inputs*:
- `input`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

## AsString
*Inputs*:
- `input`: float32|int32(NHWC)

*Outputs*:
- `output`: float32|int32

*Backend*:
- - [x] CPU: supports Bilinear
- - [ ] Metal
- - [ ] Vulkan: supports Bilinear
- - [ ] OpenCL

*Limit*:
- DataType only support float and bool

*Tensorflow op*:
- AsString

## Cast
*Inputs*:
- `input`: float32|int8|int32(NHWC|NC4HW4)

*Outputs*:
- `output`: float32|int8|int32

*Backend*:
- - [x] CPU
- - [x] Metal: half->int32, int32->half, uint8->half
- - [ ] Vulkan
- - [x] OpenCL

*Tensorflow op*:
- Cast

## StridedSlice
*Inputs*:
- `input`: float32|int32(NHWC)
- `begin`: int32(NHWC)
- `end`: int32(NHWC)
- `strided`: int32(NHWC)

*Outputs*:
- `output`: float32|int32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [x] OpenCL

*Tensorflow op*:
- StridedSlice

## ReduceJoin
*Inputs*:
- `input`: int8(NHWC)
- `axis`: int32(NHWC)

*Outputs*:
- `output`: int8

*Backend*:
- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:
- ReduceJoin

## Pack
*Inputs*:
- `input0`: int32(NHWC)
- `input1`: float32|int32(NHWC)
- `inputn`: float32|int32(NHWC)

*Outputs*:
- `output`: float32|int32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [x] OpenCL

*Tensorflow op*:
- Pack

## ReLU6
*Inputs*:
- `input`: float32(NHWC|NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Tensorflow op*:
- ReLU6

## NonMaxSuppressionV2
*Inputs*:
- `boxes`: float32(NHWC)
- `scores`: float32(NHWC)
- `max_output_size`: int32(NHWC)
- `iou_threshold`: float32(NHWC)

*Outputs*:
- `output`: int32

*Backend*:
- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:
- NonMaxSuppressionV2
- NonMaxSuppressionV3

## SpaceToBatchND
*Inputs*:
- `input`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Tensorflow op*:
- SpaceToBatchND

## BatchToSpaceND
*Inputs*:
- `input`: float32(NC4HW4)

*Outputs*:
- `output`: float32

*Backend*:
- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Tensorflow op*:
- BatchToSpaceND

## Shape

*Inputs*:

- `input`: float32|int32|int8

*Outputs*:

- `output`: int32

*Backend*:

- - [x] CPU
- - [x] Metal
- - [x] Vulkan
- - [x] OpenCL

*Tensorflow op*:

- Shape

## Selu

*Inputs*:

- `input`: float32(NHWC|NC4HW4)

*Outputs*:

- `output`: float32

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- Selu

## tfQuantizedConv2D

*Inputs*:

- `intput`: uint8(NC4HW4)

*Outputs*:

- `output`: uint8

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow Lite op*:

- QuantizedConv2D

## Gather

*Inputs*:

- `params`: float32(NHWC)
- `indices`: int32(NHWC)

*Outputs*:

- `output`: float32

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- Gather

## GatherV2

*Inputs*:

- `params`: float32(NHWC)
- `indices`: int32(NHWC)

*Outputs*:

- `output`: float32

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- GatherV2

## QuantizedReshape

*Inputs*:

- `intput`: uint8(NHWC)

*Outputs*:

- `output`: uint8

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow Lite op*:

- QuantizedReshape

## QuantizedMaxPool

*Inputs*:

- `intput`: uint8(NHWC)

*Outputs*:

- `output`: uint8

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [x] OpenCL

*Tensorflow Lite op*:

- QuantizedMaxPool

## QuantizedAvgPool

*Inputs*:

- `intput`: uint8(NHWC)

*Outputs*:

- `output`: uint8

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [x] OpenCL

*Tensorflow Lite op*:

- QuantizedMaxPool

## TopKV2
*Inputs*:
- `intput`: float32
- `k`: int

*Outputs*:
- `output`: float32
- `indices`: int

*Backend*:
- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- TopKV2

## CropAndResize

*Inputs*:

- `input`: float32(NHWC)
- `boxes`: float32(NHWC)
- `box_index`: int32(NHWC)
- `crop_size`: int32(NHWC)

*Outputs*:

- `output`: int32

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- CropAndResize

## Fill

*Inputs*:

- `dims`: int32(NHWC)
- `value`: int32(NHWC)

*Outputs*:

- `output`: int32

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- Fill

## Range

*Inputs*:

- `start`: int32|int64|float32|double(NHWC)
- `limit`: int32|int64|float32|double(NHWC)
- `delta`: int32|int64|float32|double(NHWC)

*Outputs*:

- `output`: int32|int64|float32|double

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- Range

## Rank

*Inputs*:

- `input`: int32(NHWC)

*Outputs*:

- `output`: int32

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- Rank

## Size

*Inputs*:

- `input`: int32|int64|uint8|float32(NHWC)

*Outputs*:

- `output`: int32

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- Size

## SilceTf

*Inputs*:

- `input`: float32|int32(NHWC)
- `begin`: int32(NHWC)
- `size`: int32(NHWC)

*Outputs*:

- `output`: float32

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- Slice

## Transpose

*Inputs*:

- `input`: float32(NHWC)
- `perm`: int32|int64(NHWC)

*Outputs*:

- `output`: float32

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- Transpose

## Where

*Inputs*:

- `input`: int32(NHWC)

*Outputs*:

- `output`: int64

*Backend*:

- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- Where

## quantizedSoftmax

*Inputs*:

- `intput`: uint8(NHWC)

*Outputs*:

- `output`: uint8

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [x] OpenCL

*Tensorflow Lite op*:

- QuantizedSoftmax

## QuantizedDepthwiseConv2D

*Inputs*:

- `intput`: uint8(NC4HW4)

*Outputs*:

- `output`: uint8

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [x] OpenCL

*Tensorflow Lite op*:

- QuantizeDepthwise

## QuantizedAdd

*Inputs*:

- `input0`: uint8(NHWC)
- `input1`: uint8(NHWC)

*Outputs*:

- `output`: uint8

*Backend*:

- - [x] CPU
- - [x] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow Lite op*:

- QuantizedAdd

## QuantizedLogistic

*Inputs*:

- `intput`: uint8(NHWC)

*Outputs*:

- `output`: uint8

*Backend*:

- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow Lite op*:

- QuantizedLogistic

## Unpack

*Inputs*:

- `intput`: float32(NHWC)

*Outputs*:

- `output0`: float32
- `outputN`: float32

*Backend*:

- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow op*:

- Unpack

## QuantizedConcat

*Inputs*:

- `intput0`: uint8(NHWC)
- `intput1`: uint8(NHWC)
- `intputn`: uint8(NHWC)

*Outputs*:

- `output`: uint8

*Backend*:

- - [x] CPU
- - [ ] Metal
- - [ ] Vulkan
- - [ ] OpenCL

*Tensorflow Lite op*:

- QuantizedConcat

