/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly
 * prohibited.
 */
/* MediaTek Inc. (C) 2020. All rights reserved.
 *
 * BY OPENING THIS FILE, RECEIVER HEREBY UNEQUIVOCALLY ACKNOWLEDGES AND AGREES
 * THAT THE SOFTWARE/FIRMWARE AND ITS DOCUMENTATIONS ("MEDIATEK SOFTWARE")
 * RECEIVED FROM MEDIATEK AND/OR ITS REPRESENTATIVES ARE PROVIDED TO RECEIVER ON
 * AN "AS-IS" BASIS ONLY. MEDIATEK EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR NONINFRINGEMENT.
 * NEITHER DOES MEDIATEK PROVIDE ANY WARRANTY WHATSOEVER WITH RESPECT TO THE
 * SOFTWARE OF ANY THIRD PARTY WHICH MAY BE USED BY, INCORPORATED IN, OR
 * SUPPLIED WITH THE MEDIATEK SOFTWARE, AND RECEIVER AGREES TO LOOK ONLY TO SUCH
 * THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO. RECEIVER EXPRESSLY
 * ACKNOWLEDGES THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO OBTAIN FROM ANY
 * THIRD PARTY ALL PROPER LICENSES CONTAINED IN MEDIATEK SOFTWARE. MEDIATEK
 * SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK SOFTWARE RELEASES MADE TO
 * RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR STANDARD OR OPEN
 * FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S ENTIRE AND
 * CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE RELEASED HEREUNDER
 * WILL BE, AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE MEDIATEK SOFTWARE AT
 * ISSUE, OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE CHARGE PAID BY RECEIVER
 * TO MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek
 * Software") have been modified by MediaTek Inc. All revisions are subject to
 * any receiver's applicable license agreements with MediaTek Inc.
 */

/**
 * @file NeuronAdapter.h
 */

#pragma once

#ifdef __ANDROID__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-extension"
#include <android/hardware_buffer.h>
#pragma clang diagnostic pop
#endif

#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/**
 * NeuronModel is an opaque type that contains a description of the mathematical
 * operations that constitute the model.
 */
typedef struct NeuronModel NeuronModel;

/**
 * NeuronCompilation is an opaque type that can be used to compile a machine
 * learning model.
 */
typedef struct NeuronCompilation NeuronCompilation;

/**
 * NeuronExecution is an opaque type that can be used to apply a machine
 * learning model to a set of inputs.
 */
typedef struct NeuronExecution NeuronExecution;

/**
 * NeuronDevice is an opaque type that represents a device.
 *
 * This type is used to query basic properties and supported operations of the
 * corresponding device, and control which device(s) a model is to be run on.
 *
 * Available since 4.1.0
 */
typedef struct NeuronDevice NeuronDevice;

/**
 * This type is used to represent shared memory, memory mapped files, and
 * similar memories.
 *
 * It is the application's responsibility to ensure that there are no uses of
 * the memory after calling NeuronMemory_free. This includes the execution which
 * references this memory because of a call to
 * NeuronExecution_setInputFromMemory or NeuronExecution_setOutputFromMemory.
 *
 * Available since 4.1.0
 */
typedef struct NeuronMemory NeuronMemory;

/**
 * NeuronEvent is an opaque type that represents an event
 * that will be signaled once an execution completes.
 *
 * Available since 5.0.0
 */
typedef struct NeuronEvent NeuronEvent;

/**
 * Result codes.
 */
typedef enum {
  NEURON_NO_ERROR = 0,
  NEURON_OUT_OF_MEMORY = 1,
  NEURON_INCOMPLETE = 2,
  NEURON_UNEXPECTED_NULL = 3,
  NEURON_BAD_DATA = 4,
  NEURON_OP_FAILED = 5,
  NEURON_UNMAPPABLE = 6,
  NEURON_BAD_STATE = 7,
  NEURON_BAD_VERSION = 8,

  // Available since 5.0.0
  NEURON_OUTPUT_INSUFFICIENT_SIZE = 9,
  NEURON_UNAVAILABLE_DEVICE = 10,
  NEURON_MISSED_DEADLINE_TRANSIENT = 11,
  NEURON_MISSED_DEADLINE_PERSISTENT = 12,
  NEURON_RESOURCE_EXHAUSTED_TRANSIENT = 13,
  NEURON_RESOURCE_EXHAUSTED_PERSISTENT = 14,
  NEURON_DEAD_OBJECT = 15,
} NeuronAdapterResultCode;

/**
 * Operand values with size in bytes that are smaller or equal to this will be
 * immediately copied into the model.
 */
enum { NEURON_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES = 128 };

/**
 * Size of the cache token, in bytes, required from the application.
 */
enum { NEURON_BYTE_SIZE_OF_CACHE_TOKEN = 32 };

/**
 * Operand types.
 * The type of operands that can be added to a model.
 *
 * Some notes on quantized tensors
 *
 * <p>NEURON_TENSOR_QUANT8_ASYMM
 * <p>Attached to this tensor are two numbers that can be used to convert the 8
 * bit integer to the real value and vice versa. These two numbers are:
 * - scale: a 32 bit floating point value greater than zero.
 * - zeroPoint: a 32 bit integer, in range [0, 255].
 * <p>The formula is: real_value = (integer_value - zero_value) * scale.
 *
 * <p>NEURON_TENSOR_QUANT16_SYMM
 * <p>Attached to this tensor is a number representing real value scale that is
 * used to convert the 16 bit number to a real value in the following way:
 * realValue = integerValue * scale. scale is a 32 bit floating point with value
 * greater than zero.
 *
 * <p>NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL
 * <p>This tensor is associated with additional fields that can be used to
 * convert the 8 bit signed integer to the real value and vice versa. These
 * fields are:
 * - channelDim: a 32 bit unsigned integer indicating channel dimension.
 * - scales: an array of positive 32 bit floating point values.
 * <p>The size of the scales array must be equal to dimensions[channelDim].
 * NeuronModel_setOperandSymmPerChannelQuantParams must be used to set the
 * parameters for an Operand of this type. The channel dimension of this tensor
 * must not be unknown (dimensions[channelDim] != 0). The formula is:
 * realValue[..., C, ...] = integerValue[..., C, ...] * scales[C] where C is an
 * index in the Channel dimension.
 *
 * <p>NEURON_TENSOR_QUANT16_ASYMM
 * <p>Attached to this tensor are two numbers that can be used to convert the 16
 * bit integer to the real value and vice versa. These two numbers are:
 * - scale: a 32 bit floating point value greater than zero.
 * - zeroPoint: a 32 bit integer, in range [0, 65535].
 * <p>The formula is: real_value = (integer_value - zeroPoint) * scale.
 *
 * <p>NEURON_TENSOR_QUANT8_SYMM
 * <p>Attached to this tensor is a number representing real value scale that is
 * used to convert the 8 bit number to a real value in the following way:
 * realValue = integerValue * scale. scale is a 32 bit floating point with value
 * greater than zero.
 *
 * <p>NEURON_TENSOR_QUANT8_ASYMM_SIGNED
 * <P>Attached to this tensor are two numbers that can be used to convert the 8
 * bit integer to the real value and vice versa. These two numbers are:
 * - scale: a 32 bit floating point value greater than zero.
 * - zeroPoint: a 32 bit integer, in range [-128, 127].
 * <p>The formula is: real_value = (integer_value - zeroPoint) * scale.
 */
enum {
  /** A 32 bit floating point scalar value. */
  NEURON_FLOAT32 = 0,
  /** A signed 32 bit integer scalar value. */
  NEURON_INT32 = 1,
  /** An unsigned 32 bit integer scalar value. */
  NEURON_UINT32 = 2,
  /** A tensor of 32 bit floating point values. */
  NEURON_TENSOR_FLOAT32 = 3,
  /** A tensor of 32 bit integer values. */
  NEURON_TENSOR_INT32 = 4,
  /** A tensor of 8 bit integers that represent real numbers. */
  NEURON_TENSOR_QUANT8_ASYMM = 5,
  /** An 8 bit boolean scalar value. */
  NEURON_BOOL = 6,
  /** A tensor of 16 bit signed integers that represent real numbers. */
  NEURON_TENSOR_QUANT16_SYMM = 7,
  /** A tensor of IEEE 754 16 bit floating point values. */
  NEURON_TENSOR_FLOAT16 = 8,
  /** A tensor of 8 bit boolean values. */
  NEURON_TENSOR_BOOL8 = 9,
  /** An IEEE 754 16 bit floating point scalar value. */
  NEURON_FLOAT16 = 10,
  /** A tensor of 8 bit signed integers that represent real numbers. */
  NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL = 11,
  /** A tensor of 16 bit unsigned integers that represent real numbers. */
  NEURON_TENSOR_QUANT16_ASYMM = 12,
  /** A tensor of 8 bit signed integers that represent real numbers. */
  NEURON_TENSOR_QUANT8_SYMM = 13,
  /** A tensor of 8 bit signed integers that represent real numbers. */
  NEURON_TENSOR_QUANT8_ASYMM_SIGNED = 14,
  /** A reference to a model. */
  NEURON_MODEL = 15,
  /** Extended data type - tensor uint32 */
  NEURON_EXT_TENSOR_UINT32 = 9001,
  /** Extended data type -A tensor of 8 bit unsigned integers that represent
     real numbers. */
  NEURON_EXT_TENSOR_QUANT8_ASYMM_PER_CHANNEL = 9002,
  /** Extended data type -A tensor of 4 bit unsigned integers that represent
     real numbers. */
  NEURON_EXT_TENSOR_QUANT4_ASYMM = 9003,
  /** Extended data type -A tensor of 4 bit signed integers that represent real
     numbers. */
  NEURON_EXT_TENSOR_QUANT4_ASYMM_SIGNED = 9004,
  /** Extended data type -A tensor of 4 bit signed integers that represent real
     numbers. */
  NEURON_EXT_TENSOR_QUANT4_SYMM = 9005,
  /** Extended data type -A tensor of 16 bit signed integers that represent real
     numbers. */
  NEURON_EXT_TENSOR_QUANT16_ASYMM_SIGNED = 9006,
  /** Extended data type -A raw tensor. */
  NEURON_EXT_TENSOR_RAW = 9007,
  /** Extended data type -A tensor of 8 bit signed integers that represent real
     numbers. */
  NEURON_EXT_TENSOR_QUANT8_ASYMM_SIGNED_PER_CHANNEL = 9008,
};

/**
 * NeuronOperandType describes the type of an operand.
 * This structure is used to describe both scalars and tensors.
 */
typedef struct NeuronOperandType {
  /** The data type, e.g NEURON_INT8. */
  int32_t type;
  /** The number of dimensions. It should be 0 for scalars. */
  uint32_t dimensionCount;
  /** The dimensions of the tensor. It should be nullptr for scalars. */
  const uint32_t* dimensions;
  /**
   * These two fields are only used for quantized tensors.
   * They should be zero for scalars and non-fixed point tensors.
   * The dequantized value of each entry is (value - zeroPoint) * scale.
   */
  float scale;
  /** Only used with scale for quantized tensors */
  int32_t zeroPoint;
} NeuronOperandType;

/**
 * Parameters for NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL operand.
 */
typedef struct NeuronSymmPerChannelQuantParams {
  /** The index of the channel dimension. */
  uint32_t channelDim;
  /** The size of the scale array. Should be equal to dimension[channelDim] of
   * the Operand. */
  uint32_t scaleCount;
  /** The array of scaling values for each channel. Each value must be greater
   * than zero. */
  const float* scales;
} NeuronSymmPerChannelQuantParams;

/**
 * Parameters for NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL and
 * NEURON_TENSOR_QUANT8_ASYMM_PER_CHANNEL operand.
 */
typedef struct NeuronPerChannelQuantParams {
  /** The index of the channel dimension. */
  uint32_t channelDim;
  /** The size of the scale array. Should be equal to dimension[channelDim] of
   * the Operand. */
  uint32_t scaleCount;
  /** The array of scaling values for each channel. Each value must be greater
   * than zero. */
  const float* scales;
  /** The size of the zeroPoints. Should be equal to dimension[channelDim] of
   * the Operand. */
  uint32_t zeroPointCount;
  /** The array of zero point values for each channel. */
  const int32_t* zeroPoints;
} NeuronPerChannelQuantParams;

/**
 * Operation Types
 *
 * Supported operations are listed with available versions. See
 * Neuron_getVersion for querying version number.
 *
 * Attempting to compile models with operations marked as not available
 * will get a compilation failure.
 *
 * Refer to the operation support status of each hardware platform.
 * Attempting to compile models with operations supported by this library but
 * not supported by the underlying hardware platform will get a compilation
 * failure too.
 *
 * Compatible NNAPI levels are also listed.
 */
typedef enum {
  NEURON_ADD = 0, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_AVERAGE_POOL_2D = 1, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_CONCATENATION = 2, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_CONV_2D = 3, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_DEPTHWISE_CONV_2D = 4, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_DEPTH_TO_SPACE = 5, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_DEQUANTIZE = 6, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_EMBEDDING_LOOKUP = 7, ///< Not available.
  NEURON_FLOOR = 8, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_FULLY_CONNECTED = 9, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_HASHTABLE_LOOKUP = 10, ///< Not available.
  NEURON_L2_NORMALIZATION = 11, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_L2_POOL_2D = 12, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_LOCAL_RESPONSE_NORMALIZATION = 13, ///< Not available.
  NEURON_LOGISTIC = 14, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_LSH_PROJECTION = 15, ///< Not available.
  NEURON_LSTM = 16, ///< Not available.
  NEURON_MAX_POOL_2D = 17, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_MUL = 18, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_RELU = 19, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_RELU1 = 20, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_RELU6 = 21, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_RESHAPE = 22, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_RESIZE_BILINEAR = 23, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_RNN = 24, ///< Not available.
  NEURON_SOFTMAX = 25, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_SPACE_TO_DEPTH = 26, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_SVDF = 27, ///< Not available.
  NEURON_TANH = 28, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_BATCH_TO_SPACE_ND = 29, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_DIV = 30, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_MEAN = 31, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_PAD = 32, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_SPACE_TO_BATCH_ND = 33, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_SQUEEZE = 34, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_STRIDED_SLICE = 35, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_SUB = 36, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_TRANSPOSE = 37, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_ABS = 38, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_ARGMAX = 39, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_ARGMIN = 40, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_AXIS_ALIGNED_BBOX_TRANSFORM =
      41, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_BIDIRECTIONAL_SEQUENCE_LSTM = 42, ///< Not available.
  NEURON_BIDIRECTIONAL_SEQUENCE_RNN = 43, ///< Not available.
  NEURON_BOX_WITH_NMS_LIMIT = 44, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_CAST = 45, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_CHANNEL_SHUFFLE = 46, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_DETECTION_POSTPROCESSING = 47, ///< Not available.
  NEURON_EQUAL = 48, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_EXP = 49, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_EXPAND_DIMS = 50, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_GATHER = 51, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_GENERATE_PROPOSALS = 52, ///< Not available.
  NEURON_GREATER = 53, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_GREATER_EQUAL = 54, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_GROUPED_CONV_2D = 55, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_HEATMAP_MAX_KEYPOINT = 56, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_INSTANCE_NORMALIZATION =
      57, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_LESS = 58, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_LESS_EQUAL = 59, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_LOG = 60, ///< Not available.
  NEURON_LOGICAL_AND = 61, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_LOGICAL_NOT = 62, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_LOGICAL_OR = 63, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_LOG_SOFTMAX = 64, ///< Not available.
  NEURON_MAXIMUM = 65, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_MINIMUM = 66, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_NEG = 67, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_NOT_EQUAL = 68, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_PAD_V2 = 69, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_POW = 70, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_PRELU = 71, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_QUANTIZE = 72, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_QUANTIZED_16BIT_LSTM = 73, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_RANDOM_MULTINOMIAL = 74, ///< Not available.
  NEURON_REDUCE_ALL = 75, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_REDUCE_ANY = 76, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_REDUCE_MAX = 77, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_REDUCE_MIN = 78, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_REDUCE_PROD = 79, ///< Not available.
  NEURON_REDUCE_SUM = 80, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_ROI_ALIGN = 81, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_ROI_POOLING = 82, ///< Not available.
  NEURON_RSQRT = 83, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_SELECT = 84, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_SIN = 85, ///< Not available.
  NEURON_SLICE = 86, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_SPLIT = 87, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_SQRT = 88, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_TILE = 89, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_TOPK_V2 = 90, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_TRANSPOSE_CONV_2D = 91, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_UNIDIRECTIONAL_SEQUENCE_LSTM = 92, ///< Not available.
  NEURON_UNIDIRECTIONAL_SEQUENCE_RNN = 93, ///< Not available.
  NEURON_RESIZE_NEAREST_NEIGHBOR =
      94, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_QUANTIZED_LSTM = 95, ///< Not available.
  NEURON_IF = 96, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_WHILE = 97, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_ELU = 98, ///< Not available.
  NEURON_HARD_SWISH = 99, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_FILL = 100, ///< Available since 4.1.0. NNAPI level 30.
  NEURON_RANK = 101, ///< Not available.
  NEURON_BATCH_MATMUL = 102, ///< Available since 5.1.2. NNAPI FL6.
  NEURON_PACK = 103, ///< Not available.
  NEURON_MIRROR_PAD = 104, ///< Not available.
  NEURON_MIRROR_REVERSE = 105, ///< Not available.
  /**
   * Decompress HyFBC to YUV420 frame, support both YUV420_8BITS and
   * YUV420_10BITS formats. HyFBC (Hybrid Frame Buffer Compression) is a
   * compressed format used by video decoder (VDEC). This format uses YUV420 to
   * compress.
   *
   * For input part, need to set two inputs with different shape, representing Y
   * and UV plane respectively. The same HyFBC data will be used for both
   * inputs. Similarly, the output part also needs to be set to two,
   * representing Y and UV plane respectively.
   *
   * The shape of the two inputs/ outputs (inputY, inputUV, outputY, outputUV)
   * depends on the original images' shape ([batches, height, width, channels]).
   * Both height and width shold follow 64 alignment rule. For example, if
   * original height is 480, its 64 alignment should be 512. For Y plane,
   * channel size should be 1; for UV plane, channel size should be 2. Besides,
   * the height and width of UV plane should be half of Y's height and width.
   * Example:
   *
   *      original_img.shape = [1, 384, 640, 3]
   *      inputY.shape = [1, 384, 640, 1]
   *      inputUV.shape = [1, 192, 320, 2]
   *      outputY.shape = [1, 384, 640, 1]
   *      outputUV.shape = [1, 192, 320, 2]
   *
   * Supported tensor {@link OperandCode}:
   * * {@link NEURON_EXT_TENSOR_RAW} (for inputY, inputUV)
   * * {@link NEURON_TENSOR_QUANT8_ASYMM} (for outputY, outputUV)
   * * {@link NEURON_TENSOR_QUANT16_ASYMM} (for outputY, outputUV)
   * Note:
   * If image mode is YUV420_8BITS, use NEURON_TENSOR_QUANT8_ASYMM; if mode is
   * YUV420_10BITS, use NEURON_TENSOR_QUANT16_ASYMM.
   *
   * Tensor rank: both input and output require rank 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: inputY, a 4-D {@link NEURON_EXT_TENSOR_RAW} tensor.
   * * 1: inputUV, a 4-D {@link NEURON_EXT_TENSOR_RAW} tensor.
   * * 2: YHeaderAlignment, an {@link NEURON_INT32} scalar, specifying
   * the header alignment in Hyfbc format.
   * * 3: UVHeaderAlignment, an {@link NEURON_INT32} scalar, specifying
   * the header alignment in Hyfbc format.
   * * 4: xAlign, an {@link NEURON_INT32} scalar, specifying the frame
   * width alignment of video decoder.
   * * 5: yAlign, an {@link NEURON_INT32} scalar, specifying the frame
   * height alignment of video decoder.
   * * 6: xOffset, an {@link NEURON_INT32} scalar, specifying the frame
   * width offset of video decoder.
   * * 7: yOffset, an {@link NEURON_INT32} scalar, specifying the frame
   * height offset of video decoder.
   * * 8: mode, an {@link NEURON_INT32} scalar. Set to 0 for
   * YUV420_8BITS. Set to 1 for YUV420_10BITS. Note that 8b, 10b here means the
   * compressed bit width in Hyfbc frame, where the decompressed YUV420 is 8b
   * for Hyfbc_8b, and YUV420 is 16b for Hyfbc_10b.
   * * 9: outPitchN, an {@link NEURON_INT32} scalar, specifying the
   * YUV420 N-axis pitch. Must be set to 1, because only a single batch is
   * supported for HyfbcDecompress.
   * * 10: outPitchH, an {@link NEURON_INT32} scalar, specifying the
   * YUV420 H-axis pitch. Set to the original compressed image height with video
   * codec alignment.
   * * 11: outPitchW, an {@link NEURON_INT32} scalar, specifying the
   * YUV420 W-axis pitch. Set to the original compressed image width with video
   * codec alignment.
   * * 12: outPitchC, an {@link NEURON_INT32} scalar, specifying the
   * YUV420 C-axis pitch. Set to 1 for interleaved YUV420.
   *
   * Outputs:
   * * 0: output Y, a 4-D tensor. Tensor type can be either {@link
   * NEURON_TENSOR_QUANT8_ASYMM} or {@link
   * NEURON_TENSOR_QUANT16_ASYMM}, depends on YUV420 bit mode.
   * * 1: output UV, a 4-D tensor. Tensor type can be either {@link
   * NEURON_TENSOR_QUANT8_ASYMM} or {@link
   * NEURON_TENSOR_QUANT16_ASYMM}, depends on YUV420 bit mode.
   *
   * Available since NeuroPilot 7.0.0.
   */
  NEURON_HYFBCTOYUV420 = 106,
  /**
   * Compress YUV420 to AFBC frame, support both YUV420_8BITS and
   * YUV420_10BITS formats. AFBC (Arm Frame Buffer Compression) is a lossless
   * compressed image format, created by ARM to reduce the size of images.
   *
   * For input part, need to set two inputs with different shape, representing Y
   * and UV plane respectively. For output part, need to set one output for
   * AFBC.
   *
   * The shape of the two inputs (inputY, inputUV) and output (AFBC)
   * depends on the original images' shape ([batches, height, width, channels]).
   * Both height and width shold follow 64 alignment rule. For example, if
   * original height is 480, its 64 alignment should be 512. For Y plane,
   * channel size should be 1; for UV plane, channel size should be 2. Besides,
   * the height and width of UV plane should be half of Y's height and width.
   * For AFBC output, its height shoud be 3/2 of Y's height, and its width
   * equals to Y's width. Example:
   *
   *      original_img.shape = [1, 384, 640, 3]
   *      inputY.shape = [1, 384, 640, 1]
   *      inputUV.shape = [1, 192, 320, 2]
   *      output.shape = [1, 576, 640, 1]
   *
   * Supported tensor {@link OperandCode}:
   * * {@link NEURON_EXT_TENSOR_RAW} (for output)
   * * {@link NEURON_TENSOR_QUANT8_ASYMM} (for inputY, inputUV)
   * * {@link NEURON_TENSOR_QUANT16_ASYMM} (for inputY, inputUV)
   * Note:
   * If image mode is YUV420_8BITS, use NEURON_TENSOR_QUANT8_ASYMM; if mode is
   * YUV420_10BITS, use NEURON_TENSOR_QUANT16_ASYMM.
   *
   * Tensor rank: both input and output require rank 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: inputY, a 4-D tensor. Tensor type can be either {@link
   * NEURON_TENSOR_QUANT8_ASYMM} or {@link
   * NEURON_TENSOR_QUANT16_ASYMM}, depends on YUV420 bit mode.
   * * 1: inputUV, a 4-D tensor. Tensor type can be either {@link
   * NEURON_TENSOR_QUANT8_ASYMM} or {@link
   * NEURON_TENSOR_QUANT16_ASYMM}, depends on YUV420 bit mode.
   * * 2: HeaderAlignment, an {@link NEURON_INT32} scalar, specifying
   * the header alignment in AFBC format.
   * * 3: xAlign, an {@link NEURON_INT32} scalar, specifying the frame
   * width alignment of AFBC format.
   * * 4: yAlign, an {@link NEURON_INT32} scalar, specifying the frame
   * height alignment of AFBC format.
   * * 5: xOffset, an {@link NEURON_INT32} scalar, specifying the frame
   * width offset of AFBC format.
   * * 6: yOffset, an {@link NEURON_INT32} scalar, specifying the frame
   * height offset of AFBC format.
   * * 7: mode, an {@link NEURON_INT32} scalar. Set to 0 for
   * YUV420_8BITS. Set to 1 for YUV420_10BITS. Note that 8b, 10b here means the
   * compressed bit width in AFBC frame, where the YUV420 must be 8b for
   * AFBC_8b, and must be 16b for AFBC_10b.
   * * 8: inPitchN, an {@link NEURON_INT32} scalar, specifying the
   * YUV420 N-axis pitch. Must be set to 1, because only a single batch is
   * supported for AfbcCompress.
   * * 9: inPitchH, an {@link NEURON_INT32} scalar, specifying the
   * YUV420 H-axis pitch. Set to the expected compressed image height.
   * * 10: inPitchW, an {@link NEURON_INT32} scalar, specifying the
   * YUV420 W-axis pitch. Set to the expected compressed image height.
   * * 11: inPitchC, an {@link NEURON_INT32} scalar, specifying the
   * YUV420 C-axis pitch. Set to 1 for interleaved YUV420.
   *
   * Outputs:
   * * 0: output, a 4-D {@link NEURON_EXT_TENSOR_RAW} tensor.
   *
   * Available since NeuroPilot 7.0.0.
   */
  NEURON_YUV420TOAFBC = 107,
  NEURON_NUMBER_OF_OPERATIONS,
} NeuronOperationType;

/**
 * Fused activation function types.
 */
typedef enum {
  // NO fused activation function.
  NEURON_FUSED_NONE = 0,
  // Fused ReLU activation function.
  NEURON_FUSED_RELU = 1,
  // Fused ReLU1 activation function.
  NEURON_FUSED_RELU1 = 2,
  // Fused ReLU6 activation function.
  NEURON_FUSED_RELU6 = 3,
} NeuronAdapterFuseCode;

/**
 * Implicit padding algorithms.
 */
typedef enum {
  /**
   * SAME padding.
   * Padding on both ends are the "same":
   *     padding_to_beginning =  total_padding / 2
   *     padding_to_end       = (total_padding + 1)/2.
   * i.e., for even number of padding, padding to both ends are exactly
   * the same; for odd number of padding, padding to the ending is bigger
   * than the padding to the beginning by 1.
   *
   * total_padding is a function of input, stride and filter size.
   * It could be computed as follows:
   *    out_size = (input + stride - 1) / stride;
   *    needed_input = (out_size - 1) * stride + filter_size
   *    total_padding = max(0, needed_input - input_size)
   *  The computation is the same for the horizontal and vertical directions.
   */
  NEURON_PADDING_SAME = 1,

  /**
   * VALID padding.
   * No padding. When the input size is not evenly divisible by
   * the filter size, the input at the end that could not fill
   * the whole filter tile will simply be ignored.
   */
  NEURON_PADDING_VALID = 2,
} NeuronAdapterPaddingCode;

/**
 * Execution preferences.
 */
typedef enum {
  /* Prefer executing in a way that minimizes battery drain. */
  NEURON_PREFER_LOW_POWER = 0,
  /* Prefer executing as fast as possible. (more power consumption)*/
  NEURON_PREFER_FAST_SINGLE_ANSWER = 1,
  /* Prefer maximizing the throughput of successive frames */
  NEURON_PREFER_SUSTAINED_SPEED = 2,
  /* Prefer executing with turbo boost. (most power consumption) */
  NEURON_PREFER_TURBO_BOOST = 3,
} NeuronAdapterPreferenceCode;

/**
 * Relative execution priority.
 */
typedef enum {
  NEURON_PRIORITY_LOW = 90,
  NEURON_PRIORITY_MEDIUM = 100,
  NEURON_PRIORITY_HIGH = 110,
  NEURON_PRIORITY_DEFAULT = NEURON_PRIORITY_MEDIUM,
} NeuronAdapterPriorityCode;

/**
 * Compiler optimization hint.
 */
typedef enum {
  /**
   * Normal optimization.
   * Available since 4.3.1
   */
  NEURON_OPTIMIZATION_NORMAL = 0,
  /**
   * Reduce latency by utilizing as many APU cores as possible.
   * Available since 4.3.1
   */
  NEURON_OPTIMIZATION_LOW_LATENCY = 1 << 0,
  /**
   * Reducing DRAM access as more as possible.
   * Available since 4.4.0
   */
  NEURON_OPTIMIZATION_DEEP_FUSION = 1 << 1,
  /**
   * Reduce latency by using as many APU cores as possible in batch-dimension.
   * (For models with batch > 1)
   * Available since 4.4.0
   */
  NEURON_OPTIMIZATION_BATCH_PROCESSING = 1 << 2,
  /**
   * Default optimization setting.
   * Available since 4.3.1
   */
  NEURON_OPTIMIZATION_DEFAULT = NEURON_OPTIMIZATION_NORMAL,
} OptimizationCode;

/**
 * CPU cache flush hint.
 */
typedef enum {
  /**
   * Sync input buffer and invalidate output buffer.
   * Available since 5.0.1
   */
  NEURON_CACHE_FLUSH_ENABLE_ALL = 0,
  /**
   * Disable sync input buffer.
   * Available since 5.0.1
   */
  NEURON_CACHE_FLUSH_DISABLE_SYNC_INPUT = 1 << 0,
  /**
   * Disable invalidate output buffer.
   * Available since 5.0.1
   */
  NEURON_CACHE_FLUSH_DISABLE_INVALIDATE_OUTPUT = 1 << 1,
  /**
   * Default cache flush setting.
   * Available since 5.0.1
   */
  NEURON_CACHE_FLUSH_DEFAULT = NEURON_CACHE_FLUSH_ENABLE_ALL,
} CacheFlushCode;

/**
 * Compilation Type.
 */
typedef enum {
  /* Normal Compilation Available since 7.0.0 */
  COMPILATION_TYPE_NORMAL = 0,
  /* @deprecate */
  COMPILATION_TYPE_DEBUG_PLUS = 1,
  /* Batched Execution: Set input/output from memory every time.
   * Available since 7.0.0
   */
  COMPILATION_TYPE_BATCHED = 2,
  /* One compilation with multi-executions could be created.
   * Available since 7.0.0
   */
  COMPILATION_TYPE_MULTI_EXECUTIONS = 3,
  /* Batched Execution: Set input/output from memory 1st time and memcpy next
   * time. Available since 7.0.1
   */
  COMPILATION_TYPE_EXECUTION_CONTROLLER = 4,
} CompilationType;

/**
 * Supported Feature
 */
typedef enum {
  NEURON_FEATURE_NONE = 0,
  NEURON_THROUGHPUT_MODE = 1,
} NeuronFeatureType;

/**
 * The structure to represent the neuron version.
 */
typedef struct {
  uint8_t major; ///< major version
  uint8_t minor; ///< minor version
  uint8_t patch; ///< patch version
} NeuronRuntimeVersion;

/**
 * Get the version of Neuron runtime library.
 *
 * @param version the version of Neuron runtime library.
 * @return NEURON_NO_ERROR
 */
int Neuron_getVersion(NeuronRuntimeVersion* version);

/**
 * Get the supported status of feature.
 *
 * Available since 7.0.0
 *
 * @param type input feature @NeuronFeatureType to check supported or not
 * @param supported return the supported status
 * @return NEURON_NO_ERROR if successful.
 */
int Neuron_getFeatureSupportedStatus(NeuronFeatureType type, bool* supported);

/**
 * Get the size of L1 memory in APU.
 *
 * Available since 4.3.0
 *
 * @param sizeKb L1 memory size in KB
 * @return NEURON_NO_ERROR if successful.
 */
int Neuron_getL1MemorySizeKb(uint32_t* sizeKb);

/**
 * Creates a shared memory object from a file descriptor.
 *
 * For ion descriptor, application should create the ion memory and descriptor
 * first and then use it in this function.
 *
 * Available since 4.1.0 Only supports ion fd.
 *
 * @param size The requested size in bytes. Must not be larger than the file
 * size.
 * @protect The desired memory protection for the mapping. It is either
 * PROT_NONE or the bitwise OR of one or more of the following flags: PROT_READ,
 * PROT_WRITE.
 * @fd The requested file descriptor. The file descriptor has to be mmap-able.
 * @offset The offset to the beginning of the file of the area to map.
 * @memory The memory object to be created. Set to NULL if unsuccessful.
 */
int NeuronMemory_createFromFd(
    size_t size,
    int protect,
    int fd,
    size_t offset,
    NeuronMemory** memory);

#ifdef __ANDROID__
/**
 * Creates a shared memory object from an AHardwareBuffer handle.
 *
 * We only support AHardwareBuffer with format AHARDWAREBUFFER_FORMAT_BLOB and
 * it can only be used for Model inputs and outputs.
 *
 * The AHardwareBuffer with AHARDWAREBUFFER_FORMAT_BLOB format can be used the
 * same way as shared memory created from a file handle. See NeuronMemory for
 * description on how to use this shared memory.
 *
 * The provided AHardwareBuffer must outlive the NeuronMemory object.
 *
 * Available since 5.0.0
 *
 * @param ahwb The AHardwareBuffer handle.
 * @param memory The memory object to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return NEURON_NO_ERROR if the request completed normally.
 *
 */
int NeuronMemory_createFromAHardwareBuffer(
    const AHardwareBuffer* ahwb,
    NeuronMemory** memory);

#else // __ANDROID__

/**
 * Not supported at non-android platform
 *
 * @return NEURON_BAD_STATE
 */
int NeuronMemory_createFromAHardwareBuffer();

#endif

/**
 * Delete a memory object.
 *
 * For ion memory, this function cleans up the internal resource associated with
 * this memory. Applications should clean up the allocated ion memory after this
 * function.
 *
 * Available since 4.1.0
 */
void NeuronMemory_free(NeuronMemory* memory);

/**
 * Create an empty NeuronModel. The model should be constructed with calls to
 * NeuronModel_addOperation and NeuronModel_addOperand.
 *
 * Available since 4.1.0
 *
 * @param model The NeuronModel to be created. Set to NULL if unsuccessful.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_create(NeuronModel** model);

/**
 * Destroy a model. The model need not have been finished by a call to
 * NeuronModel_finish.
 *
 * Available since 4.1.0
 *
 * @param model The model to be destroyed.
 */
void NeuronModel_free(NeuronModel* model);

/**
 * Indicate that we have finished modifying a model. Required before calling
 * NeuronCompilation_compile.
 *
 * Available since 4.1.0
 *
 * @param model The model to be finished.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_finish(NeuronModel* model);

/**
 * Add an operand to a model. The order in which the operands are added is
 * important. The first one added to a model will have the index value 0, the
 * second 1, etc. These indexes are used as operand identifiers in
 * NeuronModel_addOperation.
 *
 * Available since 4.1.0
 *
 * @param model The model to be modified.
 * @param type The NeuronOperandType that describes the shape of the operand.
 * Neither the NeuronOperandType nor the dimensions it points to need to outlive
 * the call to NeuronModel_addOperand.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_addOperand(NeuronModel* model, const NeuronOperandType* type);

/**
 * Sets an operand to a constant value.
 * Values of length smaller or equal to
 * NEURON_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES are immediately copied into the
 * model. For values of length greater than
 * NEURON_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES, a pointer to the buffer is
 * stored within the model. The application must not change the content of this
 * region until all executions using this model have completed. As the data may
 * be copied during processing, modifying the data after this call yields
 * undefined results.
 *
 * Attempting to modify a model once NeuronModel_finish has been called will
 * return an error.
 *
 * A special notice on the buffer lifetime when the length is greater than
 * NEURON_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES. The provided buffer must
 * outlive the compilation of this model. I.e. user must keep the buffer
 * unchanged until NeuronCompilation_finish of this model. This is an internal
 * optimization comparing to NNAPI. In NNAPI, NN runtime will copy the buffer to
 * a shared memory between NN runtime and NNAPI HIDL service during
 * ANNModel_finish, and it will be copied again to the compiled result during
 * ANNCompilation_finish. In Neuron Adapter, there will be only one copying
 * during NeuronCompilaiton_finish, so it is required to keep the buffer alive
 * until NeuronCompilaiton_finish returned.
 *
 * Available since 4.1.0
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param length The size in bytes of the data value.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_setOperandValue(
    NeuronModel* model,
    int32_t index,
    const void* buffer,
    size_t length);
/**
 * Sets an operand to a value that is a reference to another NeuronModel.
 *
 * The referenced model must already have been finished by a call to
 * NeuronModel_finish.
 *
 * The NeuronModel_relaxComputationFloat32toFloat16 setting of referenced models
 * is overridden by that setting of the main model of a compilation.
 *
 * The referenced model must outlive the model referring to it.
 *
 * Attempting to modify a model once NeuronModel_finish has been called will
 * return an error.
 *
 * Available since 4.1.0
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param value The model to be referenced.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_setOperandValueFromModel(
    NeuronModel* model,
    int32_t index,
    const NeuronModel* value);

/**
 * Sets an operand's per channel quantization parameters
 * Sets parameters required by a tensor of type
 * NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL This function must be called for every
 * tensor of type NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL before calling
 * NeuronModel_finish
 *
 * Available since 4.1.0
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param channelQuant The per channel quantization parameters for the operand.
 * No memory in this struct needs to outlive the call to this function.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_setOperandSymmPerChannelQuantParams(
    NeuronModel* model,
    int32_t index,
    const NeuronSymmPerChannelQuantParams* channelQuant);

/**
 * Sets an operand's per channel quantization parameters
 * Sets parameters required by a tensor of type
 * NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL or
 * NEURON_TENSOR_QUANT8_ASYMM_PER_CHANNEL.
 * This function must be called for every tensor of type
 * NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL or
 * NEURON_TENSOR_QUANT8_ASYMM_PER_CHANNEL before calling NeuronModel_finish.
 *
 * Available since 6.0.0
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param channelQuant The per channel quantization parameters(include
 * per-channel offset) for the operand. No memory in this struct needs to
 * outlive the call to this function.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_setOperandPerChannelQuantParams(
    NeuronModel* model,
    int32_t index,
    const NeuronPerChannelQuantParams* channelQuant);

/**
 * Add an operation to a model.
 * The operands specified by inputs and outputs must have been previously added
 * by calls to NeuronModel_addOperand.
 *
 * Available since 4.1.0
 *
 * @param model The model to be modified.
 * @param type The NeuronOperationType of the operation.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying each operand.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying each operand.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_addOperation(
    NeuronModel* model,
    NeuronOperationType type,
    uint32_t inputCount,
    const uint32_t* inputs,
    uint32_t outputCount,
    const uint32_t* outputs);

/**
 * Add an operation extension to a model.
 * The operands specified by inputs and outputs must have been previously added
 * by calls to NeuronModel_addOperand. User needs to specify the operation
 * extension name and the desired device which will execute the operation
 * extension.
 *
 * Available since 4.1.0
 *
 * @param model The model to be modified.
 * @param name The name of the operation extension.
 * @param vendor The name of the vendor which will implement the operation
 * extension.
 * @param device The device which will execute the operation extension.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying each operand.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying each operand.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_addOperationExtension(
    NeuronModel* model,
    const char* name,
    const char* vendor,
    const NeuronDevice* device,
    uint32_t inputCount,
    const uint32_t* inputs,
    uint32_t outputCount,
    const uint32_t* outputs);

/**
 * Specfifies which operands will be the model's inputs and outputs.
 * An operand cannot be used for both input and output. Doing so will return an
 * error.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to NeuronModel_addOperand.
 *
 * Attempting to modify a model once NeuronModel_finish has been
 * called will return an error.
 *
 * Available since 4.1.0
 *
 * @param model The model to be modified.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying the input operands.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying the output operands.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_identifyInputsAndOutputs(
    NeuronModel* model,
    uint32_t inputCount,
    const uint32_t* inputs,
    uint32_t outputCount,
    const uint32_t* outputs);

/**
 * Gets the supported operations in a model.
 * This function must be called after calling NeuronModel_finish
 *
 * Available since 4.1.0
 *
 * @param model The model to be queried.
 * @param supported The boolean array to be filled. True means supported. The
 * size of the boolean array must be at least as large as the number of
 * operations in the model. The order of elements in the supported array matches
 * the order in which the corresponding operations were added to the model.
 * @param operationCount number of operations in the model
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_getSupportedOperations(
    NeuronModel* model,
    bool* supported,
    uint32_t operationCount);

/**
 * Get the supported operations for a specified set of devices.
 * If multiple devices are selected, the supported operation list is a union of
 * supported operations of all selected devices.
 *
 * Available since 4.1.0
 *
 * @param model The model to be queried.
 * @param devices Selected devices
 * @param numDevices Number of selected devices
 * @param supportedOps The boolean array to be filled. True means supported. The
 * size of the boolean array must be as least as large as the number of
 * operations in the model. The order of elements in the supportedOps array
 * matches the order in which the corresponding operations were added to the
 * model.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_getSupportedOperationsForDevices(
    const NeuronModel* model,
    const NeuronDevice* const* devices,
    uint32_t numDevices,
    bool* supportedOps);

/**
 * Specifies whether NEURON_TENSOR_FLOAT32 is allowed to be calculated with
 * range and/or precision as low as that of the IEEE 754 16-bit floating-point
 * format. By default, NEURON_TENSOR_FLOAT32 must be calculated using at least
 * the range and precision of the IEEE 754 32-bit floating-point format.
 *
 * Available since 4.1.0
 *
 * @param model The model to be modified.
 * @param allow 'true' indicates NEURON_TENSOR_FLOAT32 may be calculated with
 * range and/or precision as low as that of the IEEE 754 16-bit floating point
 * format. 'false' indicates NEURON_TENSOR_FLOAT32 must be calculated using at
 * least the range and precision of the IEEE 754 32-bit floating point format.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_relaxComputationFloat32toFloat16(
    NeuronModel* model,
    bool allow);

/**
 * Hint compiler to suppress the input data conversion, the users have to
 * convert the input data into platform-expected format before inference.
 *
 * Available since 4.2.0
 *
 * @param model The model to be modified.
 * @param suppress True to suppress the input data conversion.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_suppressInputConversion(NeuronModel* model, bool suppress);

/**
 * Hint compiler to suppress the output data conversion, the users have to
 * convert the output data from platform-generated format before inference.
 *
 * Available since 4.2.0
 *
 * @param model The model to be modified.
 * @param suppress True to suppress the output data conversion.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_suppressOutputConversion(NeuronModel* model, bool suppress);

/**
 * Restore the compiled network using user provided buffer.
 *
 * The restored NeuronCompilaton could be used in creating executing instance.
 * The restored NeuronModel cannot be recompiled.
 *
 * Available since 4.3.0
 *
 * @param model Restored model.
 * @param compilation Restored compilation
 * @param buffer User provided buffer to restore the compiled network.
 * @param size Size of the user provided buffer in bytes.
 * @return NEURON_NO_ERROR if compiled network is successfully copied to the
 * user allocated buffer. NEURON_BAD_DATA if it fails to load the compiled
 * network, this could either be the version is not matched or the data is
 * corrupted.
 */
int NeuronModel_restoreFromCompiledNetwork(
    NeuronModel** model,
    NeuronCompilation** compilation,
    const void* buffer,
    const size_t size);

/**
 * Restore the compiled network using user provided buffer.
 * Support multiple compilation type; choices are: COMPILATION_TYPE_BATCHED,
 * COMPILATION_TYPE_EXECUTION_CONTROLLER, COMPILATION_TYPE_EXECUTION_CONTROLLER,
 * and COMPILATION_TYPE_NORMAL.
 *
 * There are two ways to use Batched Compilation:
 * 1) load from DLA.
 * 2) create batched compilation directly.
 * To load DLA, one should call NeuronCompilation_create and
 * NeuronModel_restoreFromCompiledNetworkV2. To create directly, one should call
 * NeuronCompilation_createForBatch.
 *
 * The restored NeuronCompilaton could be used in creating executing instance.
 * The restored NeuronModel cannot be recompiled.
 *
 * Available since 7.0.0
 *
 * @param model Restored model.
 * @param compilation Restored compilation
 * @param buffer User provided buffer to restore the compiled network.
 * @param size Size of the user provided buffer in bytes.
 * @param type Type of the compilation needed to be restored.
 * @return NEURON_NO_ERROR if compiled network is successfully copied to the
 * user allocated buffer. NEURON_BAD_DATA if it fails to load the compiled
 * network, this could either be the version is not matched or the data is
 * corrupted.
 */
int NeuronModel_restoreFromCompiledNetworkV2(
    NeuronModel** model,
    NeuronCompilation** compilation,
    const void* buffer,
    const size_t size,
    const CompilationType& type);

/**
 * Set a string into model that can be used for recognition for user.
 * It's only used for debug, the string can be dumped into log and make users
 * check the model behavior easily.
 *
 * Available since 7.0.0
 *
 * @param model The model to be modified.
 * @param name The string, user can free buffer 'name' after calling this API.
 * @return NEURON_NO_ERROR if the string is set success. NEURON_UNEXPECTED_NULL
 * if the input param is nullptr.
 */
int NeuronModel_setName(NeuronModel* model, const char* name);

/**
 * Create a NeuronCompilation to compile the given model.
 *
 * This function only creates the object. Compilation is only performed once
 * NeuronCompilation_finish is invoked. NeuronCompilation_finish should be
 * called once all desired properties have been set on the compilation.
 * NeuronModel_free should be called once the compilation is no longer needed.
 * The provided model must outlive the compilation. The model must already have
 * been finished by a call to NeuronModel_finish.
 *
 * Available since 4.1.0
 *
 * @param model The NeuronModel to be compiled.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return NEURON_NO_ERROR if successful
 */
int NeuronCompilation_create(
    NeuronModel* model,
    NeuronCompilation** compilation);

/**
 * Create a NeuronCompilation with different purpose to compile the given model.
 *
 * This function only creates the object. Compilation is only performed once
 * NeuronCompilation_finish is invoked. NeuronCompilation_finish should be
 * called once all desired properties have been set on the compilation.
 * NeuronModel_free should be called once the compilation is no longer needed.
 * The provided model must outlive the compilation. The model must already have
 * been finished by a call to NeuronModel_finish.
 *
 * Available since 7.0.1
 *
 * @param model The NeuronModel to be compiled.
 * @param type Type of the compilation needed to be created.
 * @param options The options which used to create with compilation.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return NEURON_NO_ERROR if successful
 */
int NeuronCompilation_createV2(
    NeuronModel* model,
    CompilationType type,
    const char* options,
    NeuronCompilation** compilation);

/**
 * Destroy a compilation.
 *
 * Available since 4.1.0
 *
 * @param compilation The compilation to be destroyed.
 */
void NeuronCompilation_free(NeuronCompilation* compilation);

/**
 * Compilation is finished once NeuronCompilation_finish is invoked. Required
 * before calling NeuronExecution_create. This function must only be called once
 * for a given compilation.
 *
 * Available since 4.1.0
 *
 * @param compilation The compilation to be finished.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_finish(NeuronCompilation* compilation);

/**
 * Gets the supported operations in a model with specific optimized configures.
 * This function must be called before calling NeuronCompilation_finish.
 *
 * Available since 7.0.0
 *
 * @param compilation The compilation to be queried.
 * @param operationCount number of operations in the model
 * @param supported The boolean array to be filled. True means supported. The
 * size of the boolean array must be at least as large as the number of
 * operations in the model. The order of elements in the supported array matches
 * the order in which the corresponding operations were added to the model.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_getSupportedOperations(
    NeuronCompilation* compilation,
    uint32_t operationCount,
    bool* supported);

/**
 * Provides optional caching information for faster re-compilation.
 *
 * Available since 4.1.0
 *
 * @param compilation The compilation to be cached.
 * @param cacheDir The cache directory for storing and retrieving caching data.
 * The user should choose a directory local to the application, and is
 * responsible for managing the cache entries.
 * @param token The token provided by the user to specify a model must be of
 * length NEURON_BYTE_SIZE_OF_CACHE_TOKEN. The user should ensure that the token
 * is unique to a model within the application. Neuron cannot detect token
 * collisions; a collision will result in a failed execution or in a successful
 * execution that produces incorrect output values.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_setCaching(
    NeuronCompilation* compilation,
    const char* cacheDir,
    const uint8_t* token);

/**
 * Hint compiler with the size of L1 memory, this value should not be larger
 * than real platform's settings. The user can get the platform's L1 memory size
 * in KB by calling Neuron_getL1MemorySizeKb.
 *
 * Available since 4.3.0
 *
 * @param compilation The compilation to be modified.
 * @param sizeKb L1 memory size in KB.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_setL1MemorySizeKb(
    NeuronCompilation* compilation,
    uint32_t sizeKb);

/**
 * Create a NeuronCompilation to compile the given model for a specified set of
 * devices. The user must handle all compilation and execution failures from the
 * specified set of devices. This is in contrast to a use of
 * NeuronCompilation_create, where neuron will attempt to recover from such
 * failures.
 *
 * Available since 4.1.0
 *
 * @param model The NeuronModel to be compiled.
 * @param devices The set of devices. Must not contain duplicates.
 * @param numDevices The number of devices in the set.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return NEURON_NO_ERROR if successful, NEURON_BAD_DATA if the model is
 * invalid.
 */
int NeuronCompilation_createForDevices(
    NeuronModel* model,
    const NeuronDevice* const* devices,
    uint32_t numDevices,
    NeuronCompilation** compilation);

/**
 * Create a NeuronCompilation. Which can divide one graph into several subgraph
 * and use the information to debug.
 *
 * Only be used in debug purpose, no guarantees performance and thread safe.
 *
 * Available since 5.0.0
 *
 * @param model The NeuronModel to be compiled.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return NEURON_NO_ERROR if successful, NEURON_BAD_DATA if the model is
 * invalid.
 */
int NeuronCompilation_createForDebug(
    NeuronModel* model,
    NeuronCompilation** compilation);

/**
 * Sets the execution preference associated with this compilation.
 *
 * Default value of preference is PREFER_SINGLE_FAST_ANSWER
 *
 * Available since 4.1.0
 *
 * @param compilation The compilation to be modified.
 * @param preference Either NEURON_PREFER_LOW_POWER,
 * NEURON_PREFER_SINGLE_FAST_ANSWER, or NEURON_PREFER_SUSTAINED_SPEED.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_setPreference(
    NeuronCompilation* compilation,
    int32_t preference);

/**
 * Sets the execution priority associated with this compilation.
 *
 * Execution priorities are relative to other executions created by the same
 * application (specifically same uid) for the same device. Specifically,
 * priorities of executions from one application will not affect executions from
 * another application.
 *
 * Higher priority executions may use more compute resources than lower priority
 * executions, and may preempt or starve lower priority executions.
 *
 * Available since 4.1.0
 *
 * @param compilation The compilation to be modified.
 * @param priority The relative priority of the execution compared to other
 * executions created by the application. Must be one of NEURON_PRIORITY_*.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_setPriority(NeuronCompilation* compilation, int priority);

/**
 * Get the padded dimensional information of the specified input operand of the
 * compilation. This function must be called after calling
 * NeuronCompilation_finish. If NeuronModel_suppressInputConversion was not
 * applied to the model to be compiled, the returned dimensions are the padded
 * dimension after NeuronCompilation_finish to satisfy the optimization
 * requirement from the underlying hardware accelerators.
 * If NeuronModel_suppressInputConversion was applied to the model to be
 * compiled, the returned dimensions are the same as the original dimensions
 * given from user.
 *
 * Available since 4.2.0
 *
 * @param compilation The compilation to be queried.
 * @param index The index of the input operand we are querying. It is an index
 * into the lists passed to NeuronModel_identifyInputsAndOutputs. It is not the
 * index associated with NeuronModel_addOperand.
 * @param dimensions The dimension array to be filled. The size of the array
 * must be exactly as large as the rank of the input operand to be queried in
 * the model.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_getInputPaddedDimensions(
    NeuronCompilation* compilation,
    int32_t index,
    uint32_t* dimensions);

/**
 * Get the padded dimensional information of the specified output operand of the
 * compilation. This function must be called after calling
 * NeuronCompilation_finish. If NeuronModel_suppressOutputConversion was not
 * applied to the model to be compiled, the returned dimensions are the padded
 * dimension after NeuronCompilation_finish to satisfy the optimization
 * requirement from the underlying hardware accelerators.
 * If NeuronModel_suppressOutputConversion was applied to the model to be
 * compiled, the returned dimensions are the same as the original dimensions
 * given from user.
 *
 * Available since 4.2.0
 *
 * @param compilation The compilation to be queried.
 * @param index The index of the output operand we are querying. It is an index
 * into the lists passed to NeuronModel_identifyInputsAndOutputs. It is not the
 * index associated with NeuronModel_addOperand.
 * @param dimensions The dimension array to be filled. The size of the array
 * must be exactly as large as the rank of the output operand to be queried in
 * the model.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_getOutputPaddedDimensions(
    NeuronCompilation* compilation,
    int32_t index,
    uint32_t* dimensions);

/**
 * Get the expected buffer size (bytes) of the specified input operand of the
 * compilation. If NeuronModel_suppressInputConversion was not applied to the
 * model to be compiled, the returned size are the padded size after
 * NeuronCompilation_finish to satisfy the optimization requirement from the
 * underlying hardware accelerators. If NeuronModel_suppressInputConversion was
 * applied to the model to be compiled, the returned size are the same as the
 * original size given from user.
 *
 * Available since 4.2.0
 *
 * @param compilation The compilation to be queried.
 * @param index The index of the input operand we are querying. It is an index
 * into the lists passed to NeuronModel_identifyInputsAndOutputs. It is not the
 * index associated with NeuronModel_addOperand.
 * @param size the expected buffer size in bytes.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_getInputPaddedSize(
    NeuronCompilation* compilation,
    int32_t index,
    size_t* size);

/**
 * Get the expected buffer size (bytes) of the specified output operand of the
 * compilation. If NeuronModel_suppressOutputConversion was not applied to the
 * model to be compiled, the returned size are the padded size after
 * NeuronCompilation_finish to satisfy the optimization requirement from the
 * underlying hardware accelerators. If NeuronModel_suppressOutputConversion was
 * applied to the model to be compiled, the returned size are the same as the
 * original size given from user.
 *
 * Available since 4.2.0
 *
 * @param compilation The compilation to be queried.
 * @param index The index of the output operand we are querying. It is an index
 * into the lists passed to NeuronModel_identifyInputsAndOutputs. It is not the
 * index associated with NeuronModel_addOperand.
 * @param size the expected buffer size in bytes.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_getOutputPaddedSize(
    NeuronCompilation* compilation,
    int32_t index,
    size_t* size);

/**
 * Get the compiled network size of the compilation.
 *
 * This must be called after NeuronCompilation_finished and before
 * NeuronExecution_create. It is not allowed to call this with a compilation
 * restored from cache.
 *
 * Available since 4.3.0
 *
 * @param compilation The compilation to be queried.
 * @param size The compiled network size in bytes.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_getCompiledNetworkSize(
    NeuronCompilation* compilation,
    size_t* size);

/**
 * Store the compiled network.
 *
 * Users have to allocate the buffer with the specified size before calling this
 * function.
 *
 * This must be called after NeuronCompilation_finished and before
 * NeuronExecution_create. It is not allowed to call this with a compilation
 * restored from cache.
 *
 * Available since 4.3.0
 *
 * @param compilation The compilation to be queried.
 * @param buffer User allocated buffer to store the compiled network.
 * @param size Size of the user allocated buffer in bytes.
 * @return NEURON_NO_ERROR if compiled network is successfully copied to the
 * user allocated buffer.
 */
int NeuronCompilation_storeCompiledNetwork(
    NeuronCompilation* compilation,
    void* buffer,
    const size_t size);
/**
 * Hint the compiler to apply the optimization strategy according to the user
 * specified parameters.
 *
 * Available since 4.3.0
 *
 * @param compilation The compilation to be modified.
 * @param optimizationCode User specified optimization strategy. Must be one of
 * NEURON_OPTIMIZATION_* or the inclusive OR value of multiple
 * NEURON_OPTIMIZATION_*.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_setOptimizationHint(
    NeuronCompilation* compilation,
    uint32_t optimizationCode);

/**
 * Hint the compiler to apply the optimization strategy according to the user
 * specified arguments in a null-terminated string.
 *
 * Available since 4.6.0
 *
 * @param compilation The compilation to be modified.
 * @param optimizationString A null-terminated string to represent the user
 * specified optimization strategy.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_setOptimizationString(
    NeuronCompilation* compilation,
    const char* optimizationString);

/**
 * Only allow users' optimization string(from
 * NeuronCompilation_setOptimizationString), the system won't set any compiler
 * options for them.
 *
 * Available since 6.0.5
 *
 * @param compilation The compilation to be modified.
 * @param allow Allow only use user's setting or not.
 * strategy.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_setOnlyAllowOptimizationString(
    NeuronCompilation* compilation,
    bool allow);

/**
 * Get the compiler hints which are used to apply the optimization strategy
 * according to the user specified arguments in a null-terminated string.
 *
 * Available since 6.0.5
 *
 * @param compilation The compilation to be modified.
 * @param optimizationString A null-terminated string to represent the user
 * specified optimization strategy.
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_getOptimizationString(
    NeuronCompilation* compilation,
    const char** optimizationString);

/**
 * Hint compiler to trim the model IO alignment.
 *
 * Available since 4.4.8
 *
 * @param compilation The compilation to be modified.
 * @param enable 'true' for trimming model IO alignment.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_setTrimIOAlignment(
    NeuronCompilation* compilation,
    bool enable);

/**
 * Hint compiler to use software dilated convolution
 *
 * Available since 4.4.8
 *
 * @param compilation The compilation to be modified.
 * @param enable 'true' indicates a hint to compiler to use software dilated
 * convolution
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_setSWDilatedConv(
    NeuronCompilation* compilation,
    bool enable);

/**
 * Create a new execution instance by calling the NeuronExecution_create
 * function. The provided compilation must outlive the execution.
 *
 * Available since 4.1.0
 *
 * @param compilation The NeuronCompilation to be evaluated.
 * @param execution The newly created object or NULL if unsuccessful.
 *
 * @return NEURON_NO_ERROR if successful
 */
int NeuronExecution_create(
    NeuronCompilation* compilation,
    NeuronExecution** execution);

/**
 * Destroy an execution.
 *
 * Available since 4.1.0
 *
 * @param execution The execution to be destroyed.
 */
void NeuronExecution_free(NeuronExecution* execution);

/**
 * Associate a user buffer with an input of the model of the NeuronExecution.
 * The provided buffer must outlive the execution.
 *
 * Available since 4.1.0
 *
 * @param execution The execution to be modified.
 * @param index The index of the input argument we are setting. It is an index
 * into the lists passed to NeuronModel_identifyInputsAndOutputs. It is not the
 * index associated with NeuronModel_addOperand.
 * @param type The NeuronOperandType of the operand. Currently NeuronAdapter
 * only takes NULL.
 * @param buffer The buffer containing the data.
 * @param length The length in bytes of the buffer.
 *
 * @return NEURON_NO_ERROR if successful, NEURON_BAD_DATA if the name is not
 * recognized or the buffer is too small for the input.
 */
int NeuronExecution_setInput(
    NeuronExecution* execution,
    int32_t index,
    const NeuronOperandType* type,
    const void* buffer,
    size_t length);

/**
 * Associate a user buffer with an output of the model of the NeuronExecution.
 * The provided buffer must outlive the execution.
 *
 * Available since 4.1.0
 *
 * @param execution The execution to be modified.
 * @param index The index of the output argument we are setting. It is an index
 * into the lists passed to NeuronModel_identifyInputsAndOutputs. It is not the
 * index associated with NeuronModel_addOperand.
 * @param type The NeuronOperandType of the operand. Currently NeuronAdapter
 * only takes NULL.
 * @param buffer The buffer where the data is to be written.
 * @param length The length in bytes of the buffer.
 *
 * @return NEURON_NO_ERROR if successful, NEURON_BAD_DATA if the name is not
 * recognized or the buffer is too small for the output.
 */
int NeuronExecution_setOutput(
    NeuronExecution* execution,
    int32_t index,
    const NeuronOperandType* type,
    void* buffer,
    size_t length);

/**
 * Associate part of a memory object with an input of the model of the
 * NeuronExecution.
 *
 * The provided memory must outlive the execution and should not be changed
 * during computation.
 *
 * Available since 4.1.0
 *
 * @param execution The execution to be modified.
 * @param index The index of the input argument we are setting. It is an index
 * into the lists passed to NeuronModel_identifyInputsAndOutputs. It is not the
 * index associated with Neuronodel_addOperand.
 * @param type The NeuronOperandType of the operand. Currently NueronAdapter
 * only takes NULL.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory. The
 * offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return NEURON_NO_ERROR if successful, NEURON_BAD_DATA if the name is not
 * recognized or the buffer is too small for the input.
 */
int NeuronExecution_setInputFromMemory(
    NeuronExecution* execution,
    uint32_t index,
    const NeuronOperandType* type,
    const NeuronMemory* memory,
    size_t offset,
    size_t length);

/**
 * Associate part of a memory object with an output of the model of the
 * NeuronExecution.
 *
 * The provided memory must outlive the execution and should not be changed
 * during computation.
 *
 * Available since 4.1.0
 *
 * @param execution The execution to be modified.
 * @param index The index of the output argument we are setting. It is an index
 * into the lists passed to NeuronModel_identifyInputsAndOutputs. It is not the
 * index associated with Neuronodel_addOperand.
 * @param type The NeuronOperandType of the operand. Currently NueronAdapter
 * only takes NULL.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory. The
 * offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return NEURON_NO_ERROR if successful, NEURON_BAD_DATA if the name is not
 * recognized or the buffer is too small for the input.
 */
int NeuronExecution_setOutputFromMemory(
    NeuronExecution* execution,
    uint32_t index,
    const NeuronOperandType* type,
    const NeuronMemory* memory,
    size_t offset,
    size_t length);

/**
 * Schedule synchronous evaluation of the execution.
 * Returns once the execution has completed and the outputs are ready to be
 * consumed.
 *
 * Available since 4.1.0
 *
 * @param execution The execution to be scheduled and executed.
 *
 * @return NEURON_NO_ERROR if the execution completed normally. NEURON_BAD_STATE
 * if the inference fails. Add two return code since 5.0.0
 * (NEURON_MISSED_DEADLINE_TRANSIENT if  inference timeout, and
 * NEURON_OUTPUT_INSUFFICIENT_SIZE if given outsize is not sufficient for real
 * output)
 *
 */
int NeuronExecution_compute(NeuronExecution* execution);

/**
 * Schedule asynchronous evaluation of the execution with dependencies.
 *
 * The execution will wait for all the depending events to be signaled before
 * starting the evaluation. Once the execution has completed and the outputs
 * are ready to be consumed, the returned event will be signaled. Depending on
 * which devices are handling the execution, the event could be backed by a sync
 * fence. Use NeuronEvent_wait to wait for that event.
 *
 * NeuronEvent_wait must be called to recurperate the resources used by the
 * execution.
 *
 * If parts of the execution are scheduled on devices that do not support fenced
 * execution, the function call may wait for such parts to finish before
 * returning.
 *
 * The function will return an error if any of the events in dependencies is
 * already in a bad state. After the execution is scheduled, if any of the
 * events in dependencies does not complete normally, the execution will fail,
 * and NeuronEvent_wait on the returned event will return an error.
 *
 * The function will return an error if any of the execution outputs has a
 * tensor operand type that is not fully specified.
 *
 * @param execution The execution to be scheduled and executed.
 * @param dependencies A set of depending events. The actual evaluation will not
 * start until all the events are signaled.
 * @param num_dependencies The number of events in the dependencies set.
 * @param duration currently not used
 * @param event The event that will be signaled on completion. event is set to
 *              NULL if there's an error.
 *
 * @return NEURON_NO_ERROR if the evaluation is successfully scheduled.
 *
 * Available since 5.0.0
 */
int NeuronExecution_startComputeWithDependencies(
    NeuronExecution* execution,
    const NeuronEvent* const* dependencies,
    uint32_t num_dependencies,
    uint64_t duration,
    NeuronEvent** event);

/**
 * Set the maximum duration of WHILE loops in the specified execution.
 *
 * @param execution The execution to be modified.
 * @param duration The maximum amount of time in nanoseconds.
 * @return NEURON_NO_ERROR if successful.
 *
 * Available since 5.0.0
 */
int NeuronExecution_setLoopTimeout(
    NeuronExecution* execution,
    uint64_t duration);

/**
 * Get the default timeout value for WHILE loops.
 *
 * @return The default timeout value in nanoseconds.
 *
 * Available since 5.0.0
 */
uint64_t Neuron_getDefaultLoopTimeout();

/**
 * Get the maximum timeout value for WHILE loops.
 *
 * @return The maximum timeout value in nanoseconds.
 *
 * Available since 5.0.0
 */
uint64_t Neuron_getMaximumLoopTimeout();

/**
 * Sets the execution boost hint associated with this execution. Required before
 * calling NeuronExecution_compute.
 *
 * Execution boost is the hint for the device frequency, ranged between 0
 * (lowest) to 100 (highest). For the compilation with preference set as
 * NEURON_PREFER_SUSTAINED_SPEED, scheduler guarantees that the executing boost
 * value would equal to the boost value hint.
 *
 * On the other hand, for the compilation with preference set as
 * NEURON_PREFER_LOW_POWER, scheduler would try to save power by configuring the
 * executing boost value with some value that is not higher than the boost value
 * hint.
 *
 * Available since 4.1.0
 *
 * @param execution The execution to be modified.
 * @param boostValue The hint for the device frequency, ranged between 0
 * (lowest) to 100 (highest).
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronExecution_setBoostHint(
    NeuronExecution* execution,
    uint8_t boostValue);

/**
 * Sets the execution CPU cache flush hint associated with this execution.
 * Required before calling NeuronExecution_setInputFromMemory and
 * NeuronExecution_setOutputFromMemory.
 *
 * Default value of preference is NEURON_CACHE_FLUSH_ENABLE_ALL
 *
 * Available since 5.0.1
 *
 * @param execution The execution to be modified.
 * @param hint  It is either NEURON_CACHE_FLUSH_ENABLE_ALL or the bitwise OR
 * of one or more of the following flags: NEURON_CACHE_FLUSH_DISABLE_SYNC_INPUT,
 * NEURON_CACHE_FLUSH_DISABLE_INVALIDATE_OUTPUT.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronExecution_setCacheFlushHint(
    NeuronExecution* execution,
    uint8_t flushHint);

/**
 * Get the dimensional information of the specified output operand of the model
 * of the latest computation evaluated on {@link NeuronExecution}.
 *
 * This function may only be invoked when the execution is in the completed
 * state.
 *
 * Available since 5.0.0
 *
 * @param execution The execution to be queried.
 * @param index The index of the output argument we are querying. It is
 *              an index into the lists passed to {@link
 * NeuronModel_identifyInputsAndOutputs}.
 * @param rank The rank of the output operand.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronExecution_getOutputOperandRank(
    NeuronExecution* execution,
    int32_t index,
    uint32_t* rank);

/**
 * Get the dimensional information of the specified output operand of the model
 * of the latest computation evaluated on {@link NeuronExecution}. The target
 * output operand cannot be a scalar.
 *
 * This function may only be invoked when the execution is in the completed
 * state.
 *
 * Available since 5.0.0
 *
 * @param execution The execution to be queried.
 * @param index The index of the output argument we are querying. It is
 *              an index into the lists passed to {@link
 * NeuronModel_identifyInputsAndOutputs}.
 * @param dimensions The dimension array to be filled. The size of the array
 * must be exactly as large as the rank of the output operand to be queried in
 * the model.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronExecution_getOutputOperandDimensions(
    NeuronExecution* execution,
    int32_t index,
    uint32_t* dimensions);

/**
 * Create a NeuronCompilation which can create executions with shared static
 * memory.
 *
 * This function only creates the object. Compilation is only performed once
 * NeuronCompilation_finish is invoked. NeuronCompilation_finish should be
 * called once all desired properties have been set on the compilation.
 * NeuronModel_free should be called once the compilation is no longer needed.
 * The provided model must outlive the compilation. The model must already have
 * been finished by a call to NeuronModel_finish.
 *
 * Available since 7.0.0
 *
 * @param model The NeuronModel to be compiled.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return NEURON_NO_ERROR if successful
 */
int NeuronCompilation_createForBatch(
    NeuronModel* model,
    NeuronCompilation** compilation);

/**
 * Set the size of runner pool, and create same number of runners.
 *
 * The execution must created by the following steps:
 * NeuronCompilation_createForBatch, NeuronCompilation_finish,
 * NeuronExecution_create.
 *
 * The execution created from this compilation has to use
 * NeuronExecution_setRunnerPoolSize to create thread pool and then set a series
 * of inputs & outputs into the execution. The execution will inference with the
 * series of inputs.
 *
 * Available since 7.0.0
 *
 * @param execution The NeuronExecution to be utilized.
 * @param numRunners The number of runner need to be created.
 *
 * @return NEURON_NO_ERROR if successful
 * @return NEURON_BAD_STATE if the compilation is not created via
 * NeuronCompilation_createForBatch.
 */
int NeuronExecution_setRunnerPoolSize(
    NeuronExecution* execution,
    uint8_t numRunners);

/**
 * Notify the execution that all inputs / outputs have been set.
 * Should be called after NeuronExecution_setInputFromMemory and
 * NeuronExecution_setOutputFromMemory.
 *
 * The execution must created by the following steps:
 * NeuronCompilation_createForBatch, NeuronCompilation_finish,
 * NeuronExecution_create.
 *
 * Available since 7.0.0
 *
 * @param execution The NeuronExecution to be utilized.
 *
 * @return NEURON_NO_ERROR if successful
 * @return NEURON_BAD_STATE if the compilation is not created via
 * NeuronCompilation_createForBatch.
 */
int NeuronExecution_setBatchDone(NeuronExecution* execution);

/**
 * Notify the execution that all inputs / outputs have been set.
 * Should be called after NeuronExecution_setInputFromMemory and
 * NeuronExecution_setOutputFromMemory.
 *
 * The execution must created by the following steps:
 * 1. NeuronCompilation_createV2 with COMPILATION_TYPE_EXECUTION_CONTROLLER
 * 2. NeuronCompilation_finish
 * 3. NeuronExecution_create.
 * or
 * 1. NeuronModel_restoreFromCompiledNetworkV2  with
 * COMPILATION_TYPE_EXECUTION_CONTROLLER
 * 2. NeuronExecution_create.
 *
 * Available since 7.0.1
 *
 * @param execution The NeuronExecution to be utilized.
 * @param idx The index of runner to set the previous inputs and outputs.
 *
 * @return NEURON_NO_ERROR if successful
 * @return NEURON_BAD_STATE if the compilation is not created via
 *             COMPILATION_TYPE_EXECUTION_CONTROLLER.
 */
int NeuronExecution_setIODone(NeuronExecution* execution, int idx);

/**
 * Create a NeuronCompilation which can create executions with shared static
 * memory.
 *
 * This function only creates the object. Compilation is only performed once
 * NeuronCompilation_finish is invoked. NeuronCompilation_finish should be
 * called once all desired properties have been set on the compilation.
 * NeuronModel_free should be called once the compilation is no longer needed.
 * The provided model must outlive the compilation. The model must already have
 * been finished by a call to NeuronModel_finish.
 *
 * The executions created from this compilation can be executed at the same
 * time.
 *
 * Available since 7.0.0
 *
 * @param model The NeuronModel to be compiled.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return NEURON_NO_ERROR if successful
 */
int NeuronCompilation_createForMultiExecutions(
    NeuronModel* model,
    NeuronCompilation** compilation);

/**
 * Set report path for debug plus.
 *
 * Only be used in debug purpose, the execution should be created by
 * NeuronCompilation_createForDebug compilation.
 *
 * Available since 5.0.0
 *
 * @param model The model need to be debug.
 * @param path The path of execution report.
 *
 * @return NEURON_NO_ERROR if successful, NEURON_BAD_DATA if the path is empty.
 */
int NeuronDebug_setReportPath(NeuronModel* model, const char* path);

/**
 * Get the number of available devices.
 *
 * Available since 4.1.0
 * @param numDevices The number of devices returned.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int Neuron_getDeviceCount(uint32_t* numDevices);

/**
 * Get the representation of the specified device.
 *
 * Available since 4.1.0
 *
 * @param devIndex The index of the specified device. Must be less than the
 * number of available devices.
 * @param device The representation of the specified device. The same
 * representation will always be returned for the specified device.
 *
 * @return NEURONNO_ERROR if successful.
 */
int Neuron_getDevice(uint32_t devIndex, NeuronDevice** device);

/**
 * Get the name of the specified device.
 *
 * Available since 4.1.0
 *
 * @param device The representation of the specified device.
 * @param name The returned name of the specified device. The name will remain
 * valid for the duration of the application.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronDevice_getName(const NeuronDevice* device, const char** name);

/**
 * Get the description of the specified device.
 *
 * Available since 5.0.0
 *
 * @param device The representation of the specified device.
 * @param description The returned description of the specified device. The
 * description will remain valid for the duration of the application.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronDevice_getDescription(
    const NeuronDevice* device,
    const char** description);

/*
 * Destroys the event.
 *
 * See NeuronExecution for information on multithreaded usage.
 *
 * Available since 5.0.0
 *
 * @param event The event object to be destroyed. Passing NULL is acceptable and
 *              results in no operation.
 */
void NeuronEvent_free(NeuronEvent* event);

/*
 * Force destroys the event without calling NeuronEvent_wait().
 * If user wants do wait before destroying the event, they should use
 * NeuronEvent_free.
 *
 * See NeuronExecution for information on multithreaded usage.
 *
 * Available since 6.0.0
 *
 * @param event The event object to be destroyed. Passing NULL is acceptable and
 *              results in no operation.
 */
void NeuronEvent_freeForce(NeuronEvent* event);

/**
 * Waits until the execution completes.
 *
 * More than one thread can wait on an event. When the execution completes,
 * all threads will be released.
 *
 * SeeNeuronExecution for information on multithreaded usage.
 *
 * Available since 5.0.0
 *
 * @param event The event that will be signaled on completion.
 * @return NEURON_NO_ERROR if the execution completed normally.
 *         NEURON_UNMAPPABLE if the execution input or output memory cannot
 *         be properly mapped.
 */
int NeuronEvent_wait(NeuronEvent* event);

/**
 * Create a NeuronEventfrom a sync_fence file descriptor.
 *
 * The newly created NeuronEvent does not take ownership of the provided
 * sync_fence_fd, it will instead dup the provided sync_fence_fd and own the
 * duplicate.
 *
 * @param sync_fence_fd The sync_fence file descriptor.
 * @param event The newly created object or NULL if unsuccessful.
 *
 * @return NEURON_NO_ERROR if successful.
 *
 * Available since 5.0.0
 */
int NeuronEvent_createFromSyncFenceFd(int sync_fence_fd, NeuronEvent** event);

/**
 * Get sync_fence file descriptor from the event.
 *
 * If the NeuronEvent is not backed by a sync fence, the sync_fence_fd
 * will be set to -1, and NEURON_BAD_DATA will be returned.
 *
 * See NeuronEvent_createFromSyncFenceFd and
 * NeuronExecution_startComputeWithDependencies to see how to create an event
 * backed by a sync fence.
 *
 * The user takes ownership of the returned fd, and must close the returned file
 * descriptor when it is no longer needed.
 *
 * @param event An event that is backed by a sync fence.
 * @param sync_fence_fd The sync_fence file descriptor. The file descriptor will
 *                      be set to -1 if there is an error.
 *
 * @return NEURON_NO_ERROR if successful.
 *
 * Available since 5.0.0
 */
int NeuronEvent_getSyncFenceFd(const NeuronEvent* event, int* sync_fence_fd);

/**
 * Queries whether an extension is supported by the driver implementation of the
 * specified device.
 *
 * @param extension The extension name.
 * @param isExtensionSupported The boolean value indicating whether the
 * extension is supported.
 *
 * @return NEURON_NO_ERROR if successful.
 *
 * Available since 5.0.0
 */
// Note: Remove "device"
int NeuronDevice_getExtensionSupport(
    const char* extensionName,
    bool* isExtensionSupported);

/**
 * Creates an operand type from an extension name and an extension operand code.
 *
 * See {@link NeuronModel} for information on multithreaded usage.
 *
 * Available since 5.0.0
 *
 * @param model The model to contain the operand.
 * @param extensionName The extension name.
 * @param operandCodeWithinExtension The extension operand code.
 * @param type The operand type.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_getExtensionOperandType(
    NeuronModel* model,
    const char* extensionName,
    uint16_t operandCodeWithinExtension,
    int32_t* type);

/**
 * Creates an operation type from an extension name and an extension operation
 * code.
 *
 * See {@link NeuronModel} for information on multithreaded usage.
 *
 * Available since 5.0.0
 *
 * @param model The model to contain the operation.
 * @param extensionName The extension name.
 * @param operationCodeWithinExtension The extension operation code.
 * @param type The operation type.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_getExtensionOperationType(
    NeuronModel* model,
    const char* extensionName,
    uint16_t operationCodeWithinExtension,
    int32_t* type);

/**
 * Sets extension operand parameters.
 *
 * Available since 5.0.0
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param data A pointer to the extension operand data.
 *             The data does not have to outlive the call to this function.
 * @param length The size in bytes of the data value.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronModel_setOperandExtensionData(
    NeuronModel* model,
    int32_t index,
    const void* data,
    size_t length);

/**
 * Gets the execution preference associated with this compilation.
 * This function must be called after calling NeuronCompilation_finish.
 *
 * Available since 6.0.0
 *
 * @param compilation The compilation to be queried.
 * @param preference The execution preference will be one of NEURON_PREFER_*.
 * Ignore preference value if this function doesn't return NEURON_NO_ERROR.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_getPreference(
    NeuronCompilation* compilation,
    int* preference);

/**
 * Gets the execution priority associated with this compilation.
 * This function must be called after calling NeuronCompilation_finish.
 *
 * Available since 6.0.0
 *
 * @param compilation The compilation to be queried.
 * @param priority The priority will be one of NEURON_PRIORITY_*. Ignore
 * priority value if this function doesn't return NEURON_NO_ERROR.
 *
 * @return NEURON_NO_ERROR if successful.
 */
int NeuronCompilation_getPriority(
    NeuronCompilation* compilation,
    int* priority);

int NeuronCompilation_createWithOptions(
    NeuronModel* model,
    NeuronCompilation** compilation,
    const char* options);
__END_DECLS
