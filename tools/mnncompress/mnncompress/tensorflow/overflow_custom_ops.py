# ==============================================================================
# Copyright 2019 The Alibaba Authors. All Rights Reserved.
# @author Shuofan, Yizhi, Mingyang
# @email hongwei.xhw@alibaba-inc.com
# @company AI Labs, Alibaba Group
# @cite Overflow Aware Quantization: More Efficient Inference on Ubiquitous Embedded Hardware
# @modified 2019/11/25
# @file overflow_custom_ops.py
# @desc int16-mac convolution and matmul.
#
# ==============================================================================
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op


def matmul2d(A, B, dtype=None):
  """
  custom matmul2d
  :param A:
  :param B:
  :param dtype:
  :return:
  """
  with variable_scope.variable_scope('matmul2d'):
    if dtype is not None:
      A = math_ops.cast(A, dtype)
      B = math_ops.cast(B, dtype)

    a_shape = A.shape
    b_shape = B.shape

    assert len(a_shape) == 2 and len(b_shape) == 2 , 'Expect A and B are 2D inputs'
    assert a_shape[1] == b_shape[0], 'Expect M*N and N*K'

    A_tile = array_ops.tile(A, [1, b_shape[1]])  # copy itself
    A_tile = array_ops.reshape(A_tile, [a_shape[0], b_shape[1],
                                        a_shape[1]])  # reshape for multiply

    B_transpose = array_ops.transpose(B, [1, 0])
    multi_result = math_ops.multiply(A_tile, B_transpose)
    ret = math_ops.reduce_sum(multi_result, axis=-1)
    return ret


def conv2d(input_tensor, weights, strides, padding='SAME', dtype=None):
  """
  custom conv2d
  :param input_tensor:
  :param weights:
  :param strides:
  :param padding:
  :param dtype:
  :return:
  """
  with variable_scope.variable_scope('MyConv2d'):
    # filter shape: [filter_height, filter_width, in_channels, out_channels]
    # flatten filters
    if dtype is not None:
      input_tensor = math_ops.cast(input_tensor, dtype)
      weights = math_ops.cast(weights, dtype)
    filter_height = int(weights.shape[0])
    filter_width = int(weights.shape[1])
    in_channels = int(weights.shape[2])
    out_channels = int(weights.shape[3])

    flat_w = array_ops.reshape(weights,
                               [filter_height * filter_width * in_channels,
                                out_channels])
    patches = array_ops.extract_image_patches(
        input_tensor,
        ksizes=[1, filter_height, filter_width, 1],
        strides=strides,
        rates=[1, 1, 1, 1],
        padding=padding
    )

    patches_reshape = patches.get_shape().as_list()
    features = math_ops.reduce_sum(math_ops.multiply(flat_w[:, 0], patches),
                                   axis=3, keepdims=True)
    features = array_ops.expand_dims(features, axis=0)

    a = constant_op.constant(1, name='ii')
    n = constant_op.constant(out_channels, name='nn')

    def cond(a, n, features):
      return a < n

    def body(a, n, features):
      tmp = array_ops.expand_dims(
          math_ops.reduce_sum(math_ops.multiply(flat_w[:, a], patches), axis=3,
                              keepdims=True), axis=0)
      features = array_ops.concat([features, tmp], axis=0)
      a = a + 1
      return a, n, features

    a, n, features = control_flow_ops.while_loop(cond, body, [a, n, features],
                                                 shape_invariants=[
                                                     a.get_shape(),
                                                     n.get_shape(),
                                                     tensor_shape.TensorShape(
                                                         [None,
                                                          patches_reshape[0],
                                                          patches_reshape[1],
                                                          patches_reshape[2],
                                                          1]
                                                     )
                                                 ])
    temp_patches_reshape = patches_reshape
    for i in range(len(temp_patches_reshape)):
      if temp_patches_reshape[i] is None:
        temp_patches_reshape[i] = -1

    features = array_ops.reshape(features, [out_channels, temp_patches_reshape[0],
                                            temp_patches_reshape[1],
                                            temp_patches_reshape[2], 1])
    features = array_ops.unstack(features, axis=0)
    features = array_ops.concat(features, axis=-1)

    return features


def depthwise_conv2d(input_tensor, weights, strides, padding='SAME', dtype=None):
  """
  custom depthwise_conv2d
  :param input_tensor:
  :param weights:
  :param strides:
  :param padding:
  :param dtype:
  :return:
  """
  with variable_scope.variable_scope('MyDepthwiseConv2d'):
    # filter shape: [filter_height, filter_width, in_channels, out_channels]
    # flatten filters
    if dtype is not None:
      input_tensor = math_ops.cast(input_tensor, dtype)
      weights = math_ops.cast(weights, dtype)
    filter_height = int(weights.shape[0])
    filter_width = int(weights.shape[1])
    in_channels = int(weights.shape[2])
    out_channels = int(weights.shape[3])
    assert out_channels == 1, 'Expect output channel is 1.'

    flat_w = array_ops.reshape(weights,
                               [filter_height * filter_width, in_channels])

    patches = array_ops.extract_image_patches(
        input_tensor,
        ksizes=[1, filter_height, filter_width, 1],
        strides=strides,
        rates=[1, 1, 1, 1],
        padding=padding
    )

    patches_reshape = patches.shape.as_list()
    for i in range(len(patches_reshape)):
      if patches_reshape[i] is None:
        patches_reshape[i] = -1
    patches_depthwise = array_ops.reshape(patches,
                                          [patches_reshape[0],
                                           patches_reshape[1],
                                           patches_reshape[2],
                                           filter_height * filter_width,
                                           in_channels])
    flat_w_t = array_ops.transpose(flat_w, [1, 0])
    patches_depthwise_t = array_ops.transpose(patches_depthwise,
                                              [0, 1, 2, 4, 3])
    tmp = math_ops.multiply(patches_depthwise_t, flat_w_t)
    features = array_ops.squeeze(
        math_ops.reduce_sum(tmp, axis=-1, keepdims=True), axis=-1)

    return features


def add(A, B, dtype=None):
  """
  custom add
  :param A:
  :param B:
  :param dtype:
  :return:
  """
  if dtype is not None:
    A = math_ops.cast(A, dtype)
    B = math_ops.cast(B, dtype)
  return math_ops.add(A, B)


def matmul(A, B, dtype=None):
  """
  custom matmul
  :param A:
  :param B:
  :param dtype:
  :return:
  """
  if dtype is not None:
    A = math_ops.cast(A, dtype)
    B = math_ops.cast(B, dtype)

  dtype = A.dtype
  if dtype in [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
               dtypes.complex64, dtypes.complex128]:
    return math_ops.matmul(A, B)

  a_shape = A.shape
  b_shape = B.shape

  assert len(a_shape) == 3 and len(b_shape) == 3, 'Expect A and B are 3D inputs'
  assert a_shape[-1] == b_shape[1] ,'Expect M*N*K and M*K*D'

  A_tile = array_ops.tile(A, [1, b_shape[-1], 1])  # copy itself
  A_tile = array_ops.reshape(A_tile, [a_shape[0], b_shape[-1], a_shape[1],
                                      a_shape[2]])  # reshape for multiply
  A_tile_transpose = array_ops.transpose(A_tile,
                                         [0, 2, 1, 3])  # reshap for multiply
  B_transpose = array_ops.transpose(B, [0, 2, 1])
  B_transpose_tile = array_ops.tile(B_transpose, [1, a_shape[1], 1])
  B_transpose_tile = array_ops.reshape(B_transpose_tile,
                                       [b_shape[0], a_shape[1], b_shape[2],
                                        b_shape[1]])
  ret = math_ops.reduce_sum(
      math_ops.multiply(A_tile_transpose, B_transpose_tile), axis=-1)
  return ret
