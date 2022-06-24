from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorly as tl
import scipy
from tensorly.decomposition import partial_tucker

from mnncompress.common import VBMF
from ..common import MNN_compression_pb2 as compress_pb
from mnncompress.common.log import mnn_logger
from mnncompress.common.helper import get_align_channels
import uuid
from .helpers import is_weight_tensor, update_consumer_inputs
from .graph_checker import check_for_grad_ops

_Support_Ops = ['Conv2D', 'MatMul']
_MNN_variable_collection_name = "MNN_decompose_variables"

origin_params_num = decompose_model_params_num = 0

def _decompose_conv_1x1(weights_info, op, align_channels, reserved_singular_value_ratio, scope, ori_args):
    assert op.type == "Conv2D"
    feature = op.inputs[0]
    weight = weights_info[op.name]
    kernel_shape = [weight.shape[0], weight.shape[1]]
    assert kernel_shape == [1, 1]

    global decompose_model_params_num

    if len(weight.squeeze().shape) != 2:
        decompose_model_params_num += weight.size
        print("skip svd for", op.name, "weight shape:", weight.shape)
        return
    
    weight_2d = weight.squeeze().transpose()
    u, s, v = scipy.linalg.svd(weight_2d)
    singular_value_sum = np.sum(s)
    n_dim = 1
    temp_sum = 0.0
    for i in range(0, s.size):
        temp_sum += s[i]
        n_dim = i+1
        if temp_sum / singular_value_sum >= reserved_singular_value_ratio:
            break
    n_dim = get_align_channels(n_dim, s.size, align_channels)
    print("svd for", op.name, "weight shape:", weight.shape, "n_dim:", n_dim)

    conv1_weight = np.matmul(np.diag(s[0:n_dim]), v[0:n_dim, :]).transpose().reshape((1, 1, -1, n_dim))
    conv2_weight = u[:, 0:n_dim].transpose().reshape((1, 1, n_dim, -1))

    decompose_model_params_num += (conv1_weight.size + conv2_weight.size)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        conv1_weight_variable = tf.get_variable(name='svd_conv_w1', initializer=conv1_weight, trainable=True)
        tf.add_to_collection(_MNN_variable_collection_name, conv1_weight_variable)
        conv2_weight_variable = tf.get_variable(name='svd_conv_w2', initializer=conv2_weight, trainable=True)
        tf.add_to_collection(_MNN_variable_collection_name, conv2_weight_variable)

        conv1 = tf.nn.conv2d(feature, conv1_weight_variable, strides=ori_args["strides"], padding=ori_args["padding"], dilations=ori_args["dilations"])
        conv2 = tf.nn.conv2d(conv1, conv2_weight_variable, padding="VALID")
        update_consumer_inputs(op.outputs[0], conv2)


def _decompose_conv(weights_info, op, align_channels, tucker_minimal_ratio, reserved_singular_value_ratio, scope):
    assert op.type == "Conv2D"
    feature = op.inputs[0]
    weight = weights_info[op.name]
    kernel_shape = [weight.shape[0], weight.shape[1]]
    in_out_channels = [weight.shape[2], weight.shape[3]]

    global decompose_model_params_num

    if in_out_channels[0] <= align_channels or in_out_channels[1] <= align_channels:
        decompose_model_params_num += weight.size
        print("skip decompose for conv:", op.name, "in_out_channels:", in_out_channels)
        return

    ori_args = {}
    ori_args["strides"] = op.get_attr("strides")
    ori_args["padding"] = op.get_attr("padding")
    ori_args["dilations"] = op.get_attr("dilations")

    if kernel_shape == [1, 1]:
        _decompose_conv_1x1(weights_info, op, align_channels, reserved_singular_value_ratio, scope, ori_args)
    else:
        weight_nchw = weight.transpose((3, 2, 0, 1))
        u0 = tl.base.unfold(weight_nchw, 0)
        u1 = tl.base.unfold(weight_nchw, 1)
        res0 = VBMF.EVBMF(u0)
        res1 = VBMF.EVBMF(u1)
        rank0 = get_align_channels(res0[1].shape[0], in_out_channels[1], align_channels, tucker_minimal_ratio)
        rank1 = get_align_channels(res1[1].shape[1], in_out_channels[0], align_channels, tucker_minimal_ratio)
        ranks = [rank0, rank1]

        core, [last, first] = partial_tucker(weight_nchw, modes=[0, 1], rank=ranks, init='svd')
        print("tucker for", op.name, ":", in_out_channels, "<===>", [core.shape[1], core.shape[0]], "ranks:", ranks)

        conv1_weight = np.expand_dims(np.expand_dims(first.transpose(), -1), -1).transpose((2, 3, 1, 0))
        conv2_weight = core.transpose((2, 3, 1, 0))
        conv3_weight = np.expand_dims(np.expand_dims(last, -1), -1).transpose((2, 3, 1, 0))

        decompose_model_params_num += (conv1_weight.size + conv2_weight.size + conv3_weight.size)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            conv1_weight_variable = tf.get_variable(name='tucker_conv_w1', initializer=conv1_weight, trainable=True)
            tf.add_to_collection(_MNN_variable_collection_name, conv1_weight_variable)
            conv2_weight_variable = tf.get_variable(name='tucker_conv_w2', initializer=conv2_weight, trainable=True)
            tf.add_to_collection(_MNN_variable_collection_name, conv2_weight_variable)
            conv3_weight_variable = tf.get_variable(name='tucker_conv_w3', initializer=conv3_weight, trainable=True)
            tf.add_to_collection(_MNN_variable_collection_name, conv3_weight_variable)

            conv1 = tf.nn.conv2d(feature, conv1_weight_variable, padding="VALID")
            conv2 = tf.nn.conv2d(conv1, conv2_weight_variable, strides=ori_args["strides"], padding=ori_args["padding"], dilations=ori_args["dilations"])
            conv3 = tf.nn.conv2d(conv2, conv3_weight_variable, padding="VALID")
            update_consumer_inputs(op.outputs[0], conv3)


def _decompose_matmul(weights_info, op, align_channels, reserved_singular_value_ratio, scope):
    assert op.type == "MatMul"
    if is_weight_tensor(op.inputs[0]) and is_weight_tensor(op.inputs[1]):
        return
    if (not is_weight_tensor(op.inputs[0])) and (not is_weight_tensor(op.inputs[1])):
        return

    global decompose_model_params_num

    for index in range(len(op.inputs)):
        input_tensor = op.inputs[index]
        if not is_weight_tensor(input_tensor):
            continue

        weight = weights_info[op.name]
        if len(weight.squeeze().shape) != 2:
            decompose_model_params_num += weight.size
            print("skip svd for", op.name, "weight shape:", weight.shape)
            continue

        if weight.shape[0] <= align_channels or weight.shape[1] <= align_channels:
            decompose_model_params_num += weight.size
            print("skip svd for", op.name, "weight shape:", weight.shape)
            continue

        u, s, v = scipy.linalg.svd(weight)
        singular_value_sum = np.sum(s)
        n_dim = 1
        temp_sum = 0.0
        for i in range(0, s.size):
            temp_sum += s[i]
            n_dim = i+1
            if temp_sum / singular_value_sum >= reserved_singular_value_ratio:
                break
        n_dim = get_align_channels(n_dim, s.size, align_channels)
        print("svd for", op.name, "weight shape:", weight.shape, "n_dim:", n_dim)

        fc1_weight = np.matmul(np.diag(s[0:n_dim]), v[0:n_dim, :])
        fc2_weight = u[:, 0:n_dim]

        decompose_model_params_num += (fc1_weight.size + fc2_weight.size)

        if index == 0:
            feature = op.inputs[1]
            ori_output_tensor = op.outputs[0]
            weight_trans = op.get_attr("transpose_a")
            feature_trans = op.get_attr("transpose_b")

            if not weight_trans:
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                    fc1_weight_variable = tf.get_variable(name='svd_fc_w1', initializer=fc1_weight, trainable=True)
                    tf.add_to_collection(_MNN_variable_collection_name, fc1_weight_variable)
                    fc2_weight_variable = tf.get_variable(name='svd_fc_w2', initializer=fc2_weight, trainable=True)
                    tf.add_to_collection(_MNN_variable_collection_name, fc2_weight_variable)

                    fc1 = tf.matmul(fc1_weight_variable, feature, transpose_b=feature_trans)
                    fc2 = tf.matmul(fc2_weight_variable, fc1)
                    update_consumer_inputs(ori_output_tensor, fc2)
            else:
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                    fc1_weight_variable = tf.get_variable(name='svd_fc_w1', initializer=fc1_weight.transpose(), trainable=True)
                    tf.add_to_collection(_MNN_variable_collection_name, fc1_weight_variable)
                    fc2_weight_variable = tf.get_variable(name='svd_fc_w2', initializer=fc2_weight.transpose(), trainable=True)
                    tf.add_to_collection(_MNN_variable_collection_name, fc2_weight_variable)

                    fc1 = tf.matmul(fc2_weight_variable, feature, transpose_b=feature_trans)
                    fc2 = tf.matmul(fc1_weight_variable, fc1)
                    update_consumer_inputs(ori_output_tensor, fc2)
        
        if index == 1:
            feature = op.inputs[0]
            ori_output_tensor = op.outputs[0]
            weight_trans = op.get_attr("transpose_b")
            feature_trans = op.get_attr("transpose_a")

            if not weight_trans:
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                    fc1_weight_variable = tf.get_variable(name='svd_fc_w1', initializer=fc1_weight, trainable=True)
                    tf.add_to_collection(_MNN_variable_collection_name, fc1_weight_variable)
                    fc2_weight_variable = tf.get_variable(name='svd_fc_w2', initializer=fc2_weight, trainable=True)
                    tf.add_to_collection(_MNN_variable_collection_name, fc2_weight_variable)

                    fc1 = tf.matmul(feature, fc2_weight_variable, transpose_a=feature_trans)
                    fc2 = tf.matmul(fc1, fc1_weight_variable)
                    update_consumer_inputs(ori_output_tensor, fc2)
            else:
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                    fc1_weight_variable = tf.get_variable(name='svd_fc_w1', initializer=fc1_weight.transpose(), trainable=True)
                    tf.add_to_collection(_MNN_variable_collection_name, fc1_weight_variable)
                    fc2_weight_variable = tf.get_variable(name='svd_fc_w2', initializer=fc2_weight.transpose(), trainable=True)
                    tf.add_to_collection(_MNN_variable_collection_name, fc2_weight_variable)

                    fc1 = tf.matmul(feature, fc1_weight_variable, transpose_a=feature_trans)
                    fc2 = tf.matmul(fc1, fc2_weight_variable)
                    update_consumer_inputs(ori_output_tensor, fc2)


def low_rank_decompose(graph, weight_npy_file, compress_params_file, skip_layers=[""], align_channels=8, tucker_minimal_ratio=0.25, reserved_singular_value_ratio=0.5, append=False):
    weights_info = np.load(weight_npy_file, allow_pickle=True).item()

    grad_ops = check_for_grad_ops(graph)
    if grad_ops:
        raise ValueError('gradient op found in graph, exiting %s\nplease invoke with inference graph only, before construct model optimizer\n' % grad_ops)
    
    global origin_params_num, decompose_model_params_num

    all_ops = graph.get_operations()
    decompose_layers = []
    for op in all_ops:
        if 'gradients/' in op.name and '_grad' in op.name:
            continue

        if op.name in skip_layers:
            if op.type == "Conv2D" and len(op.outputs[0].consumers()) > 0:
                origin_params_num += op.inputs[1].shape.num_elements()
                decompose_model_params_num += op.inputs[1].shape.num_elements()
            if op.type == "MatMul" and len(op.outputs[0].consumers()) > 0:
                for inp in op.inputs:
                    if is_weight_tensor(inp):
                        origin_params_num += inp.shape.num_elements()
                        decompose_model_params_num += inp.shape.num_elements()

        if (op.type in _Support_Ops) and (op.name not in skip_layers):
            if len(op.outputs[0].consumers()) > 0:
                decompose_layers.append(op)

                if op.type == "Conv2D":
                    origin_params_num += op.inputs[1].shape.num_elements()
                if op.type == "MatMul":
                    for inp in op.inputs:
                        if is_weight_tensor(inp):
                            origin_params_num += inp.shape.num_elements()
    
    for i in range(len(decompose_layers)):
        op = decompose_layers[i]
        scope = "MNN_decompose_" + str(i)
        if op.type == "Conv2D":
            print("decompose conv:", op.name)
            _decompose_conv(weights_info, op, align_channels, tucker_minimal_ratio, reserved_singular_value_ratio, scope)
        
        if op.type == "MatMul":
            print("decompose matmul:", op.name)
            _decompose_matmul(weights_info, op, align_channels, reserved_singular_value_ratio, scope)

    compress_proto = compress_pb.Pipeline()
    if append:
        f = open(compress_params_file, 'rb')
        compress_proto.ParseFromString(f.read())

    compress_proto.version = "0.0.0"
    if compress_proto.mnn_uuid == '':
        model_guid = str(uuid.uuid4())
        compress_proto.mnn_uuid = model_guid
    else:
        model_guid = compress_proto.mnn_uuid

    f = open(compress_params_file, 'wb')
    f.write(compress_proto.SerializeToString())
    f.close()

    detail = {"algorithm": "low_rank_decompose", "compression_rate": origin_params_num / decompose_model_params_num, \
        "ori_model_size": origin_params_num * 4.0 / 1024.0 / 1024.0, \
        "config": {"skip_layers": skip_layers, "align_channels": align_channels, "tucker_minimal_ratio": tucker_minimal_ratio, "reserved_singular_value_ratio": reserved_singular_value_ratio}}

    mnn_logger.on_done("pytorch", model_guid, detail)


def get_op_weight_values(sess, npy_file_name):
    graph = sess.graph

    grad_ops = check_for_grad_ops(graph)
    if grad_ops:
        raise ValueError('gradient op found in graph, exiting %s\nplease invoke with inference graph only, before construct model optimizer\n' % grad_ops)
    
    all_ops = graph.get_operations()
    op_name_weight_values = {}

    for op in all_ops:
        if 'gradients/' in op.name and '_grad' in op.name:
            continue

        if op.type == "Conv2D":
            weight = sess.run(op.inputs[1])
            op_name_weight_values[op.name] = weight

        if op.type == "MatMul":
            if is_weight_tensor(op.inputs[0]) and is_weight_tensor(op.inputs[1]):
                continue
            if (not is_weight_tensor(op.inputs[0])) and (not is_weight_tensor(op.inputs[1])):
                continue
            
            if is_weight_tensor(op.inputs[0]):
                weight = sess.run(op.inputs[0])
                op_name_weight_values[op.name] = weight

            if is_weight_tensor(op.inputs[1]):
                weight = sess.run(op.inputs[1])
                op_name_weight_values[op.name] = weight

    np.save(npy_file_name, op_name_weight_values)
    print("op weights saved to npy file:", npy_file_name)
