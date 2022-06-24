from __future__ import print_function
import tensorflow as tf
import yaml
import numpy as np
from .graph_checker import check_for_grad_ops

def get_op_name_of_type(graph, type_list):
    ops = graph.get_operations()
    names = []
    for one_op in ops:
        if one_op.type in type_list:
            names.append(one_op.name)
    
    return names


def get_input_tensor_index(op, input_tensor):
    return_list = []
    for i in range(len(op.inputs)):
        if input_tensor == op.inputs[i]:
            return_list.append(i)
    return return_list


_Variable_types = ['Variable', 'VariableV2', 'Const']

def is_weight_tensor(tensor):
    cond1 = len(tensor.op.inputs) == 1
    if cond1 == False:
        return False

    if tensor.op.type == 'Identity':
        if tensor.op.inputs[0].op.type in _Variable_types:
            return True
    else:
    # if tensor.op.type == 'Enter':
        if tensor.op.inputs[0].op.type == "Identity":
            if tensor.op.inputs[0].op.inputs[0].op.type in _Variable_types:
                return True
    
    return False

def get_trainable_weights(session):
    variables = tf.trainable_variables()
    np_values = session.run(variables)
    return variables, np_values

def kronecker_tf(A, B):
    A_shape = A.get_shape()
    B_shape = B.get_shape()

    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            if j == 0:
                # temp = tf.squeeze(A[i, j] * B)
                temp = A[i, j] * B
            else:
                # temp = tf.concat([temp, tf.squeeze(A[i, j] * B)], 1)
                temp = tf.concat([temp, A[i, j] * B], 1)
        if i == 0:
            result = temp
        else:
            result = tf.concat([result, temp], 0)
    return result

def generate_prune_config_yaml(graph, filename, default_prune_ratio=0.5, generate_template=False):
    f = open(filename, 'w')
    config = {}
    prune_ratio_config = {}
    if generate_template:
        conv2d_op_names = get_op_name_of_type(graph, ['Conv2D'])
        for op_name in conv2d_op_names:
            prune_ratio_config[op_name] = '-.--'
        depthwise_conv_op_names = get_op_name_of_type(graph, ['DepthwiseConv2dNative'])
        for op_name in depthwise_conv_op_names:
            prune_ratio_config[op_name] = ',.,,'
        matmul_op_names = get_op_name_of_type(graph, ['MatMul'])
        for op_name in matmul_op_names:
            prune_ratio_config[op_name] = '--fc--'
    else:
        op_names = get_op_name_of_type(graph, ['Conv2D', 'DepthwiseConv2dNative', 'MatMul'])
        for op_name in op_names:
            prune_ratio_config[op_name] = default_prune_ratio
    
    config['prune_ratios'] = prune_ratio_config
    f.write(yaml.dump(config))
    f.close()

def get_weight_variable_name(tensor):
    return tensor.op.inputs[0].op.name

def get_variable_by_name(name, graph=None):
    if len(tf.global_variables()) == 0:
        name = name + ":0"
        return graph.get_tensor_by_name(name)
    else:
        res = None
        all_variable_names = [v.op.name for v in tf.global_variables()]
        if name in all_variable_names:
            res = [v for v in tf.global_variables() if v.op.name == name][0]
        return res

def get_variable_by_tensor(tensor):
    if tensor.op.type == "ExpandDims":
        tensor = tensor.op.inputs[0]
    if tensor.op.type == "Switch":
        tensor = tensor.op.inputs[0].op.inputs[0]
    variable_name = get_weight_variable_name(tensor)
    if variable_name.endswith("/read"):
        variable_name = variable_name[:-5]
    variable = get_variable_by_name(variable_name, tensor.graph)
    return variable

@tf.custom_gradient
def custom_scale_grad(x, num, clamp_value):
    def grad(dy):
        gs = 1. / tf.math.sqrt(tf.cast(num, tf.float32) * clamp_value + 0.0)
        return dy * gs, None, None
    
    return tf.identity(x), grad

def update_consumer_inputs(ori_output_tensor, new_output_tensor):
    output_consumers = ori_output_tensor.consumers()
    for c in output_consumers:
        index = get_input_tensor_index(c, ori_output_tensor)
        for id in index:
            c._update_input(id, new_output_tensor)

def check_return_cond(output_tensor):
    output_no_grad_consumers = [c for c in output_tensor.consumers() if 'gradients/' not in c.name]
    if len(output_no_grad_consumers) != 1:
        if len(output_no_grad_consumers) != 2:
            return True
        else:
            c1 = output_tensor.consumers()[0]
            c2 = output_tensor.consumers()[1]
            if (c1.type != 'Switch') or (c2.type != 'Switch'):
                return True
    return False

def skip_identity(tensor):
    while (len(tensor.consumers()) == 1) and (tensor.consumers()[0].type == 'Identity'):
        tensor = tensor.consumers()[0].outputs[0]
    return tensor

def find_batchnorm_output_tensor(output_tensor):
    def _get_switch_index(switch_op):
        for i in range(len(switch_op.outputs)):
            if len(switch_op.outputs[i].consumers()) > 0:
                return i
        raise ValueError("switch op's outputs don't have consumers")

    # batch norm with tf.cond
    output_no_grad_consumers = [c for c in output_tensor.consumers() if 'gradients/' not in c.name]

    if (len(output_no_grad_consumers) == 2) and (output_tensor.consumers()[0].type == 'Switch') and (output_tensor.consumers()[1].type == 'Switch'):
        switch_op1 = output_tensor.consumers()[0]
        switch_op2 = output_tensor.consumers()[1]

        index = _get_switch_index(switch_op1)
        no_grad_consumers = [c for c in switch_op1.outputs[index].consumers() if 'gradients/' not in c.name]
        cond1 = (len(no_grad_consumers) == 1)
        c1 = switch_op1.outputs[index].consumers()[0]
        cond2 = ('FusedBatchNorm' in c1.type)
        merge_op = c1.outputs[0].consumers()[0]
        cond3 = (merge_op.type == 'Merge')

        index = _get_switch_index(switch_op2)
        no_grad_consumers = [c for c in switch_op2.outputs[index].consumers() if 'gradients/' not in c.name]
        cond4 = (len(no_grad_consumers) == 1)
        c2 = switch_op2.outputs[index].consumers()[0]
        cond5 = ('FusedBatchNorm' in c2.type)
        cond6 = (len(merge_op.inputs) == 2)

        sub_cond1 = ((merge_op.inputs[0] == c1.outputs[0]) and (merge_op.inputs[1] == c2.outputs[0]))
        sub_cond2 = ((merge_op.inputs[0] == c2.outputs[0]) and (merge_op.inputs[1] == c1.outputs[0]))

        cond7 = (sub_cond1 or sub_cond2)

        final_cond = (cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7)

        if final_cond is True:
            output_tensor = merge_op.outputs[0]
            return output_tensor

    if 'FusedBatchNorm' in output_tensor.consumers()[0].type:
        c = output_tensor.consumers()[0]
        output_tensor = c.outputs[0]
        return output_tensor

    return output_tensor

def find_quant_output_tensor(conv2d_op):
    output_tensor = conv2d_op.outputs[0]
    op_output = output_tensor
    if check_return_cond(output_tensor):
        return output_tensor

    output_tensor = skip_identity(output_tensor)
    if check_return_cond(output_tensor):
        return op_output
    
    # conv1d squeeze
    if output_tensor.consumers()[0].type == "Squeeze":
        squeeze_op = output_tensor.consumers()[0]
        output_tensor = squeeze_op.outputs[0]

        output_tensor = skip_identity(output_tensor)
        if check_return_cond(output_tensor):
            # if conv-squeeze, but no bias or relu
            return op_output

    # bias add
    if output_tensor.consumers()[0].type in ['Add', 'AddV2', 'BiasAdd']:
        bias_add_op = output_tensor.consumers()[0]
        output_tensor = bias_add_op.outputs[0]
        op_output = output_tensor

        output_tensor = skip_identity(output_tensor)
        if check_return_cond(output_tensor):
            return op_output

    # TODO: MNN converter will turn matmul to conv2d, bias is fused, but bn relu is not, so bn relu is not fused here.
    if conv2d_op.type == "MatMul":
        return op_output

    # batch norm
    output_tensor = find_batchnorm_output_tensor(output_tensor)
    op_output = output_tensor
    output_tensor = skip_identity(output_tensor)

    if len(output_tensor.consumers()) != 1:
        return op_output

    # relu or relu6
    if output_tensor.consumers()[0].type in ['Relu', 'Relu6']:
        relu_op = output_tensor.consumers()[0]
        output_tensor = relu_op.outputs[0]
        return output_tensor

    return output_tensor

def find_bias_add_op(conv2d_op):
    output_tensor = conv2d_op.outputs[0]
    if check_return_cond(output_tensor):
        return None

    output_tensor = skip_identity(output_tensor)
    if check_return_cond(output_tensor):
        return None
    
    # bias add
    if output_tensor.consumers()[0].type in ['Add', 'AddV2', 'BiasAdd']:
        bias_add_op = output_tensor.consumers()[0]
        return bias_add_op
    
    return None

def get_batch_norm_statistics(conv2d_op):
    statistics = [None, None, None, None, None] # gamma, beta, moving_mean, moving_variance, epsilon

    output_tensor = conv2d_op.outputs[0]
    if check_return_cond(output_tensor):
        return statistics

    output_tensor = skip_identity(output_tensor)
    if check_return_cond(output_tensor):
        return statistics
    
    # bias add
    if output_tensor.consumers()[0].type in ['Add', 'AddV2', 'BiasAdd']:
        bias_add_op = output_tensor.consumers()[0]
        output_tensor = bias_add_op.outputs[0]

        output_tensor = skip_identity(output_tensor)
        if check_return_cond(output_tensor):
            return statistics

    def _get_switch_index(switch_op):
        for i in range(len(switch_op.outputs)):
            if len(switch_op.outputs[i].consumers()) > 0:
                return i
        raise ValueError("switch op's outputs don't have consumers")

    # batch norm with tf.cond
    output_no_grad_consumers = [c for c in output_tensor.consumers() if 'gradients/' not in c.name]

    if (len(output_no_grad_consumers) == 2) and (output_tensor.consumers()[0].type == 'Switch') and (output_tensor.consumers()[1].type == 'Switch'):
        switch_op1 = output_tensor.consumers()[0]
        switch_op2 = output_tensor.consumers()[1]

        index = _get_switch_index(switch_op1)
        no_grad_consumers = [c for c in switch_op1.outputs[index].consumers() if 'gradients/' not in c.name]
        cond1 = (len(no_grad_consumers) == 1)
        c1 = switch_op1.outputs[index].consumers()[0]
        cond2 = ('FusedBatchNorm' in c1.type)
        merge_op = c1.outputs[0].consumers()[0]
        cond3 = (merge_op.type == 'Merge')

        index = _get_switch_index(switch_op2)
        no_grad_consumers = [c for c in switch_op2.outputs[index].consumers() if 'gradients/' not in c.name]
        cond4 = (len(no_grad_consumers) == 1)
        c2 = switch_op2.outputs[index].consumers()[0]
        cond5 = ('FusedBatchNorm' in c2.type)
        cond6 = (len(merge_op.inputs) == 2)

        sub_cond1 = ((merge_op.inputs[0] == c1.outputs[0]) and (merge_op.inputs[1] == c2.outputs[0]))
        sub_cond2 = ((merge_op.inputs[0] == c2.outputs[0]) and (merge_op.inputs[1] == c1.outputs[0]))

        cond7 = (sub_cond1 or sub_cond2)

        final_cond = (cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7)

        if final_cond is True:
            bn_scope = merge_op.name.strip('cond/Merge') + "/"
            gamma_op = get_variable_by_name(bn_scope + "gamma")
            beta_op = get_variable_by_name(bn_scope + "beta")
            moving_mean_op = get_variable_by_name(bn_scope + "moving_mean")
            moving_variance_op = get_variable_by_name(bn_scope + "moving_variance")

            statistics = []
            statistics.append(tf.identity(gamma_op))
            statistics.append(tf.identity(beta_op))
            statistics.append(tf.identity(moving_mean_op))
            statistics.append(tf.identity(moving_variance_op))
            statistics.append(c1.get_attr("epsilon"))

            return statistics

    if 'FusedBatchNorm' in output_tensor.consumers()[0].type:
        c = output_tensor.consumers()[0]
        gamma_op = None
        beta_op = None
        bn_scope = None
        input_names = [t.op.name for t in c.inputs]
        for name in input_names:
            if name.endswith("gamma/read"):
                gamma_op = conv2d_op.graph.get_operation_by_name(name)
            if name.endswith("beta/read"):
                beta_op = conv2d_op.graph.get_operation_by_name(name)
                bn_scope = name.rsplit("beta/read", 1)[0]
        
        assert bn_scope is not None
        moving_mean_op = conv2d_op.graph.get_operation_by_name(bn_scope + "moving_mean/read")
        moving_variance_op = conv2d_op.graph.get_operation_by_name(bn_scope + "moving_variance/read")

        statistics = []
        if gamma_op is not None:
            statistics.append(gamma_op.outputs[0])
        else:
            statistics.append(gamma_op)
        statistics.append(beta_op.outputs[0])
        statistics.append(moving_mean_op.outputs[0])
        statistics.append(moving_variance_op.outputs[0])
        statistics.append(c.get_attr("epsilon"))
        
        return statistics

    return statistics

def get_save_tensor_name(name):
    if name.endswith(':0'):
        return name[:-2]
    else:
        return name

def get_quant_weight_io_map(graph, file_name="quant_weight_io_map.npy", remove_name_str=["while/",]):
    grad_ops = check_for_grad_ops(graph)
    if grad_ops:
        raise ValueError('gradient op found in graph, exiting %s\nplease invoke with inference graph only. create quantizer before construct model optimizer\n' % grad_ops)

    supported_types = ["Conv2D", "DepthwiseConv2dNative", "MatMul"]
    all_ops = graph.get_operations()
    quant_ops = [op for op in all_ops if op.type in supported_types]

    weight_io_map = {}

    for op in quant_ops:
        op_name = op.name
        if 'gradients/' in op_name and '_grad' in op_name:
            continue
        
        if op.type in ["Conv2D", "DepthwiseConv2dNative"]:
            weight_name = get_save_tensor_name(get_variable_by_tensor(op.inputs[1]).name)
            input_name = get_save_tensor_name(op.inputs[0].name)
            output_name = get_save_tensor_name(find_quant_output_tensor(op).name)

            for remove_str in remove_name_str:
                weight_name = weight_name.replace(remove_str, "")
                input_name = input_name.replace(remove_str, "")
                output_name = output_name.replace(remove_str, "")
            
            weight_io_map[weight_name] = (input_name, output_name, op_name)
        
        if op.type == "MatMul":
            if is_weight_tensor(op.inputs[0]) and is_weight_tensor(op.inputs[1]):
                continue
            if (not is_weight_tensor(op.inputs[0])) and (not is_weight_tensor(op.inputs[1])):
                continue
            
            weight_name, input_name, output_name = None, None, None

            for input_tensor in op.inputs:
                if is_weight_tensor(input_tensor):
                    weight_name = get_save_tensor_name(get_variable_by_tensor(input_tensor).name)
                else:
                    input_name = get_save_tensor_name(input_tensor.name)
            
            assert (weight_name != None) and (input_name != None)

            output_name = get_save_tensor_name(find_quant_output_tensor(op).name)

            for remove_str in remove_name_str:
                weight_name = weight_name.replace(remove_str, "")
                input_name = input_name.replace(remove_str, "")
                output_name = output_name.replace(remove_str, "")
            
            weight_io_map[weight_name] = (input_name, output_name, op_name)

    np.save(file_name, weight_io_map)
    print("quant weight io map saved to:", file_name)
