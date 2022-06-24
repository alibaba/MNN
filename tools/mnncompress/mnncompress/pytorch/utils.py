from __future__ import print_function
from numpy import isin
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

def find_conv_bn_module_pairs(model):
    gm = torch.fx.symbolic_trace(model)
    nodes = list(gm.graph.nodes)
    nm = dict(model.named_modules())

    conv_bn_pairs = {}

    for node in nodes:
        if node.target in nm.keys() and isinstance(nm[node.target], torch.nn.Conv2d):
            if len(node.users) == 1:
                user_node = [n for n in node.users.keys()][0]
                if user_node.target in nm.keys() and isinstance(nm[user_node.target], torch.nn.BatchNorm2d):
                    conv2d_name = node.target
                    bn_name = user_node.target
                    conv_bn_pairs[conv2d_name] = bn_name
    
    return conv_bn_pairs


def find_conv_or_linear_output_node(graph_module, conv_or_linear_node):
    gm = graph_module
    nm = dict(gm.named_modules())
    output_node = conv_or_linear_node
    bn_name = None

    # TODO: linear's bn or relu not supported
    if conv_or_linear_node.target is F.linear:
        return output_node, bn_name

    if len(output_node.users) > 1:
        return output_node, bn_name
    
    node = [n for n in output_node.users.keys()][0]
    has_bn = node.target in nm.keys() and isinstance(nm[node.target], torch.nn.BatchNorm2d)
    if has_bn:
        bn = nm[node.target]
        if bn.running_var is None:
            return output_node, bn_name
        
        output_node = node
        bn_name = node.target
        if len(output_node.users) > 1:
            return output_node, bn_name
    
    node = [n for n in output_node.users.keys()][0]
    has_relu = node.target is torch.relu or node.target is F.relu
    has_relu = has_relu or (node.op == "call_method" and node.target in ["relu", "relu_"])
    # TODO: mnn not support this relu6 yet
    # has_relu6 = node.target is F.relu6
    has_relu6 = False

    if node.op == "call_module":
        if isinstance(nm[node.target], nn.ReLU):
            has_relu = True

        if isinstance(nm[node.target], nn.ReLU6):
            has_relu6 = True

    if has_relu or has_relu6:
        output_node = node
        return output_node, bn_name

    return output_node, bn_name


def get_module_parameter_num(module):
    nm = dict(module.named_modules())
    count_types=(nn.Conv2d, nn.Linear)

    num_params = 0
    for n, m in nm.items():
        if isinstance(m, count_types):
            if m.weight is not None:
                num_params += m.weight.numel()
            if m.bias is not None:
                num_params += m.bias.numel()

    return num_params


chanel_prune_safe_ops = (nn.Conv2d, nn.Dropout)

def not_safe_to_prune_weights(model, safe_ops=chanel_prune_safe_ops):
    gm = torch.fx.symbolic_trace(model)
    nodes = list(gm.graph.nodes)
    nm = dict(model.named_modules())
    not_safe_weights = []

    for node in nodes:
        if node.target in nm.keys() and isinstance(nm[node.target], torch.nn.Conv2d):
            output_node, _ = find_conv_or_linear_output_node(gm, node)
            user_nodes = [n for n in output_node.users.keys()]
            for user_node in user_nodes:
                print("pairs:", node.target, user_node.target, user_node.target in nm.keys() and isinstance(nm[user_node.target], safe_ops))
                if not (user_node.target in nm.keys() and isinstance(nm[user_node.target], safe_ops)):
                    name = node.target + ".weight"
                    not_safe_weights.append(name)

        if node.target in nm.keys() and isinstance(nm[node.target], torch.nn.Linear):
            raise ValueError("don't support nn.Linear, please use (reshape(x, [-1, len, 1, 1]) --> 1x1 Conv2d --> ... --> flatten(x, 1)) combination instead")

    return not_safe_weights
