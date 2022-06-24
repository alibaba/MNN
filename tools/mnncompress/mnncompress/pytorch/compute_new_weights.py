"""
Functions used to contract ExpandNets back to SmallNet

- from_expandnet_cl_to_snet: from ExpandNet-CL/ExpandNet-FC/ExpandNet-CL+FC to SmallNet
- from_expandnet_ck_to_snet: from ExpandNet-CK/ExpandNet-FC/ExpandNet-CK+FC to SmallNet

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_cl(s_1, s_2):
    """
    Compute weights from s_1 and s_2
    :param s_1: 1*1 conv layer
    :param s_2: 3*3 conv layer
    :return: new weight and bias
    """
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    w_s_1_ = w_s_1.view(w_s_1.size(0), w_s_1.size(1) * w_s_1.size(2) * w_s_1.size(3))
    w_s_2_tmp = w_s_2.view(w_s_2.size(0), w_s_2.size(1),  w_s_2.size(2) * w_s_2.size(3))

    new_weight = torch.Tensor(w_s_2.size(0), w_s_1.size(1), w_s_2.size(2)*w_s_2.size(3)).to(w_s_1.device)
    for i in range(w_s_2.size(0)):
        tmp = w_s_2_tmp[i, :, :].view( w_s_2.size(1),  w_s_2.size(2) * w_s_2.size(3))
        new_weight[i, :, :] = torch.matmul(w_s_1_.t(), tmp)
    new_weight = new_weight.view(w_s_2.size(0), w_s_1.size(1),  w_s_2.size(2), w_s_2.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, b_s_1) + b_s_2  # with bias
    elif b_s_1 is None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = None

    return {'weight': new_weight, 'bias': new_bias}


def compute_cl_2(s_1, s_2):
    """
    compute weights from former computation and last 1*1 conv layer
    :param s_1: 3*3 conv layer
    :param s_2: 1*1 conv layer
    :return: new weight and bias
    """
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    w_s_1_ = w_s_1.view(w_s_1.size(0), w_s_1.size(1),  w_s_1.size(2) * w_s_1.size(3))
    w_s_2_ = w_s_2.view(w_s_2.size(0), w_s_2.size(1) * w_s_2.size(2) * w_s_2.size(3))
    new_weight_ = torch.Tensor(w_s_2.size(0), w_s_1.size(1), w_s_1.size(2)*w_s_1.size(3)).to(w_s_1.device)
    for i in range(w_s_1.size(1)):
        tmp = w_s_1_[:, i, :].view(w_s_1.size(0),  w_s_1.size(2) * w_s_1.size(3))
        new_weight_[:, i, :] = torch.matmul(w_s_2_, tmp)
    new_weight = new_weight_.view(w_s_2.size(0), w_s_1.size(1),  w_s_1.size(2), w_s_1.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        new_bias = torch.matmul(w_s_2_, b_s_1) + b_s_2  # with bias
    elif b_s_1 is None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = None

    return {'weight': new_weight, 'bias': new_bias}

def compute_ck(s_1, s_2):
    """
    Compute weight from 2 conv layers, whose kernel size larger than 3*3
    After derivation, F.conv_transpose2d can be used to compute weight of original conv layer
    :param s_1: 3*3 or larger conv layer
    :param s_2: 3*3 or larger conv layer
    :return: new weight and bias
    """
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    w_s_2_tmp = w_s_2.view(w_s_2.size(0), w_s_2.size(1), w_s_2.size(2) * w_s_2.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, b_s_1) + b_s_2
    elif b_s_1 is None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = None

    new_weight = F.conv_transpose2d(w_s_2, w_s_1)

    return {'weight': new_weight, 'bias': new_bias}


def compute_fc(s_1, s_2):
    """
    Compute weight from 2 fc layers
    :param s_1: p * n
    :param s_2: m * p
    :return: new weight m*n and bias
    """
    if isinstance(s_1, nn.Linear):
        w_s_1 = s_1.weight
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Linear):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    if b_s_1 is not None and b_s_2 is not None:
        new_bias = torch.matmul(w_s_2, b_s_1) + b_s_2
    elif b_s_1 is None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = None

    new_weight = torch.matmul(w_s_2, w_s_1)

    return {'weight': new_weight, 'bias': new_bias}

def from_expandnet_cl_identity_to_snet(Enet, Snet, exp_layer_names=['conv0'], identity_layer_names=[]):
    Edict = Enet.state_dict()
    Sdict = Snet.state_dict()
    Edict = {k: v for k, v in Edict.items() if (k in Sdict and k not in exp_layer_names)}

    Sdict.update(Edict)
    Snet.load_state_dict(Sdict)
    if exp_layer_names is not None:
        for m in exp_layer_names:
            print(m)
            conv1_1 = getattr(Enet, m)[0]
            conv1_2 = getattr(Enet, m)[1]
            conv1_3 = getattr(Enet, m)[2]
            if m.find('conv') != -1 or m.find('concat_out') != -1:
                tmp = compute_cl(conv1_1, conv1_2)
                tmp = compute_cl_2(tmp, conv1_3)
                print('passed')
            else:
                tmp = compute_fc(conv1_1, conv1_2)
                tmp = compute_fc(tmp, conv1_3)

            if m in identity_layer_names:
                tmp = fuse_identity(tmp)

            conv1 = getattr(Snet, m)
            if conv1.bias is not None:
                conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
            else:
                conv1.weight.data = tmp['weight']

    return Snet

def fuse_identity(conv):
    kernel = conv['weight']

    #in_channel is kernel size 1
    in_channels = kernel.size(1)
    kernel_size = kernel.size(2)
    #bias = conv.bias
    #kernel_value = np.zeros((in_channels, in_channels, 3, 3), dtype=np.float32)
    kernel_value = np.zeros((in_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
    for i in range(in_channels):
        kernel_value[i, i % in_channels, 1, 1] = 1
    #id_tensor = torch.from_numpy(kernel_value).to(conv.weight.device)
    id_tensor = torch.from_numpy(kernel_value).to(kernel.device)
    conv['weight'] = kernel + id_tensor

    return conv

def fuse_add_conv(s_1, s_2):
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    conv = dict()
    conv['weight'] = w_s_1 + w_s_2
    conv['bias'] = b_s_1 + b_s_2

    return conv


def from_expandnet_cl_to_snet(Enet, Snet, exp_layer_names=['conv0']):
    Edict = Enet.state_dict()
    Sdict = Snet.state_dict()
    Edict = {k: v for k, v in Edict.items() if (k in Sdict and k not in exp_layer_names)}

    Sdict.update(Edict)
    Snet.load_state_dict(Sdict)
    if exp_layer_names is not None:
        for m in exp_layer_names:
            print(m)
            conv1_1 = getattr(Enet, m)[0]
            conv1_2 = getattr(Enet, m)[1]
            conv1_3 = getattr(Enet, m)[2]
            if m.find('conv') != -1 or m.find('concat_out') != -1:
                tmp = compute_cl(conv1_1, conv1_2)
                tmp = compute_cl_2(tmp, conv1_3)
                print('passed')
            else:
                tmp = compute_fc(conv1_1, conv1_2)
                tmp = compute_fc(tmp, conv1_3)

            conv1 = getattr(Snet, m)
            if conv1.bias is not None:
                conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
            else:
                conv1.weight.data = tmp['weight']

    return Snet


def from_expandnet_ck_to_snet(Enet, Snet, exp_layer_names=['conv0']):
    Edict = Enet.state_dict()
    Sdict = Snet.state_dict()
    Edict = {k: v for k, v in Edict.items() if (k in Sdict and k not in exp_layer_names)}

    Sdict.update(Edict)
    Snet.load_state_dict(Sdict)

    if exp_layer_names is not None:
        for m in exp_layer_names:
            layers = getattr(Enet, m)
            tmp = layers[0]
            for i in range(len(layers)-1):
                conv1_2 = layers[i+1]
                if m.find('conv') != -1:
                    tmp = compute_ck(tmp, conv1_2)
                else:
                    tmp = compute_fc(tmp, conv1_2)

            conv1 = getattr(Snet, m)
            if conv1.bias is not None:
                conv1.weight.data, conv1.bias.data = tmp['weight'], tmp['bias']
            else:
                conv1.weight.data = tmp['weight']

    return Snet
