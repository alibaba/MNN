from __future__ import print_function
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
import scipy
from tensorly.decomposition import partial_tucker
from mnncompress.common import VBMF
import mnncompress.common.MNN_compression_pb2 as compress_pb
from .utils import get_module_parameter_num
from mnncompress.common.log import mnn_logger
from mnncompress.common.helper import get_align_channels
import uuid


def low_rank_decompose(model, compress_params_file, skip_layers=[""], align_channels=8, in_place=False, tucker_minimal_ratio=0.25, reserved_singular_value_ratio=0.5, append=False):
    origin_params_num = get_module_parameter_num(model)
    decompose_model = model
    if not in_place:
        decompose_model = copy.deepcopy(model)

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
    
    def _decompose_module(module, name=""):
        for n, m in module.named_children():
            m_name = name + "." + n
            if name == "":
                m_name = n
            if not isinstance(m, (nn.Conv2d, nn.Linear)):
                _decompose_module(m, m_name)
            else:
                if m_name in skip_layers:
                    print("skip decomposition:", m_name)
                    continue

                if isinstance(m, nn.Conv2d) and m.groups == 1 and m.kernel_size != (1, 1):
                    weight = m.weight.data.detach().cpu().numpy()

                    if m.in_channels <= align_channels or m.out_channels <= align_channels:
                        print("skip tucker for:", m_name, "weight shape:", weight.shape)
                        continue

                    u0 = tl.base.unfold(weight, 0)
                    u1 = tl.base.unfold(weight, 1)
                    res0 = VBMF.EVBMF(u0)
                    res1 = VBMF.EVBMF(u1)
                    rank0 = get_align_channels(res0[1].shape[0], m.out_channels, align_channels, tucker_minimal_ratio)
                    rank1 = get_align_channels(res1[1].shape[1], m.in_channels, align_channels, tucker_minimal_ratio)
                    ranks = [rank0, rank1]

                    core, [last, first] = partial_tucker(weight, modes=[0, 1], rank=ranks, init='svd')
                    print("tucker for", m_name, ":", [m.in_channels, m.out_channels], "<===>", [core.shape[1], core.shape[0]], "ranks:", ranks)

                    has_bias = True
                    if m.bias is None:
                        has_bias = False

                    first_layer = nn.Conv2d(in_channels=first.shape[0], \
                            out_channels=first.shape[1], kernel_size=1,
                            stride=1, padding=0, bias=False)

                    core_layer = nn.Conv2d(in_channels=core.shape[1], \
                            out_channels=core.shape[0], kernel_size=m.kernel_size,
                            stride=m.stride, padding=m.padding, dilation=m.dilation,
                            bias=False)

                    last_layer = nn.Conv2d(in_channels=last.shape[1], \
                        out_channels=last.shape[0], kernel_size=1, stride=1,
                        padding=0, bias=has_bias)

                    if has_bias:
                        last_layer.bias.data = m.bias.data

                    first_layer.weight.data = torch.transpose(torch.Tensor(first.copy()), 1, 0).unsqueeze(-1).unsqueeze(-1)
                    last_layer.weight.data = torch.Tensor(last.copy()).unsqueeze(-1).unsqueeze(-1)
                    core_layer.weight.data = torch.Tensor(core.copy())

                    # first_bn = nn.BatchNorm2d(first_layer.out_channels)
                    # core_bn = nn.BatchNorm2d(core_layer.out_channels)
                    # last_bn = nn.BatchNorm2d(last_layer.out_channels)

                    decomposed_layers = [first_layer, core_layer, last_layer]
                    module.__setattr__(n, nn.Sequential(*decomposed_layers))

                if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.stride == (1, 1) and m.padding == (0, 0) and m.dilation == (1, 1) and m.groups == 1):
                    weight = m.weight.data.detach().cpu().numpy()
                    squeeze_shape = weight.squeeze().shape
                    if len(squeeze_shape) != 2:
                        print("skip svd for", m_name, "weight shape:", weight.shape)
                        continue

                    if squeeze_shape[0] <= align_channels or squeeze_shape[1] <= align_channels:
                        print("skip svd for", m_name, "weight shape:", weight.shape)
                        continue

                    u, s, v = scipy.linalg.svd(weight.squeeze())
                    singular_value_sum = np.sum(s)
                    n_dim = 1
                    temp_sum = 0.0
                    for i in range(0, s.size):
                        temp_sum += s[i]
                        n_dim = i+1
                        if temp_sum / singular_value_sum >= reserved_singular_value_ratio:
                            break
                    n_dim = get_align_channels(n_dim, s.size, align_channels)

                    has_bias = True
                    if m.bias is None:
                        has_bias = False

                    if isinstance(m, nn.Conv2d):
                        print("svd for", m_name, ":", [m.in_channels, m.out_channels], "<===>", [m.in_channels, n_dim, m.out_channels])
                        fc1_weight = (np.matmul(np.diag(s[0:n_dim]), v[0:n_dim, :])).reshape((n_dim, -1, 1, 1))
                        fc2_weight = u[:, 0:n_dim].reshape((-1, n_dim, 1, 1))
                        fc1 = nn.Conv2d(m.in_channels, n_dim, 1, bias=False)
                        fc2 = nn.Conv2d(n_dim, m.out_channels, 1, bias=has_bias)
                    else:
                        print("svd for", m_name, ":", [m.in_features, m.out_features], "<===>", [m.in_features, n_dim, m.out_features])
                        fc1_weight = np.matmul(np.diag(s[0:n_dim]), v[0:n_dim, :])
                        fc2_weight = u[:, 0:n_dim]
                        fc1 = nn.Linear(m.in_features, n_dim, bias=False)
                        fc2 = nn.Linear(n_dim, m.out_features, bias=has_bias)
                    
                    fc1.weight.data = torch.Tensor(fc1_weight.copy())
                    fc2.weight.data = torch.Tensor(fc2_weight.copy())
                    
                    if has_bias:
                        fc2.bias.data = m.bias.data
                    
                    decomposed_layers = [fc1, fc2]
                    module.__setattr__(n, nn.Sequential(*decomposed_layers))

    _decompose_module(decompose_model)

    decompose_model_params_num = get_module_parameter_num(decompose_model)

    detail = {"algorithm": "low_rank_decompose", "compression_rate": origin_params_num / decompose_model_params_num, \
        "ori_model_size": origin_params_num * 4.0 / 1024.0 / 1024.0, \
        "config": {"skip_layers": skip_layers, "align_channels": align_channels, "tucker_minimal_ratio": tucker_minimal_ratio, "reserved_singular_value_ratio": reserved_singular_value_ratio}}

    mnn_logger.on_done("pytorch", model_guid, detail)

    return decompose_model
