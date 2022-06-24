from __future__ import print_function, with_statement
import copy
import os
from numpy import add
import torch
import torch.nn as nn
import torch.nn.functional as F
from .compute_new_weights import compute_cl, compute_cl_2, fuse_identity
import mnncompress.common.MNN_compression_pb2 as compress_pb
from .utils import get_module_parameter_num
from mnncompress.common.log import mnn_logger
import uuid


class _ExpandLayerWithBypass(nn.Module):
    def __init__(self, conv2dm, expand_rate, add_batchnorm, add_bypass):
        super().__init__()
        self.first = nn.Conv2d(conv2dm.in_channels, conv2dm.in_channels*expand_rate, 1, padding=conv2dm.padding)
        self.middle = nn.Conv2d(conv2dm.in_channels*expand_rate, conv2dm.out_channels*expand_rate, conv2dm.kernel_size, stride=conv2dm.stride, dilation=conv2dm.dilation)
        self.last = nn.Conv2d(conv2dm.out_channels*expand_rate, conv2dm.out_channels, 1)
        self.with_bypass = False
        self.add_batchnorm = add_batchnorm
        self.add_bypass = add_bypass

        if add_batchnorm:
            self.first_bn = nn.BatchNorm2d(conv2dm.in_channels*expand_rate)
            self.middle_bn = nn.BatchNorm2d(conv2dm.out_channels*expand_rate)

    def forward(self, x_input):
        x = self.first(x_input)
        if self.add_batchnorm:
            x = self.first_bn(x)
        
        x = self.middle(x)
        if self.add_batchnorm:
            x = self.middle_bn(x)
        
        x = self.last(x)

        if self.add_bypass and (self.with_bypass or (x.shape == x_input.shape)):
            x = x + x_input
            self.with_bypass = True

        return x

class LOP(object):
    def __init__(self, model):
        self._expand_model = copy.deepcopy(model)
        self._merged_model = None
        self._module_expand_attr = {}
        self._expand_rate = 1
        self._add_bn = True
        self._add_bypass = True

    def linear_merge_layers(self):
        self._merged_model = copy.deepcopy(self._expand_model)
        nm = dict(self._merged_model.named_modules())
        for n, m in nm.items():
            if isinstance(m, _ExpandLayerWithBypass):
                conv_name = n.split(".")[-1]
                parent_module = nm[n[0:-(len(conv_name)+1)]]
                expanded_layer = parent_module.__getattr__(conv_name)
                first = expanded_layer.first
                middle = expanded_layer.middle
                last = expanded_layer.last
                with_bypass = expanded_layer.with_bypass

                if self._add_bn:
                    first_bn = expanded_layer.first_bn
                    middle_bn = expanded_layer.middle_bn

                    first.weight.data = first.weight * first_bn.weight.reshape((-1, 1, 1, 1)) / torch.sqrt(first_bn.running_var + first_bn.eps).reshape((-1, 1, 1, 1))
                    first.bias.data = (first.bias - first_bn.running_mean) * first_bn.weight / torch.sqrt(first_bn.running_var + first_bn.eps) + first_bn.bias
                    middle.weight.data = middle.weight * middle_bn.weight.reshape((-1, 1, 1, 1)) / torch.sqrt(middle_bn.running_var + middle_bn.eps).reshape((-1, 1, 1, 1))
                    middle.bias.data = (middle.bias - middle_bn.running_mean) * middle_bn.weight / torch.sqrt(middle_bn.running_var + middle_bn.eps) + middle_bn.bias

                merged_layer = nn.Conv2d(first.in_channels, last.out_channels, middle.kernel_size, stride=middle.stride, padding=first.padding, dilation=middle.dilation, bias=lambda: True if last.bias is not None else False)
                temp = compute_cl(first, middle)
                wb = compute_cl_2(temp, last)
                if with_bypass:
                    wb = fuse_identity(wb)
                merged_layer.weight.data = wb['weight']
                if merged_layer.bias is not None:
                    merged_layer.bias.data = wb['bias']

                parent_module.__setattr__(conv_name, merged_layer.to(first.weight.device))

        return self._merged_model

    def linear_expand_layers(self, expand_rate, compress_params_file, add_batchnorm=True, add_bypass=True, append=False):
        self._expand_rate = expand_rate
        self._add_bn = add_batchnorm
        self._add_bypass = add_bypass

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

        origin_params_num = get_module_parameter_num(self._expand_model)

        def _expand_module(module, name=""):
            for n, m in module.named_children():
                m_name = name + "." + n
                if name == "":
                    m_name = n
                if not isinstance(m, (nn.Conv2d, nn.Linear)):
                    _expand_module(m, m_name)
                else:
                    if isinstance(m, nn.Conv2d) and m.groups == 1 and m.kernel_size != (1, 1):
                        expanded_layer = _ExpandLayerWithBypass(m, expand_rate, add_batchnorm, add_bypass).to(m.weight.device)
                        module.__setattr__(n, expanded_layer)
                        self._module_expand_attr[module] = n
        
        _expand_module(self._expand_model)

        expand_model_params_num = get_module_parameter_num(self._expand_model)

        detail = {"algorithm": "linear_over_param", "compression_rate": expand_model_params_num / origin_params_num, \
            "expand_model_size": expand_model_params_num * 4.0 / 1024.0 / 1024.0, \
            "config": {"expand_rate": expand_rate, "add_batchnorm": add_batchnorm, "add_bypass": add_bypass}}

        mnn_logger.on_done("pytorch", model_guid, detail)

        return self._expand_model
