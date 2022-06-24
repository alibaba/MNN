from __future__ import print_function, with_statement
import copy
import numpy as np
from onnx import numpy_helper
import torch
try:
    import torch.fx as fx
except ImportError:
    print("need torch version >= 1.8.0, try 'pip install -U torch torchvision torchaudio' to upgrade torch")
from torch.fx import symbolic_trace
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import onnx
import mnncompress.common.MNN_compression_pb2 as compress_pb
import mnncompress
from mnncompress.common.log import mnn_logger
from mnncompress.common.helper import get_pipeline_methods
import uuid


_Supported_Modules = (torch.nn.Conv2d, torch.nn.Linear)
_MNN_Quant_prefix = "mnn_quant_"
_eps = 1e-6


class _Conv2dWeightFakeQuantModule(nn.Module):
    def __init__(self, prune_mask, clamp_value = 127.0):
        super().__init__()
        self._clamp_value = clamp_value
        self._prune_mask = nn.Parameter(prune_mask, requires_grad=False)
    
    def forward(self, w):
        w.data = w.data * self._prune_mask
        abs_w = torch.abs(w)
        max3 = torch.max(abs_w, 3, keepdim=True)
        max2 = torch.max(max3.values, 2, keepdim=True)
        max1 = torch.max(max2.values, 1, keepdim=True)
        scale = max1.values / self._clamp_value + _eps

        quanted = w / scale
        rounded = torch.detach(torch.round(quanted + torch.sign(quanted)*_eps) - quanted) + quanted
        clamped = torch.clamp(rounded, -self._clamp_value, self._clamp_value)
        dequanted = clamped * scale

        return dequanted

class _LinearWeightFakeQuantModule(nn.Module):
    def __init__(self, prune_mask, clamp_value = 127.0):
        super().__init__()
        self._clamp_value = clamp_value
        self._prune_mask = nn.Parameter(prune_mask, requires_grad=False)
    
    def forward(self, w):
        w.data = w.data * self._prune_mask
        abs_w = torch.abs(w)
        max1 = torch.max(abs_w, 1, keepdim=True)
        scale = max1.values / self._clamp_value + _eps

        quanted = w / scale
        rounded = torch.detach(torch.round(quanted + torch.sign(quanted)*_eps) - quanted) + quanted
        clamped = torch.clamp(rounded, -self._clamp_value, self._clamp_value)
        dequanted = clamped * scale

        return dequanted


class WeightQuantizer(object):
    def __init__(self, model, bits = 8, debug_info = False, mode = 'symmetric'):
        self._model = copy.deepcopy(model)
        self._gm = symbolic_trace(model)

        nm = dict(model.named_modules())
        supported_nm = {n:m for n, m in nm.items() if isinstance(m, _Supported_Modules)}

        # self._skip_quant_types = skip_quant_types
        if bits < 2 or bits > 8:
            raise ValueError("bits must be a integer in [2, 8]")
        self._bits = bits
        self._clamp_value = float(pow(2, bits-1) - 1)
        # self._layer_weight_bits_config = layer_weight_bits_config
        self._debug = debug_info

        if mode not in ['symmetric', 'asymmetric']:
            raise ValueError("mode should be 'symmetric' or 'asymmetric'")
        # TODO: support asymmetric
        self._mode = mode

        self._pname_mask = {}
        self._quant_module_nodes = []
        self._quant_func_nodes = []
        self._init_prune_ratios = {}
        self._ori_prune_weights = {}
        self._weight_node_ori_args = {}
        self._quant_user_nodes_update_args = {}
        self._wq_ops_stripped = False
        self._weight_quant_info_map = {}
        self._reported = False
        self._total_weight_num = 0.0
        self._remain_weight_num = 0.0

        self._init()

    def save_compress_params(self, filename, append=False):
        nm = dict(self._gm.named_modules())
        named_params = dict(self._gm.named_parameters())

        compress_proto = compress_pb.Pipeline()
        if append:
            f = open(filename, 'rb')
            compress_proto.ParseFromString(f.read())

        compress_proto.version = "0.0.0"
        if compress_proto.mnn_uuid == '':
            self._guid = str(uuid.uuid4())
            compress_proto.mnn_uuid = self._guid
        else:
            self._guid = compress_proto.mnn_uuid

        if not self._reported:
            detail = {"algorithm": "WQ", "pipeline": get_pipeline_methods(compress_proto), "compression_rate": self._total_weight_num / self._remain_weight_num, \
                "ori_model_size": self._total_weight_num * 4.0 / 1024.0 / 1024.0, \
                "config": {"bits": self._bits, "init_prune_ratios": self._init_prune_ratios}}
            self._reported = mnn_logger.on_done("pytorch", self._guid, detail)

        f = open(filename, 'wb')
        f.write(compress_proto.SerializeToString())
        f.close()

        print("compress proto saved to:", filename)

    def strip_wq_ops(self):
        for node, index_weight_node in self._weight_node_ori_args.items():
            ori_args = list(node.args)
            for index, ori_node in index_weight_node:
                ori_args[index] = ori_node
            node.args = tuple(ori_args)

        self._gm.recompile()
        self._wq_ops_stripped = True

    def resume_wq_graph(self):
        if not self._wq_ops_stripped:
            return
        
        for node, new_args in self._quant_user_nodes_update_args.items():
            node.args = new_args
        
        self._gm.recompile()
        self._wq_ops_stripped = False

    def _current_prune_ratios(self):
        prune_ratios = {}
        for n, m in self._ori_prune_weights.items():
            ratio = 1 - np.mean(np.abs(m.cpu().numpy()) > _eps)
            prune_ratios[n] = ratio
        return prune_ratios

    def _init(self):
        for module_name, module in self._model.named_modules():
            if isinstance(module, _Supported_Modules):
                for n, p in module.named_parameters():
                    if 'weight' == n:
                        pname = module_name + "." + n
                        mask = (torch.abs(p.data) > _eps).float()
                        self._pname_mask[pname] = mask
                        self._ori_prune_weights[pname] = p.data
                        self._init_prune_ratios[pname] = 1 - torch.mean(mask).cpu().numpy().tolist()

        print("init pruning ratios:")
        for key, value in self._init_prune_ratios.items():
            print(key, ":", value)

        nm = dict(self._gm.named_modules())
        for node in self._gm.graph.nodes:
            if node.target in nm.keys() and isinstance(nm[node.target], _Supported_Modules):
                self._quant_module_nodes.append(node)

        for node in self._gm.graph.nodes:
            if node.target is F.conv2d or node.target is F.linear:
                self._quant_func_nodes.append(node)

        self._extract()

    def _extract(self):
        gm = self._gm
        nm = dict(gm.named_modules())
        for n in self._quant_module_nodes:
            if n.target in nm.keys() and isinstance(nm[n.target], torch.nn.Conv2d):
                conv_module = nm[n.target]
                with gm.graph.inserting_after(n):
                    w = gm.graph.get_attr(n.target+".weight")

                self._total_weight_num += dict(self._model.named_parameters())[n.target + ".weight"].data.numel()
                self._remain_weight_num += (dict(self._model.named_parameters())[n.target + ".weight"].data.numel() / (32.0 / self._bits) * (1 - self._init_prune_ratios[n.target + ".weight"]))
                
                p = w
                b = None
                if conv_module.bias is not None:
                    with gm.graph.inserting_after(w):
                        b = gm.graph.get_attr(n.target+".bias")
                        p = b
            
                pt = n.args[0]
                if conv_module.padding_mode != 'zeros':
                    with gm.graph.inserting_after(p):
                        pad = gm.graph.call_function(F.pad)
                        pad.args = (n.args[0],)
                        pad.kwargs = {'pad':conv_module._reversed_padding_repeated_twice, 'mode':conv_module.padding_mode}
                        p = pad
                        pt = pad

                with gm.graph.inserting_after(p):
                    conv_func = gm.graph.call_function(F.conv2d)
                    conv_func.args = (pt, w, b)
                    if conv_module.padding_mode != 'zeros':
                        conv_func.kwargs = {'stride': conv_module.stride, 'padding': _pair(0), 'dilation': conv_module.dilation, 'groups': conv_module.groups}
                    else:
                        conv_func.kwargs = {'stride': conv_module.stride, 'padding': conv_module.padding, 'dilation': conv_module.dilation, 'groups': conv_module.groups}
                
                n.replace_all_uses_with(conv_func)
                self._quant_func_nodes.append(conv_func)
                gm.graph.erase_node(n)

            if n.target in nm.keys() and isinstance(nm[n.target], torch.nn.Linear):
                linear_module = nm[n.target]
                with gm.graph.inserting_after(n):
                    w = gm.graph.get_attr(n.target+".weight")

                self._total_weight_num += dict(self._model.named_parameters())[n.target + ".weight"].data.numel()
                self._remain_weight_num += (dict(self._model.named_parameters())[n.target + ".weight"].data.numel() / (32.0 / self._bits) * (1 - self._init_prune_ratios[n.target + ".weight"]))

                p = w
                b = None
                if linear_module.bias is not None:
                    with gm.graph.inserting_after(w):
                        b = gm.graph.get_attr(n.target+".bias")
                        p = b

                with gm.graph.inserting_after(p):
                    linear_func = gm.graph.call_function(F.linear)
                    linear_func.args = (n.args[0], w, b)

                n.replace_all_uses_with(linear_func)
                self._quant_func_nodes.append(linear_func)
                gm.graph.erase_node(n)
        
        gm.recompile()

    def convert(self):
        for i, node in zip(range(len(self._quant_func_nodes)), self._quant_func_nodes):
            if node.target is F.conv2d or node.target is F.linear:
                self._quant_one_layer(node, _MNN_Quant_prefix + str(i) + "_" + node.name, node.target)
        
        self._gm.recompile()
        return self._gm

    def _quant_one_layer(self, node, prefix, type):
        input_node = node.args[0]
        weight_node = node.args[1]

        weight_node_user_num = 0
        for n in self._gm.graph.nodes:
            if n.op == "get_attr" and n.target == weight_node.target:
                weight_node_user_num += len(n.users.keys())
        if weight_node_user_num != 1:
            print("weight_node:", weight_node.target, "has", weight_node_user_num, "users, skip quant for this layer")
            return

        weight_scale_name = self._fake_quant_weight(weight_node, prefix + "_weight", type)

        self._weight_quant_info_map[weight_node.target] = [weight_scale_name]

    def _fake_quant_weight(self, weight_node, name, type):
        if _MNN_Quant_prefix in weight_node.name:
            return weight_node.name + ".scale"

        nps = dict(self._gm.named_parameters())
        if type is F.conv2d:
            p = nps[weight_node.target]
            self._gm.add_module(name, _Conv2dWeightFakeQuantModule(self._pname_mask[weight_node.target], self._clamp_value))

        if type is F.linear:
            p = nps[weight_node.target]
            self._gm.add_module(name, _LinearWeightFakeQuantModule(self._pname_mask[weight_node.target], self._clamp_value))

        with self._gm.graph.inserting_after(weight_node):
            quant_weight = self._gm.graph.call_module(name)
            quant_weight.args = (weight_node,)
        
        nodes_update_args = []
        for node in weight_node.users.keys():
            if _MNN_Quant_prefix not in node.name:
                new_args = tuple()
                if node not in self._weight_node_ori_args.keys():
                    self._weight_node_ori_args[node] = []
                for i in range(0, len(node.args)):
                    if node.args[i] == weight_node:
                        new_args += (quant_weight,)
                        self._weight_node_ori_args[node].append([i, weight_node])
                    else:
                        new_args += (node.args[i],)
                nodes_update_args.append((node, new_args))
                self._quant_user_nodes_update_args[node] = new_args

        for node, new_args in nodes_update_args:
            node.args = new_args
        
        return name + ".scale"
