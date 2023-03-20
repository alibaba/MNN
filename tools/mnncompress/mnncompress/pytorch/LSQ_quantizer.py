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


class _FeatureFakeQuantModule(nn.Module):
    def __init__(self, clamp_value = 127.0, mode='online'):
        super().__init__()
        self.scale = nn.Parameter(torch.rand(1)[0])
        self._initialized = nn.Parameter(torch.BoolTensor([False]), requires_grad=False)
        self._clamp_value = clamp_value
        if mode not in ['online', 'offline']:
            raise ValueError("mode should be 'online' or 'offline'")
        self._mode = mode
        self._momentum = 0.99
        self.zero_point = nn.Parameter(torch.zeros(1)[0])
    
    def forward(self, x):
        if not self._initialized:
            if self._mode == 'online':
                if torch.distributed.is_initialized():
                    init_scale = torch.max(torch.abs(x)) / self._clamp_value + _eps
                    torch.distributed.all_reduce(init_scale, torch.distributed.ReduceOp.MAX)
                else:
                    init_scale = torch.max(torch.abs(x)) / self._clamp_value + _eps
                self.scale.data = init_scale.detach()
            else:
                self.scale.requires_grad = False
                self.zero_point.requires_grad = False
                self.min = torch.min(x)
                self.max = torch.max(x)
            self._initialized.data = torch.BoolTensor([True])
        
        if self._mode == "online":
            temp = self.scale / np.sqrt(x.numel() * self._clamp_value)
            s_scale = torch.detach(self.scale - temp) + temp
            quanted = x / s_scale
            rounded = torch.detach(torch.round(quanted + torch.sign(quanted)*_eps) - quanted) + quanted
            clamped = torch.clamp(rounded, -self._clamp_value, self._clamp_value)
            dequanted = clamped * s_scale
        else:
            inst_min = torch.minimum(torch.min(x), torch.Tensor([0.0])[0]).detach()
            inst_max = torch.maximum(torch.max(x), torch.Tensor([0.0])[0]).detach()
            scale = ((inst_max - inst_min) / (2 * self._clamp_value)).detach()
            zero_point = torch.round((0 - inst_min) / scale - self._clamp_value).detach()
            
            nudge_min = (-self._clamp_value - zero_point) * scale
            nudge_max = (self._clamp_value - zero_point) * scale
            nudge_min = torch.minimum(nudge_min, torch.Tensor([0.0])[0]).detach()
            nudge_max = torch.maximum(nudge_max, torch.Tensor([0.0])[0]).detach()

            self.min = self._momentum * self.min + (1 - self._momentum) * nudge_min
            self.max = self._momentum * self.max + (1 - self._momentum) * nudge_max
            self.min = torch.minimum(self.min, torch.Tensor([0.0])[0]).detach()
            self.max = torch.maximum(self.max, torch.Tensor([0.0])[0]).detach()

            self.scale.data = (self.max - self.min) / (2 * self._clamp_value)
            self.scale.detach()
            self.zero_point.data = torch.round((0 - self.min) / self.scale - self._clamp_value)
            self.zero_point.detach()

            temp = x / self.scale + self.zero_point
            quanted = torch.round(temp + torch.sign(temp)*_eps)
            clamped = torch.clamp(quanted, -self._clamp_value, self._clamp_value)
            dequanted = (clamped - self.zero_point) * self.scale

        return dequanted

class _Conv2dWeightFakeQuantModule(nn.Module):
    def __init__(self, w_num, prune_mask, clamp_value = 127.0):
        super().__init__()
        self.scale = nn.Parameter(torch.rand(w_num, 1, 1, 1))
        self._initialized = nn.Parameter(torch.BoolTensor([False]), requires_grad=False)
        self._clamp_value = clamp_value
        self._prune_mask = nn.Parameter(prune_mask, requires_grad=False)
    
    def forward(self, w):
        w.data = w.data * self._prune_mask

        if not self._initialized:
            abs_w = torch.abs(w)
            max3 = torch.max(abs_w, 3, keepdim=True)
            max2 = torch.max(max3.values, 2, keepdim=True)
            max1 = torch.max(max2.values, 1, keepdim=True)
            init_scale = max1.values / self._clamp_value + _eps
            self.scale.data = init_scale
            self._initialized.data = torch.BoolTensor([True])

        temp = self.scale / np.sqrt(w.numel() * self._clamp_value)
        s_scale = torch.detach(self.scale - temp) + temp
        quanted = w / s_scale
        rounded = torch.detach(torch.round(quanted + torch.sign(quanted)*_eps) - quanted) + quanted
        clamped = torch.clamp(rounded, -self._clamp_value, self._clamp_value)
        dequanted = clamped * s_scale

        return dequanted

class _LinearWeightFakeQuantModule(nn.Module):
    def __init__(self, w_num, prune_mask, clamp_value = 127.0):
        super().__init__()
        self.scale = nn.Parameter(torch.rand(w_num, 1))
        self._initialized = nn.Parameter(torch.BoolTensor([False]), requires_grad=False)
        self._clamp_value = clamp_value
        self._prune_mask = nn.Parameter(prune_mask, requires_grad=False)
    
    def forward(self, w):
        w.data = w.data * self._prune_mask

        if not self._initialized:
            abs_w = torch.abs(w)
            max1 = torch.max(abs_w, 1, keepdim=True)
            init_scale = max1.values / self._clamp_value + _eps
            self.scale.data = init_scale
            self._initialized.data = torch.BoolTensor([True])

        temp = self.scale / np.sqrt(w.numel() * self._clamp_value)
        s_scale = torch.detach(self.scale - temp) + temp
        quanted = w / s_scale
        rounded = torch.detach(torch.round(quanted + torch.sign(quanted)*_eps) - quanted) + quanted
        clamped = torch.clamp(rounded, -self._clamp_value, self._clamp_value)
        dequanted = clamped * s_scale

        return dequanted


class LSQQuantizer(object):
    def __init__(self, model, skip_quant_layers = [], bits = 8, debug_info = False, mode = 'online', retain_sparsity=False):
        self._model = copy.deepcopy(model)
        self._gm = symbolic_trace(model)

        nm = dict(model.named_modules())
        supported_nm = {n:m for n, m in nm.items() if isinstance(m, _Supported_Modules)}

        for name in skip_quant_layers:
            if name not in supported_nm.keys():
                raise ValueError("skip quant layer '" + name + "' is not found in model's named_modules. availables are:", [n for n in supported_nm.keys()])
        
        self._skip_quant_layers = skip_quant_layers

        # self._skip_quant_types = skip_quant_types
        if bits < 2 or bits > 8:
            raise ValueError("bits must be a integer in [2, 8]")
        self._bits = bits
        self._clamp_value = float(pow(2, bits-1) - 1)
        # self._layer_weight_bits_config = layer_weight_bits_config
        self._debug = debug_info

        if mode not in ['online', 'offline']:
            raise ValueError("mode should be 'online' or 'offline'")
        self._mode = mode
        self._retain_sparsity = retain_sparsity

        self._pname_mask = {}
        self._quant_module_nodes = []
        self._quant_func_nodes = []
        self._init_prune_ratios = {}
        self._ori_prune_weights = {}
        self._feature_node_ori_args = {}
        self._weight_node_ori_args = {}
        self._quant_user_nodes_update_args = {}
        self._qat_ops_stripped = False
        self._weight_quant_info_map = {}
        self._reported = False
        self._total_weight_num = 0.0
        self._remain_weight_num = 0.0

        self._init()

    def save_compress_params(self, onnx_inference_model_file, filename, append=False):
        nm = dict(self._gm.named_modules())
        named_params = dict(self._gm.named_parameters())

        onnx_model = onnx.load(onnx_inference_model_file)
        onnx_model_bn_fused = True
        for node in onnx_model.graph.node:
            if node.op_type == "Conv":
                output_name = node.output[0]
                output_nodes = self._find_onnx_output_nodes(output_name, onnx_model)
                if len(output_nodes) == 1:
                    if  output_nodes[0].op_type == "BatchNormalization":
                        onnx_model_bn_fused = False
                        break

        compress_proto = compress_pb.Pipeline()
        if append:
            f = open(filename, 'rb')
            compress_proto.ParseFromString(f.read())

            pop_index = []
            for i in range(len(compress_proto.algo)):
                if compress_proto.algo[i].type == compress_pb.CompressionAlgo.CompressionType.QUANTIZE:
                    pop_index.append(i)
            for i in pop_index:
                compress_proto.algo.pop(i)

        compress_proto.version = "0.0.0"
        if compress_proto.mnn_uuid == '':
            self._guid = str(uuid.uuid4())
            compress_proto.mnn_uuid = self._guid
        else:
            self._guid = compress_proto.mnn_uuid
        quant_algorithm = compress_proto.algo.add()

        for weight_name, info in self._weight_quant_info_map.items():
            used_bn_name = info[3]
            if onnx_model_bn_fused == False:
                used_bn_name = None

            input_name, output_name, bn_info = self._get_input_output_name_bn_info(onnx_model, weight_name, used_bn_name, False)
            assert (input_name is not None) and (output_name is not None), "input_name and output_name not found for key: " + weight_name
            
            l = quant_algorithm.quant_params.layer.add()

            input_params = compress_pb.LayerQuantizeParams.ActivationParams()
            input_params.name = input_name
            input_params.bits = self._bits
            input_params.scales.append(named_params[info[0][0]].detach().cpu().numpy().tolist())
            input_params.zero_point = int(named_params[info[0][1]].detach().cpu().numpy().tolist())
            input_params.clamp_min = -int(self._clamp_value)
            input_params.clamp_max = int(self._clamp_value)
            l.input.append(input_params)

            output_params = compress_pb.LayerQuantizeParams.ActivationParams()
            output_params.name = output_name
            output_params.bits = self._bits
            output_params.scales.append(named_params[info[1][0]].detach().cpu().numpy().tolist())
            output_params.zero_point = int(named_params[info[1][1]].detach().cpu().numpy().tolist())
            output_params.clamp_min = -int(self._clamp_value)
            output_params.clamp_max = int(self._clamp_value)
            l.output.append(output_params)

            weight_params = compress_pb.LayerQuantizeParams.WeightParams()
            weight_params.name = weight_name
            weight_params.bits = self._bits
            weight_scales = named_params[info[2]].flatten().detach().cpu().numpy()
            gamma = np.ones_like(weight_scales)
            var = np.ones_like(weight_scales)
            if info[3] is not None:
                bn = nm[info[3]]
                gamma = bn.weight.detach().cpu().numpy()
                var = bn.running_var.detach().cpu().numpy()
                weight_scales = weight_scales * gamma / np.sqrt(var + bn.eps)

            weight_scales = weight_scales.tolist()
            for ws in weight_scales:
                weight_params.scales.append(abs(ws))
            weight_params.clamp_min = -int(self._clamp_value)
            weight_params.clamp_max = int(self._clamp_value)
            l.weight.append(weight_params)

        if not self._reported:
            detail = {"algorithm": "LSQ", "pipeline": get_pipeline_methods(compress_proto), "compression_rate": self._total_weight_num / self._remain_weight_num, \
                "ori_model_size": self._total_weight_num * 4.0 / 1024.0 / 1024.0, \
                "config": {"bits": self._bits, "skip_quant_layers": self._skip_quant_layers, "init_prune_ratios": self._init_prune_ratios}}
            self._reported = mnn_logger.on_done("pytorch", self._guid, detail)

        f = open(filename, 'wb')
        f.write(compress_proto.SerializeToString())
        f.close()

        print("compress proto saved to:", filename)
    
    def _get_input_output_name_bn_info(self, onnx_model, weight_name, bn_name, from_onnx_changed):
        input_name = None
        output_name = None
        bn_info = None

        for node in onnx_model.graph.node:
            if weight_name in node.input:
                if node.op_type not in ["Conv", "Gemm", "MatMul"]:
                    raise ValueError("only quantize onnx Conv, Gemm, MatMul ops, but encountered op type:" + node.op_type)
                input_name = node.input[0]
                output_name, bn_info = self._get_onnx_output_name_bn_info(node, onnx_model)
                return input_name, output_name, bn_info

        if input_name == None and from_onnx_changed == False:
            input_name, output_name, bn_info = self._get_input_output_name_onnx_name_changed(onnx_model, weight_name, bn_name)

        return input_name, output_name, bn_info

    def _get_input_output_name_onnx_name_changed(self, onnx_model, weight_name, bn_name):
        input_name = None
        output_name = None
        bn_info = None

        nm = dict(self._gm.named_modules())

        named_params = dict(self._gm.named_parameters())
        weight = named_params[weight_name].detach().cpu().numpy()
        bn_stat_shape = [weight.shape[0]] + [1 for i in range(len(weight.shape)-1)]
        gamma = np.ones(bn_stat_shape).astype(weight.dtype)
        var = np.ones(bn_stat_shape).astype(weight.dtype)
        if bn_name is not None:
            bn = nm[bn_name]
            gamma = bn.weight.detach().cpu().numpy().reshape(bn_stat_shape)
            var = bn.running_var.detach().cpu().numpy().reshape(bn_stat_shape)
            weight = weight * (gamma / np.sqrt(var + bn.eps))

        for iz in onnx_model.graph.initializer:
            izv = numpy_helper.to_array(iz)
            node_type = None
            for node in onnx_model.graph.node:
                if iz.name in node.input:
                    node_type = node.op_type
            if node_type == "MatMul" and len(weight.shape) == 2:
                izv = izv.transpose((1, 0))
            if weight.shape == izv.shape:
                if np.mean(np.abs(weight - izv) < 1e-6) > 0.99:
                    input_name, output_name, bn_info = self._get_input_output_name_bn_info(onnx_model, iz.name, bn_name, True)
                    return input_name, output_name, bn_info

        return input_name, output_name, bn_info

    def _node_is_relu6(self, node, onnx_model):
        if node.op_type != "Clip":
            return False
        
        if len(node.input) == 1:
            clip_min, clip_max = None, None
            attrs = node.attribute
            for attr in attrs:
                if attr.name == "min":
                    clip_min = attr.f
                if attr.name == "max":
                    clip_max = attr.f
            
            if clip_min == 0.0 and clip_max == 6.0:
                return True
            else:
                return False
        
        if len(node.input) == 3:
            min_name, max_name = node.input[1], node.input[2]
            clip_min, clip_max = None, None
            for n in onnx_model.graph.node:
                if n.output[0] == min_name and n.op_type == "Constant":
                    clip_min = numpy_helper.to_array(n.attribute[0].t).tolist()
                if n.output[0] == max_name and n.op_type == "Constant":
                    clip_max = numpy_helper.to_array(n.attribute[0].t).tolist()

            if clip_min == None and clip_max == None:
                for iz in onnx_model.graph.initializer:
                    if iz.name == min_name:
                        clip_min = numpy_helper.to_array(iz).tolist()
                    if iz.name == max_name:
                        clip_max = numpy_helper.to_array(iz).tolist()
            
            if clip_min == 0.0 and clip_max == 6.0:
                return True
            else:
                return False

        return False

    def _get_onnx_output_name_bn_info(self, conv_or_gemm_node, onnx_model):
        assert len(conv_or_gemm_node.output) == 1, "conv or gemm node should only have one output"
        output_name = conv_or_gemm_node.output[0]
        bn_info = None
        bn_weight = None
        bn_var = None

        if conv_or_gemm_node.op_type == "Gemm":
            return output_name, bn_info
        
        output_nodes = self._find_onnx_output_nodes(output_name, onnx_model)
        if len(output_nodes) != 1:
            return output_name, bn_info

        # matmul add is Add
        if conv_or_gemm_node.op_type == "MatMul":
            if output_nodes[0].op_type == "Add":
                bias_name = output_nodes[0].input[0]
                weight_name = conv_or_gemm_node.input[1]
                w, b = None, None
                for iz in onnx_model.graph.initializer:
                    if iz.name == bias_name:
                        b = numpy_helper.to_array(iz)
                    if iz.name == weight_name:
                        w = numpy_helper.to_array(iz)

                if b.shape[0] in w.shape:
                    output_name = output_nodes[0].output[0]
                    output_nodes = self._find_onnx_output_nodes(output_name, onnx_model)
                    if len(output_nodes) != 1:
                        return output_name, bn_info
            # when onnx convert to mnn, matmul is converted to conv, but relu is not fused, so need return anyway.
            return output_name, bn_info

        if output_nodes[0].op_type not in ["BatchNormalization", "Relu"] and (not self._node_is_relu6(output_nodes[0], onnx_model)):
            return output_name, bn_info
        
        if output_nodes[0].op_type == "BatchNormalization":
            if not any([input_name.endswith(".running_var") for input_name in output_nodes[0].input]):
                return output_name, bn_info

            output_name = output_nodes[0].output[0]

            for iz in onnx_model.graph.initializer:
                if iz.name == output_nodes[0].input[1]:
                    bn_weight = numpy_helper.to_array(iz)
                if iz.name == output_nodes[0].input[4]:
                    bn_var = numpy_helper.to_array(iz)
        
        if bn_weight is not None and bn_var is not None:
            bn_info = [bn_weight, bn_var]

        output_nodes = self._find_onnx_output_nodes(output_name, onnx_model)
        if len(output_nodes) != 1:
            return output_name, bn_info
        if self._node_is_relu6(output_nodes[0], onnx_model):
            output_name = output_nodes[0].output[0]
            return output_name, bn_info

        if output_nodes[0].op_type == "Relu":
            output_name = output_nodes[0].output[0]
            output_nodes = self._find_onnx_output_nodes(output_name, onnx_model)
            if len(output_nodes) != 1:
                return output_name, bn_info
            # in onnx opset > 9, F.relu6 is composed by relu + clip
            if self._node_is_relu6(output_nodes[0], onnx_model):
                output_name = output_nodes[0].output[0]
                return output_name, bn_info
        
        return output_name, bn_info

    def _find_onnx_output_nodes(self, output_name, onnx_model):
        output_nodes = []
        for n in onnx_model.graph.node:
            if output_name in n.input:
                output_nodes.append(n)
        
        return output_nodes

    def strip_qat_ops(self):
        if self._qat_ops_stripped:
            return
        
        for node, index_feature_node in self._feature_node_ori_args.items():
            ori_args = list(node.args)
            for index, ori_node in index_feature_node:
                ori_args[index] = ori_node
            node.args = tuple(ori_args)
        
        for node, index_weight_node in self._weight_node_ori_args.items():
            ori_args = list(node.args)
            for index, ori_node in index_weight_node:
                ori_args[index] = ori_node
            node.args = tuple(ori_args)

        self._gm.recompile()
        self._qat_ops_stripped = True

    def resume_qat_graph(self):
        if not self._qat_ops_stripped:
            return
        
        for node, new_args in self._quant_user_nodes_update_args.items():
            node.args = new_args
        
        self._gm.recompile()
        self._qat_ops_stripped = False

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
                        if self._retain_sparsity:
                            th = _eps
                        else:
                            th = -1.0 # so that mask are all 1.0
                        mask = (torch.abs(p.data) > th).float()
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

        for module_name in self._skip_quant_layers:
            if module_name in [n.target for n in self._gm.graph.nodes]:
                self._total_weight_num += dict(self._model.named_parameters())[module_name + ".weight"].data.numel()
                self._remain_weight_num += dict(self._model.named_parameters())[module_name + ".weight"].data.numel()
                rn = {n.target : n for n in self._gm.graph.nodes}[module_name]
                self._quant_module_nodes.remove(rn)
        
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

        real_input_node = self._find_real_input(input_node)
        input_scale_name = self._fake_quant_feature(real_input_node, input_node, prefix + "_input")
        weight_scale_name = self._fake_quant_weight(weight_node, prefix + "_weight", type)
        output_node, bn_name = self._find_output_node(node)
        output_scale_name = self._fake_quant_feature(output_node, output_node, prefix + "_output")

        self._weight_quant_info_map[weight_node.target] = [input_scale_name, output_scale_name, weight_scale_name, bn_name]

    def _fake_quant_feature(self, feature_node, ori_node, name):
        if _MNN_Quant_prefix in feature_node.name:
            return [feature_node.name + ".scale", feature_node.name + ".zero_point"]

        self._gm.add_module(name, _FeatureFakeQuantModule(self._clamp_value, self._mode))

        with self._gm.graph.inserting_after(ori_node):
            quant_input = self._gm.graph.call_module(name)
            quant_input.args = (ori_node,)

        nodes_update_args = []
        for node in ori_node.users.keys():
            if _MNN_Quant_prefix not in node.name:
                new_args = tuple()
                if node not in self._feature_node_ori_args.keys():
                    self._feature_node_ori_args[node] = []
                for i in range(0, len(node.args)):
                    if node.args[i] == ori_node:
                        new_args += (quant_input,)
                        self._feature_node_ori_args[node].append([i, ori_node])
                    else:
                        new_args += (node.args[i],)
                nodes_update_args.append((node, new_args))
                self._quant_user_nodes_update_args[node] = new_args
        
        for node, new_args in nodes_update_args:
            node.args = new_args

        return [name + ".scale", name + ".zero_point"]

    def _fake_quant_weight(self, weight_node, name, type):
        if _MNN_Quant_prefix in weight_node.name:
            return weight_node.name + ".scale"

        nps = dict(self._gm.named_parameters())
        if type is F.conv2d:
            p = nps[weight_node.target]
            self._gm.add_module(name, _Conv2dWeightFakeQuantModule(p.shape[0], self._pname_mask[weight_node.target], self._clamp_value))

        if type is F.linear:
            p = nps[weight_node.target]
            self._gm.add_module(name, _LinearWeightFakeQuantModule(p.shape[0], self._pname_mask[weight_node.target], self._clamp_value))

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

    def _find_real_input(self, input_node):
        gm = self._gm
        nm = dict(gm.named_modules())
        real_input_node = input_node

        is_dropout =  input_node.target is torch.dropout or input_node.target is F.dropout
        is_dropout = is_dropout or (input_node.target in nm.keys() and isinstance(nm[input_node.target], nn.Dropout))
        if is_dropout:
            real_input_node = input_node.args[0]

        return real_input_node

    def _find_output_node(self, conv_or_linear_node):
        gm = self._gm
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
