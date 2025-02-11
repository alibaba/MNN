import os
import sys
import json
import torch
import numpy as np

from tqdm import tqdm
from .spinner import spinner_run
from .gptq import GPTQ
from .lora import LoRA

EXPORT_LOG = '.export.log'

class MNNConveter:
    def __init__(self, onnx_path, weight_ops, config):
        self.weight_ops = weight_ops
        self.config = config
        self.quant_block = config.args.quant_block
        self.quant_bit = config.args.quant_bit
        self.lm_quant_bit = config.args.lm_quant_bit
        self.symmetric = config.args.sym
        self.mnn_weight_offset = 0
        self.onnx_model_path = onnx_path
        self.mnn_name = os.path.basename(onnx_path).replace('.onnx', '.mnn')
        self.mnn_model_path = os.path.join(config.args.dst_path, self.mnn_name)
        self.mnn_weight_path = f'{self.mnn_model_path}.weight'
        if os.path.exists(config.args.mnnconvert):
            self.mnnconvert = config.args.mnnconvert
        else:
            self.mnnconvert = None

    def convert(self, convert_args):
        sfd = os.dup(1)
        log_fp = open(EXPORT_LOG, "a")
        log_fd = log_fp.fileno()
        # mnnconvert ... > .export.log
        os.dup2(log_fd, 1)
        try:
            sys.argv = convert_args
            sys.argc = len(convert_args)
            if self.mnnconvert is None:
                from MNN.tools import mnnconvert
                mnnconvert.main()
            else:
                convert_args[0] = self.mnnconvert
                cmd = ' '.join(convert_args)
                message = os.popen(cmd).read()
                print(message)
            sys.argv = []
        finally:
            os.dup2(sfd, 1)
            os.close(log_fd)

    @spinner_run(f'convert onnx model to ')
    def onnx2mnn(self, onnx_path, mnn_path, args = []):
        convert_args = [
            '',
            '-f',
            'ONNX',
            '--modelFile',
            str(onnx_path),
            '--MNNModel',
            str(mnn_path),
            '--transformerFuse',
            '--allowCustomOp'
        ]
        convert_args += args
        self.convert(convert_args)
        return mnn_path

    def mnn2json(self, mnn_path, json_path):
        convert_args = [
            '',
            '-f',
            'MNN',
            '--modelFile',
            str(mnn_path),
            '--JsonFile',
            str(json_path)
        ]
        self.convert(convert_args)
        return json_path

    def json2mnn(self, json_path, mnn_path):
        convert_args = [
            '',
            '-f',
            'JSON',
            '--modelFile',
            str(json_path),
            '--MNNModel',
            str(mnn_path)
        ]
        self.convert(convert_args)
        return mnn_path

    def removeDupOps(self, mnn_path):
        convert_args = [
            '',
            '-f',
            'MNN',
            '--modelFile',
            str(mnn_path),
            '--MNNModel',
            str(mnn_path),
            '--optimizeLevel=1'
        ]
        self.convert(convert_args)
        return mnn_path

    def export(self, quant_bit = None, quant_block = None):
        if self.weight_ops is None:
            if quant_bit is None:
                quant_bit = self.quant_bit
            if quant_block is None:
                quant_block = self.quant_block
            if quant_bit == 16:
                quant_args = ['--fp16']
            else:
                quant_args = [
                    '--weightQuantBits',
                    str(quant_bit),
                    '--weightQuantBlock',
                    str(quant_block)
                ]
            self.onnx2mnn(self.onnx_model_path, self.mnn_model_path, quant_args)
        else:
            mnn_json = f'{self.mnn_model_path}.json'
            self.onnx2mnn(self.onnx_model_path, self.mnn_model_path)
            self.mnn2json(self.mnn_model_path, mnn_json)
            self.rebuild(mnn_json)
            self.json2mnn(mnn_json, self.mnn_model_path)
            self.removeDupOps(self.mnn_model_path)
            if self.config.args.gptq_path is not None:
                self.apply_gptq(mnn_json)
            if self.config.args.lora_path is not None and self.config.args.lora_split:
                 self.export_lora(mnn_json)

    @spinner_run(f'apply gptq to ')
    def apply_gptq(self, mnn_json):
        GPTQ(self.config.args.gptq_path).apply(mnn_json, self.mnn_weight_path)
        return self.mnn_weight_path

    @spinner_run(f'export split lora to ')
    def export_lora(self, mnn_json):
        lora_model = os.path.join(self.config.args.dst_path, 'lora.mnn')
        lora_json = f'{lora_model}.json'
        LoRA(self.config.args.lora_path).apply(mnn_json, lora_json)
        self.json2mnn(lora_json, lora_model)
        if os.path.exists(lora_json):
            os.remove(lora_json)
        return lora_model

    @spinner_run(f'quant model weight to ', True)
    def rebuild(self, json_path):
        mnn_graph = json.load(open(json_path, 'rt'))
        new_ops = []
        with open(self.mnn_weight_path, 'wb') as self.mnn_weight:
            for op in tqdm(mnn_graph['oplists'], 'Quant weights'):
                if op['type'] == 'Extra':
                    new_ops += self.rebuild_op(op, mnn_graph)
                else:
                    new_ops.append(op)
        mnn_graph['oplists'] = new_ops
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(mnn_graph, file, ensure_ascii=False, indent=4)
        return self.mnn_weight_path

    def quant(self, weight, quant_bit, quant_block, symmetric):
        if torch.cuda.is_available():
            weight = weight.cuda()
        if torch.backends.mps.is_available():
            weight = weight.to('mps')
        oc, ic = weight.shape
        if quant_block == 0:
            block_size = ic
        else:
            block_size = quant_block
        while ic % block_size != 0:
            block_size /= 2
        block_size = int(block_size)
        block_num = ic // block_size
        weight = weight.reshape(oc, block_num, block_size)
        offset = 1 << (quant_bit - 1)
        clip_max = offset - 1
        if symmetric:
            clip_min = -clip_max
            abs_max, _ = torch.max(torch.abs(weight), axis=-1, keepdims=True)
            scale = abs_max / clip_max
            q_weight = torch.round(weight / scale)
            q_weight = (torch.clamp(q_weight.flatten(), clip_min, clip_max) + offset).to(torch.uint8)
            alpha = scale.flatten()
        else:
            clip_min = -offset
            max_val, _ = torch.max(weight, axis=-1, keepdims=True)
            min_val, _ = torch.min(weight, axis=-1, keepdims=True)
            scale = (max_val - min_val) / (clip_max - clip_min)

            if self.config.args.awq:
                q_weight = torch.round(weight / scale) - torch.round(min_val / scale) + clip_min
                zeros =  (torch.round(min_val / scale) - clip_min) * scale
            else:
                q_weight = torch.round((weight - min_val) / scale) + clip_min
                zeros =  min_val - scale * clip_min
            q_weight = (torch.clamp(q_weight.flatten(), clip_min, clip_max) + offset).to(torch.uint8)
            alpha = torch.stack([zeros.flatten(), scale.flatten()], axis=-1).flatten()

        q_weight = q_weight.reshape(-1, 2)
        if quant_bit == 4:
            q_weight = q_weight[:, 0] * 16 + q_weight[:, 1]

        # support for `MNN >= 2.9.6`
        clip_min = 1

        if q_weight.device is not torch.device('cpu'):
            return q_weight.cpu(), alpha.float().cpu(), clip_min
        return q_weight, alpha.float(), clip_min

    def write_weight(self, data):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        return self.mnn_weight.write(data.tobytes())

    def write_header(self, ic, oc, quant_bit):
        dim_num = self.mnn_weight.write(b'\x02')
        shape_dtype = np.int16
        if oc > 65535 or ic > 65535:
            shape_dtype = np.int32
        dim_length = self.write_weight(np.array([oc, ic]).astype(shape_dtype))
        offset = 1 << (quant_bit - 1)
        weight_map = [i for i in range(-offset, offset)]
        if len(weight_map) == 256:
            weight_map.insert(0, 0)
        else:
            weight_map.insert(0, len(weight_map))
        map_length = self.write_weight(np.array(weight_map, dtype=np.int8))
        header_length = dim_num + dim_length + map_length
        return header_length, shape_dtype == np.int32

    def build_weight(self, linear, quant_bit, quant_block, symmetric):
        ic, oc = linear.in_features, linear.out_features
        if quant_bit == 16:
            half_weight = linear.weight.data.flatten().half()
            weight_len = self.write_weight(half_weight)
            alpha_len, q_min, shape_int32, header_len = 0, 0, False, 0
        else:
            assert(quant_bit in (4, 8))
            q_weight, alpha, q_min = self.quant(linear.weight.data, quant_bit, quant_block, symmetric)
            header_len, shape_int32 = self.write_header(ic, oc, quant_bit)
            weight_len = self.write_weight(q_weight) + header_len
            alpha_len = self.write_weight(alpha)
        if linear.bias is not None:
            bias = linear.bias.data.flatten().float()
            bias_length = self.write_weight(bias)
        else:
            bias_length = 0
            # bias = np.zeros([oc], dtype=np.float32)
            # bias_length = self.write_weight(bias)
        external = [self.mnn_weight_offset, weight_len, alpha_len, bias_length, 0]
        self.mnn_weight_offset += (weight_len + alpha_len + bias_length)
        return external, q_min, shape_int32, header_len

    def build_tensor(self, graph, tensor_name):
        tensor_idx = [len(graph['tensorName'])]
        graph['tensorName'].append(tensor_name)
        return tensor_idx

    def rebuild_op(self, op, graph):
        op_type = op['main']['type']
        if op_type == 'FakeLinear':
            return self.rebuild_linear(op, graph)
        if op_type == 'FusedAttention':
            return self.rebuild_attnention(op, graph)

    def rebuild_attnention(self, op, graph):
        attrs = op['main']['attr']
        for attr in attrs:
            if attr['key'] == 'name':
                name = attr['s']
        origin_input = op['inputIndexes']
        origin_output = op['outputIndexes']
        fused_attention = {
            "inputIndexes": origin_input,
            "main_type": "AttentionParam",
            "main": { "kv_cache": True },
            "name": name,
            "outputIndexes": origin_output,
            "type": "Attention",
            "defaultDimentionFormat": "NHWC"
        }
        return [fused_attention]

    def rebuild_linear(self, op, graph):
        attrs = op['main']['attr']
        for attr in attrs:
            if attr['key'] == 'name':
                name = attr['s']
            elif attr['key'] == "in_features":
                ic = attr["i"]
            elif attr['key'] == "out_features":
                oc = attr["i"]
            elif attr['key'] == "has_bias":
                has_bias = attr["i"]
        linear = self.weight_ops[name]
        assert(linear.in_features == ic and
               linear.out_features == oc and
               (linear.bias is not None) == has_bias)

        is_lm = 'lm_head' in name
        quant_bit = self.lm_quant_bit if is_lm else self.quant_bit
        block_size = ic if self.quant_block == 0 else self.quant_block
        external, q_min, shape_int32, header_len = self.build_weight(linear, quant_bit, self.quant_block, self.symmetric)
        if is_lm and self.config.tie_word_embeddings:
            weight_offset = external[0] + header_len
            alpha_offset = external[0] + external[1]
            alpha_size = external[2]
            self.config.llm_config['tie_embeddings'] = [weight_offset, alpha_offset, alpha_size, quant_bit, self.quant_block]

        origin_input = op['inputIndexes']
        origin_output = op['outputIndexes']
        # build new tensor
        pre_reshape_name = f'{name}/pre_reshape'
        pre_convert_name = f'{name}/pre_convert'
        conv_name = name
        post_convert_name = f'{name}/post_convert'
        post_reshape_name = f'{name}/post_reshape'
        pre_reshape_output = self.build_tensor(graph, pre_reshape_name)
        pre_convert_output = self.build_tensor(graph, pre_convert_name)
        conv_output = self.build_tensor(graph, conv_name)
        post_convert_output = self.build_tensor(graph, post_convert_name)
        # [batch, seq, hidden_size_i] -[Linear] -> [batch, seq, hidden_size_o]
        # [1, seq, hidden_size_i] ->[Reshape]-> [seq, hidden_size_i, 1, 1]
        # -[Convert]-[Convolution]-[Convert]-> [Reshape] -> [1, seq, hidden_size_o]
        pre_reshape = {
            "name": pre_reshape_name,
            "type": "Reshape",
            "inputIndexes": origin_input,
            "outputIndexes": pre_reshape_output,
            "main_type": "Reshape",
            "main": {
                "dims": [-1, ic, 1, 1],
                "dimType": "NCHW"
            },
            "defaultDimentionFormat": "NHWC"
        }
        pre_convert = {
            "name": pre_convert_name,
            "inputIndexes": pre_reshape_output,
            "outputIndexes": pre_convert_output,
            "type": "ConvertTensor",
            "main_type": "TensorConvertInfo",
            "main": {
                "source": "NCHW",
                "dest": "NC4HW4"
            },
            "defaultDimentionFormat": "NHWC"
        }

        if quant_bit == 16:
            quanParameter = { "type": 3 }
        else:
            if self.symmetric:
                aMin = 0
                readType = 0
            else:
                aMin = q_min
                readType = oc * (ic // block_size)

            quanParameter = {
                "quantScale": 1.0, "scaleIn": 0.0, "scaleOut": 0.0,
                "useInt32": False, "has_scaleInt": False, "shapeInt32": shape_int32,
                "type": 1, "aMax": 0, "aMin": aMin, "readType": readType, "weightSize": 0
            }
        conv_op = {
            "name": conv_name,
            "inputIndexes": pre_convert_output,
            "outputIndexes": conv_output,
            "type": "Convolution",
            "main_type": "Convolution2D",
            "main": {
                'common': {
                    'dilateX': 1, 'dilateY': 1, 'strideX': 1, 'strideY': 1,
                    'kernelX': 1, 'kernelY': 1, 'padX': 0, 'padY': 0, 'group': 1,
                    'outputCount': oc, 'relu': False, 'padMode': 'CAFFE',
                    'relu6': False, 'inputCount': ic, 'hasOutputShape': False
                },
                "quanParameter": quanParameter,
                "external": external
            },
            "defaultDimentionFormat": "NHWC"
        }
        post_convert = {
            "name": post_convert_name,
            "inputIndexes": conv_output,
            "outputIndexes": post_convert_output,
            "type": "ConvertTensor",
            "main_type": "TensorConvertInfo",
            "main": {
                "source": "NC4HW4",
                "dest": "NCHW"
            },
            "defaultDimentionFormat": "NHWC"
        }
        post_reshape = {
            "name": post_reshape_name,
            "type": "Reshape",
            "inputIndexes": post_convert_output,
            "outputIndexes": origin_output,
            "main_type": "Reshape",
            "main": {
                "dims": [1, -1, oc],
                "dimType": "NCHW"
            },
            "defaultDimentionFormat": "NHWC"
        }
        return [pre_reshape, pre_convert, conv_op, post_convert, post_reshape]