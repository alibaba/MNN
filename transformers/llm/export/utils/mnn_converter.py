import os
import sys
import copy
import json
import torch
import numpy as np

from .torch_utils import quant as torch_quant
from .torch_utils import onnx_export
from tqdm import tqdm
from .spinner import spinner_run
from .gptq import GPTQ
from .lora import LoRA

EXPORT_LOG = '.export.log'

class MNNConverter:
    def __init__(self, exporter, weight_ops = None):
        self.weight_ops = weight_ops
        self.exporter = exporter
        self.args = exporter.args
        self.mnn_weight_offset = 0
        if os.path.exists(self.args.mnnconvert):
            self.mnnconvert = self.args.mnnconvert
        else:
            self.mnnconvert = None
        self.lm_weight = None
        self.tie_embeddings_info = None

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
            log_fp.close()

    @spinner_run(f'convert onnx model to ')
    def onnx2mnn(self, onnx_path, mnn_path, args = [], transformer_fuse = True, group_conv_native = False, weight_sym = False, save_external_data = True):
        convert_args = [
            '',
            '-f',
            'ONNX',
            '--modelFile',
            str(onnx_path),
            '--MNNModel',
            str(mnn_path),
            '--allowCustomOp'
        ]
        if transformer_fuse:
            convert_args += ['--transformerFuse']
        if group_conv_native:
            convert_args += ['--groupConvNative']
        if weight_sym:
            convert_args += ['--weightQuantAsymmetric=0']
        if save_external_data:
            convert_args += ['--saveExternalData']
        if self.args.hqq:
            convert_args += ['--hqq']
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

    def export(self, onnx_path, quant_bit = None, quant_block = None, transformer_fuse = True, group_conv_native = False, weight_sym = None):
        self.onnx_model_path = onnx_path
        self.mnn_name = os.path.basename(onnx_path).replace('.onnx', '.mnn')
        self.mnn_model_path = os.path.join(self.args.dst_path, self.mnn_name)
        self.mnn_weight_path = f'{self.mnn_model_path}.weight'
        if self.weight_ops is None:
            if quant_bit is None:
                quant_bit = self.args.quant_bit
            if quant_block is None:
                quant_block = self.args.quant_block
            if weight_sym is None:
                weight_sym = self.args.sym
            if quant_bit == 16:
                quant_args = ['--fp16']
            else:
                quant_args = [
                    '--weightQuantBits',
                    str(quant_bit),
                    '--weightQuantBlock',
                    str(quant_block)
                ]
            if quant_bit == 32:
                quant_args = []
            self.onnx2mnn(self.onnx_model_path, self.mnn_model_path, quant_args, transformer_fuse=transformer_fuse, group_conv_native=group_conv_native, weight_sym=weight_sym)
        else:
            mnn_json = f'{self.mnn_model_path}.json'
            self.onnx2mnn(self.onnx_model_path, self.mnn_model_path, transformer_fuse=transformer_fuse, group_conv_native=group_conv_native, weight_sym=weight_sym)
            self.mnn2json(self.mnn_model_path, mnn_json)
            self.rebuild(mnn_json)
            self.json2mnn(mnn_json, self.mnn_model_path)
            self.removeDupOps(self.mnn_model_path)
            self.mnn2json(self.mnn_model_path, mnn_json)
            if self.args.gptq_path is not None:
                self.apply_gptq(mnn_json)
            if self.args.lora_path is not None and self.args.lora_split:
                 self.export_lora(mnn_json)
            if self.args.omni:
                self.export_omni_quant(mnn_json)
            if self.args.smooth:
                self.export_smooth_quant(mnn_json)
        return self.tie_embeddings_info

    def get_experts_graphs(self, experts):
        hidden_states = torch.randn((1, self.exporter.config.hidden_size))
        layers_num = len(experts)
        expert_num = len(experts[0])
        dummy_expert = experts[0][0]
        onnx_model = f'{self.exporter.onnx_path}/expert.onnx'
        onnx_export(
            dummy_expert, (hidden_states),
            onnx_model,
            input_names=['hidden_states'],
            output_names=['hidden_states'])
        mnn_model = f'{onnx_model}.mnn'
        mnn_json = f'{mnn_model}.json'
        self.onnx2mnn(onnx_model, mnn_model)
        self.mnn2json(mnn_model, mnn_json)
        expert_graph = json.load(open(mnn_json, 'rt'))
        tensors = expert_graph['tensorName']
        nodes = expert_graph['oplists']
        # get input and output
        inputs = []
        outputs = []
        for node in nodes:
            if node['type'] == 'Input':
                inputs.append(node['outputIndexes'][0])
        for output_name in expert_graph['outputName']:
            outputs.append(tensors.index(output_name))
        subgraphs = []
        for i in range(layers_num):
            for j in range(expert_num):
                ijnodes = copy.deepcopy(nodes)
                for op in ijnodes:
                    if op['type'] == 'Extra':
                        for attr in op['main']['attr']:
                            if attr['key'] == 'name':
                                names = attr['s'].split('/')
                                names[2] = f'{i}_{j}'
                                attr['s'] = '/'.join(names)
                subgraph = {
                    'name': f'/expert/{i}_{j}',
                    'inputs': inputs,
                    'outputs': outputs,
                    'tensors': copy.deepcopy(tensors),
                    'nodes': ijnodes
                }
                subgraphs.append(subgraph)
        return subgraphs


    @spinner_run(f'apply gptq to ')
    def apply_gptq(self, mnn_json):
        GPTQ(self.args.gptq_path).apply(mnn_json, self.mnn_weight_path)
        return self.mnn_weight_path

    @spinner_run(f'export split lora to ')
    def export_lora(self, mnn_json):
        lora_model = os.path.join(self.args.dst_path, 'lora.mnn')
        lora_json = f'{lora_model}.json'
        LoRA(self.args.lora_path).apply(mnn_json, lora_json)
        self.json2mnn(lora_json, lora_model)
        if os.path.exists(lora_json):
            os.remove(lora_json)
        return lora_model

    @spinner_run(f'export smooth quant scale to ')
    def export_smooth_quant(self, mnn_json):
        self.exporter.smooth_quantizer.apply(mnn_json)
        self.json2mnn(mnn_json, self.mnn_model_path)
        return self.mnn_model_path

    @spinner_run(f'export omni quant scale to ')
    def export_omni_quant(self, mnn_json):
        self.exporter.omni_quantizer.apply(mnn_json)
        self.json2mnn(mnn_json, self.mnn_model_path)
        return self.mnn_model_path

    @spinner_run(f'quant model weight to ', True)
    def rebuild(self, json_path):
        mnn_graph = json.load(open(json_path, 'rt'))
        has_experts = hasattr(self.exporter, 'experts') and len(self.exporter.experts) > 0
        if has_experts:
            subgraphs = self.get_experts_graphs(self.exporter.experts)
            mnn_graph['subgraphs'] = subgraphs
        new_ops = []
        # Load layernorm weight from external
        with open(self.mnn_weight_path, 'rb') as f:
            for op in tqdm(mnn_graph['oplists'], 'Load LayerNorm data'):
                if op['type'] == 'LayerNorm' and 'external' in op['main']:
                    external = op['main']['external']
                    f.seek(external[0])
                    op['main']['gamma'] = np.frombuffer(f.read(external[1]), np.float32).tolist()
                    op['main']['beta'] = np.frombuffer(f.read(external[2]), np.float32).tolist()
                    del op['main']['external']
                if op['type'] == 'Const' and 'external' in op['main']:
                    external = op['main']['external']
                    f.seek(external[0])
                    op['main']['float32s'] = np.frombuffer(f.read(external[1]), np.float32).tolist()
                    del op['main']['external']
        # Rebuild ops
        with open(self.mnn_weight_path, 'wb') as self.mnn_weight:
            for op in tqdm(mnn_graph['oplists'], 'Quant weights'):
                if op['type'] == 'Extra' or op['type'] == 'LayerNorm':
                    new_ops += self.rebuild_op(op, mnn_graph)
                else:
                    new_ops.append(op)
            mnn_graph['oplists'] = new_ops
            if has_experts and 'subgraphs' in mnn_graph:
                for subgraph in tqdm(mnn_graph['subgraphs'], 'Quant subgraphs weights'):
                    new_subops = []
                    for op in subgraph['nodes']:
                        if op['type'] == 'Extra' or op['type'] == 'LayerNorm':
                            new_subops += self.rebuild_op(op, subgraph)
                        else:
                            new_subops.append(op)
                    subgraph['nodes'] = new_subops
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(mnn_graph, file, ensure_ascii=False, indent=4)
        return self.mnn_weight_path

    def quant(self, weight, quant_bit, quant_block, symmetric):
        if self.exporter.args.skip_weight:
            # Skip expensive quantization when skip_weight is enabled
            oc, ic = weight.shape
            if quant_block == 0:
                block_size = ic
            else:
                block_size = quant_block
            block_num = ic // block_size
            # alpha: oc * block_num (symmetric) or oc * block_num * 2 (asymmetric)
            alpha_num = oc * block_num * (1 if symmetric else 2)
            alpha = torch.zeros(alpha_num, dtype=torch.float32)
            # q_weight: raw size is oc * ic, packed size depends on quant_bit
            # bits < 8 packing logic matches repack_low_bits
            q_weight_num = (oc * ic * quant_bit + 7) // 8
            q_weight = torch.zeros(q_weight_num, dtype=torch.uint8)
            return q_weight, alpha

        q_weight, alpha = torch_quant(weight, quant_bit, quant_block, symmetric, self.args.awq, self.args.hqq)
        return q_weight, alpha

    def write_weight(self, data):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if isinstance(data, list):
            data = np.array(data).astype(np.float32)
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
            if self.exporter.args.skip_weight:
                # Use a small dummy buffer and skip full weight loading/conversion
                weight_len = (ic * oc * 2)
                self.mnn_weight.seek(weight_len, 1)
            else:
                half_weight = linear.weight.data.flatten().half()
                weight_len = self.write_weight(half_weight)
            alpha_len, q_min, shape_int32, header_len = 0, 0, False, 0
        else:
            q_min = 1
            assert(quant_bit in (1, 2, 4, 8))
            q_weight, alpha = self.quant(linear.weight.data, quant_bit, quant_block, symmetric)
            header_len, shape_int32 = self.write_header(ic, oc, quant_bit)
            if self.exporter.args.skip_weight:
                weight_len = len(q_weight) + header_len
                self.mnn_weight.seek(len(q_weight), 1)
                alpha_len = len(alpha) * 4
                self.mnn_weight.seek(alpha_len, 1)
            else:
                weight_len = self.write_weight(q_weight) + header_len
                alpha_len = self.write_weight(alpha)

        if linear.bias is not None:
            bias_length = (oc * 4)
            if self.exporter.args.skip_weight:
                self.mnn_weight.seek(bias_length, 1)
            else:
                bias = linear.bias.data.flatten().float()
                bias_length = self.write_weight(bias)
        else:
            bias_length = 0

        external = [self.mnn_weight_offset, weight_len, alpha_len, bias_length, 0]
        self.mnn_weight_offset += (weight_len + alpha_len + bias_length)
        return external, q_min, shape_int32, header_len

    def build_tensor(self, graph, tensor_name):
        tensor_key = 'tensorName'
        if tensor_key not in graph and 'tensors' in graph:
            tensor_key = 'tensors'
        tensor_idx = [len(graph[tensor_key])]
        graph[tensor_key].append(tensor_name)
        return tensor_idx

    def rebuild_op(self, op, graph):
        if "type" in op['main']:
            op_type = op['main']['type']
        else:
            op_type = op['type']
        if op_type == 'FakeLinear':
            return self.rebuild_linear(op, graph)
        if op_type == 'FusedAttention':
            return self.rebuild_attnention(op, graph)
        if op_type == 'FusedLinearAttention':
            return self.rebuild_linear_attnention(op, graph)
        if op_type == "LayerNorm":
            return self.rebuild_layernorm(op, graph)
        if op_type == 'MoE':
            return self.rebuild_moe(op, graph)
        return None

    def rebuild_moe(self, op, graph):
        moe = copy.deepcopy(op)
        moe['main'] = { 'attr': moe['main']['attr'][:3] }
        moe['type'] = 'MoE'
        return [moe]

    def rebuild_layernorm(self, op, graph):
        if "gamma" not in op['main'] or "beta" not in op['main']:
            return [op]
        attr = op['main']
        gamma = attr['gamma']
        beta = attr['beta']
        gamma_len = self.write_weight(gamma)
        beta_len = self.write_weight(beta)
        del attr['gamma']
        del attr['beta']
        external = [self.mnn_weight_offset, gamma_len, beta_len]
        self.mnn_weight_offset += (gamma_len + beta_len)
        attr['external'] = external
        layernorm_op = {
            "name": op['name'],
            "inputIndexes": op['inputIndexes'],
            "outputIndexes": op['outputIndexes'],
            "type": "LayerNorm",
            "main_type": "LayerNorm",
            "main": attr,
            "defaultDimentionFormat": op['defaultDimentionFormat']
        }
        return [layernorm_op]

    def rebuild_attnention(self, op, graph):
        attrs = op['main']['attr']
        for attr in attrs:
            if attr['key'] == 'name':
                name = attr['s']
            elif attr['key'] == 'kv_cache':
                kv_cache = attr['i']
        origin_input = op['inputIndexes']
        origin_output = op['outputIndexes']
        fused_attention = {
            "inputIndexes": origin_input,
            "main_type": "AttentionParam",
            "main": { "kv_cache": bool(kv_cache) },
            "name": name,
            "outputIndexes": origin_output,
            "type": "Attention",
            "defaultDimentionFormat": "NHWC"
        }
        return [fused_attention]

    def rebuild_linear_attnention(self, op, graph):
        attrs = op['main']['attr']
        num_k_heads = 0
        num_v_heads = 0
        head_k_dim = 0
        head_v_dim = 0
        attn_type = "gated_delta_rule"
        use_qk_l2norm = False
        name = ""

        # Parse attributes from Custom Op
        for attr in attrs:
            if attr['key'] == 'name':
                name = attr['s']
            elif attr['key'] == 'num_k_heads':
                num_k_heads = attr['i']
            elif attr['key'] == 'num_v_heads':
                num_v_heads = attr['i']
            elif attr['key'] == 'head_k_dim':
                head_k_dim = attr['i']
            elif attr['key'] == 'head_v_dim':
                head_v_dim = attr['i']
            elif attr['key'] == 'attn_type':
                attn_type = attr['s']
            elif attr['key'] == 'use_qk_l2norm':
                use_qk_l2norm = bool(attr['i'])

        input_indexes = op['inputIndexes']
        output_indexes = op['outputIndexes']

        linear_attention_param = {
            "attn_type": attn_type,
            "num_k_heads": num_k_heads,
            "num_v_heads": num_v_heads,
            "head_k_dim": head_k_dim,
            "head_v_dim": head_v_dim,
            "use_qk_l2norm": use_qk_l2norm
        }

        fused_linear_attention = {
            "inputIndexes": input_indexes,
            "main_type": "LinearAttentionParam",
            "main": linear_attention_param,
            "name": name,
            "outputIndexes": output_indexes,
            "type": "LinearAttention",
            "defaultDimentionFormat": "NHWC"
        }
        return [fused_linear_attention]

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
        quant_bit = self.args.lm_quant_bit if is_lm else self.args.quant_bit
        quant_block = self.args.lm_quant_block if is_lm else self.args.quant_block
        quant_sym = self.args.sym

        if self.args.quant_config is not None:
            with open(self.args.quant_config, 'r') as f:
                quant_config = json.load(f)
            if name in quant_config:
                op_config = quant_config[name]
                quant_bit = op_config.get('bits', quant_bit)
                quant_block = op_config.get('block_size', quant_block)
                quant_sym = op_config.get('symmetric', quant_sym)

        block_size = ic if quant_block == 0 else quant_block
        if is_lm and self.lm_weight is not None:
            external, q_min, shape_int32, header_len = self.lm_weight
        else:
            external, q_min, shape_int32, header_len = self.build_weight(linear, quant_bit, quant_block, quant_sym)
        if is_lm and self.lm_weight is None:
            self.lm_weight = [external, q_min, shape_int32, header_len]
        if is_lm and self.args.tie_word_embeddings:
            weight_offset = external[0] + header_len
            alpha_offset = external[0] + external[1]
            alpha_size = external[2]
            self.tie_embeddings_info = [weight_offset, alpha_offset, alpha_size, quant_bit, quant_block]

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
            if self.args.sym:
                aMin = 0
                readType = 0
            else:
                aMin = q_min
                readType = oc * (ic // block_size)

            quanParameter = {
                "quantScale": 1.0, "scaleIn": 0.0, "scaleOut": 0.0,
                "useInt32": False, "has_scaleInt": False, "shapeInt32": shape_int32,
                "type": 1, "aMaxOrBits": quant_bit, "aMin": aMin, "readType": readType, "weightSize": 0
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
        if name.startswith('/expert/'):
            post_reshape['main']['dims'] = [-1, oc]
        return [pre_reshape, pre_convert, conv_op, post_convert, post_reshape]
