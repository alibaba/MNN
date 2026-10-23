import os
import sys
import copy
import json
import torch
import numpy as np

from .torch_utils import quant as torch_quant
from tqdm import tqdm
from .spinner import spinner_run
from .gptq import GPTQ
from .lora import LoRA
from .custom_op import IndexedLinear

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
        if getattr(self.exporter, 'qat_loader', None) is not None:
            self.qat_weights = {}

    def transformer_c4_args(self):
        enabled = getattr(self.args, 'transformer_c4', True)
        # FusedLinear groups are built by MNNConvert inside FuseTransformerC4.
        # The op name records the member projection names in convs order, so
        # lora.py can patch LoRA onto a FusedLinear without unfusing it.
        qkv = enabled and getattr(self.args, 'fuse_qkv_proj', True)
        gate_up = enabled and getattr(self.args, 'fuse_gate_up_proj', True)
        ln = enabled and getattr(self.args, 'fuse_ln_proj', True)
        return [
            '--transformerFuseC4={}'.format(1 if enabled else 0),
            '--transformerFuseQkvProj={}'.format(1 if qkv else 0),
            '--transformerFuseGateUpProj={}'.format(1 if gate_up else 0),
            '--transformerFuseLnProj={}'.format(1 if ln else 0),
        ]

    def convert(self, convert_args):
        import contextlib
        log_fp = open(EXPORT_LOG, "a")
        sfd = None
        try:
            sfd = os.dup(1)
            log_fd = log_fp.fileno()
            # mnnconvert ... > .export.log
            os.dup2(log_fd, 1)
        except Exception:
            if sfd is not None:
                os.close(sfd)
                sfd = None

        try:
            with contextlib.redirect_stdout(log_fp):
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
            if sfd is not None:
                os.dup2(sfd, 1)
                os.close(sfd)
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
        convert_args += self.transformer_c4_args()
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
        convert_args += self.transformer_c4_args()
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
        convert_args += self.transformer_c4_args()
        self.convert(convert_args)
        return mnn_path

    def export(self, onnx_path, quant_bit = None, quant_block = None, transformer_fuse = True, group_conv_native = False, weight_sym = None):
        if getattr(self.exporter, 'qat_loader', None) is not None:
            self.qat_weights = {}
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
            # HunyuanVL needs the explicit q/k/v reshape boundary before Attention.
            # The extra optimize pass can bypass v_proj's post_reshape into Attention.
            if self.exporter.model_type != 'hunyuan_vl':
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
        if getattr(self.exporter, 'qat_loader', None) is not None:
            mnn_json = f'{self.mnn_model_path}.json'
            if self.weight_ops is None:
                self.mnn2json(self.mnn_model_path, mnn_json)
            self.apply_qat(mnn_json)
            self.json2mnn(mnn_json, self.mnn_model_path)
        return self.tie_embeddings_info

    @spinner_run(f'apply gptq to ')
    def apply_gptq(self, mnn_json):
        GPTQ(self.args.gptq_path).apply(mnn_json, self.mnn_weight_path)
        return self.mnn_weight_path

    @spinner_run(f'apply qat to ')
    def apply_qat(self, mnn_json):
        self.exporter.qat_loader.apply(mnn_json, self)
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

        q_weight, alpha = torch_quant(weight.cpu(), quant_bit, quant_block, symmetric, self.args.awq, self.args.hqq)
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

    def build_weight(self, linear, quant_bit, quant_block, symmetric, up_linear=None):
        projections = (linear,) if up_linear is None else (linear, up_linear)
        ic, oc = linear.in_features, sum(item.out_features for item in projections)
        if quant_bit == 16:
            if self.exporter.args.skip_weight:
                # Use a small dummy buffer and skip full weight loading/conversion
                weight_len = (ic * oc * 2)
                self.mnn_weight.seek(weight_len, 1)
            else:
                weight_len = sum(self.write_weight(item.weight.data.flatten().half()) for item in projections)
            alpha_len, q_min, shape_int32, header_len = 0, 0, False, 0
        else:
            q_min = 1
            assert(quant_bit in (1, 2, 3, 4, 8))
            header_len, shape_int32 = self.write_header(ic, oc, quant_bit)
            scale_fp16 = (self.args.scale_bit == 16)
            alpha_dtype_size = 2 if scale_fp16 else 4
            weight_len, alpha_len = header_len, 0
            alphas = []
            for item in projections:
                # HQQ's stopping criterion depends on the whole source tensor.
                # Quantize each original projection before joining its encoded rows.
                qat_loader = getattr(self.exporter, 'qat_loader', None)
                if qat_loader is not None and qat_loader.has_qparams(getattr(item, '_mnn_qat_source_weight_name', None)):
                    q_weight, alpha = qat_loader.quantize(item, quant_bit, quant_block, symmetric)
                else:
                    q_weight, alpha = self.quant(item.weight.data, quant_bit, quant_block, symmetric)
                if self.exporter.args.skip_weight:
                    weight_len += q_weight.numel()
                    self.mnn_weight.seek(q_weight.numel(), 1)
                else:
                    weight_len += self.write_weight(q_weight)
                alphas.append(alpha)
            for alpha in alphas:
                if self.exporter.args.skip_weight:
                    alpha_len += alpha.numel() * alpha_dtype_size
                    self.mnn_weight.seek(alpha.numel() * alpha_dtype_size, 1)
                    continue
                if scale_fp16:
                    alpha_np = alpha.numpy() if hasattr(alpha, 'numpy') else np.asarray(alpha)
                    alpha_fp16 = alpha_np.astype(np.float16)
                    alpha_len += self.write_weight(alpha_fp16)
                else:
                    alpha_len += self.write_weight(alpha)

        if any(item.bias is not None for item in projections):
            bias_length = (oc * 4)
            if self.exporter.args.skip_weight:
                self.mnn_weight.seek(bias_length, 1)
            else:
                bias_length = sum(self.write_weight(item.bias.data.flatten().float() if item.bias is not None else
                                                   np.zeros(item.out_features, dtype=np.float32))
                                  for item in projections)
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
        if op_type == 'FusedRoPE':
            return self.rebuild_rope(op, graph)
        if op_type == 'FusedLinearAttention':
            return self.rebuild_linear_attnention(op, graph)
        if op_type == "LayerNorm":
            return self.rebuild_layernorm(op, graph)
        if op_type == 'IndexedLinear':
            return self.rebuild_indexed_linear(op, graph)
        return None

    def const_float_data(self, graph, tensor_index):
        op_key = 'oplists' if 'oplists' in graph else 'nodes'
        for op in graph[op_key]:
            if tensor_index not in op.get('outputIndexes', []):
                continue
            if op.get('type') != 'Const':
                break
            main = op.get('main', {})
            if 'float32s' in main:
                return main['float32s']
            break
        return None

    def rebuild_rope(self, op, graph):
        attrs = op['main']['attr']
        name = op['name']
        rope_cut_head_dim = 0
        num_head = 0
        kv_num_head = 0
        head_dim = 0
        q_norm = False
        k_norm = False
        q_norm_eps = 0.0
        k_norm_eps = 0.0
        for attr in attrs:
            if attr['key'] == 'name':
                name = attr['s']
            elif attr['key'] == 'rope_cut_head_dim':
                rope_cut_head_dim = attr['i']
            elif attr['key'] == 'num_head':
                num_head = attr['i']
            elif attr['key'] == 'kv_num_head':
                kv_num_head = attr['i']
            elif attr['key'] == 'head_dim':
                head_dim = attr['i']
            elif attr['key'] == 'q_norm':
                q_norm = bool(attr['i'])
            elif attr['key'] == 'k_norm':
                k_norm = bool(attr['i'])
            elif attr['key'] == 'q_norm_eps':
                q_norm_eps = attr['f']
            elif attr['key'] == 'k_norm_eps':
                k_norm_eps = attr['f']

        rope_param = {
            "rope_cut_head_dim": rope_cut_head_dim,
            "num_head": num_head,
            "kv_num_head": kv_num_head,
            "head_dim": head_dim,
        }
        input_indexes = op['inputIndexes']
        if q_norm or k_norm:
            if len(input_indexes) < 6:
                raise RuntimeError(f'FusedRoPE {name} misses q/k norm inputs')
            if q_norm:
                q_gamma = self.const_float_data(graph, input_indexes[4])
                if q_gamma is None:
                    raise RuntimeError(f'FusedRoPE {name} misses q_norm gamma const')
                rope_param["q_norm"] = {
                    "axis": [-1],
                    "epsilon": q_norm_eps,
                    "gamma": q_gamma,
                    "useRMSNorm": True
                }
            if k_norm:
                k_gamma = self.const_float_data(graph, input_indexes[5])
                if k_gamma is None:
                    raise RuntimeError(f'FusedRoPE {name} misses k_norm gamma const')
                rope_param["k_norm"] = {
                    "axis": [-1],
                    "epsilon": k_norm_eps,
                    "gamma": k_gamma,
                    "useRMSNorm": True
                }
        rope_op = {
            "inputIndexes": input_indexes[:4],
            "main_type": "RoPEParam",
            "main": rope_param,
            "name": name,
            "outputIndexes": op['outputIndexes'],
            "type": "RoPE",
            "defaultDimentionFormat": op['defaultDimentionFormat']
        }
        return [rope_op]

    def rebuild_indexed_linear(self, op, graph):
        name = self.get_extra_attr(op, 'name')
        input_mode = self.get_extra_attr(op, 'input_mode', 0)
        out_features = self.get_extra_attr(op, 'out_features')
        fusedActivation = self.get_extra_attr(op, 'fusedActivation', -1)
        if fusedActivation not in (-1, IndexedLinear.SILU):
            raise ValueError(f'Unsupported IndexedLinear UnaryOpOperation: {fusedActivation}')
        fused = fusedActivation != -1
        up_name = self.get_extra_attr(op, 'up_name', '')
        bank = self.weight_ops[name]
        if not bank or input_mode not in (0, 1):
            raise ValueError(f'Invalid IndexedLinear bank {name}')
        ic, oc = bank[0][1].in_features, bank[0][1].out_features
        up_bank = self.weight_ops[up_name] if up_name else []
        if up_bank and (not fused or len(up_bank) != len(bank)):
            raise ValueError(f'Invalid IndexedLinear up bank {name}')
        expected_output = oc if up_bank or not fused else oc // 2
        if (fused and not up_bank and oc % 2) or out_features != expected_output:
            raise ValueError(f'IndexedLinear output size mismatch: {name}')
        quantization = self.linear_quantization(bank[0][0], bank[0][1])
        if quantization[0] not in (4, 8, 16):
            raise ValueError(f'IndexedLinear export supports only 4/8-bit or FP16 weights: {name}')
        if up_bank and self.linear_quantization(up_bank[0][0], up_bank[0][1]) != quantization:
            # The resident paired kernel shares its bit/group layout. Keep the
            # original graph when a quant_config gives gate/up different layouts.
            import copy
            tensors = graph['tensorName']
            intermediates = list(range(len(tensors), len(tensors) + 3))
            tensors.extend([name + suffix for suffix in ('/gate', '/up', '/silu')])
            graph['tensorNumber'] = len(tensors)
            result = []
            for bank_name, output in zip((name, up_name), intermediates):
                projection = copy.deepcopy(op)
                projection['main']['attr'] = [a for a in projection['main']['attr']
                                              if a['key'] not in ('name', 'fusedActivation', 'up_name')]
                projection['main']['attr'].append({'key': 'name', 's': bank_name})
                projection['outputIndexes'] = [output]
                result.extend(self.rebuild_indexed_linear(projection, graph))
            result.extend([
                {'type': 'UnaryOp', 'main_type': 'UnaryOp', 'main': {'opType': 'SILU'},
                 'name': name + '/silu', 'inputIndexes': [intermediates[0]], 'outputIndexes': [intermediates[2]]},
                {'type': 'BinaryOp', 'main_type': 'BinaryOp', 'main': {'opType': 'MUL'},
                 'name': name + '/swiglu', 'inputIndexes': [intermediates[2], intermediates[1]],
                 'outputIndexes': op['outputIndexes']},
            ])
            return result
        for original_name, linear in bank + up_bank:
            if self.linear_quantization(original_name, linear) != quantization:
                raise ValueError(f'IndexedLinear experts must use the same quantization: {original_name}')
        for original_name, linear in bank + up_bank:
            if linear.in_features != ic or linear.out_features != oc:
                raise ValueError(f'IndexedLinear expert shape mismatch: {original_name}')
        # IndexedLinear stores expert parameters in bank order, without per-expert op names.
        qat = getattr(self.exporter, 'qat_loader', None) is not None
        linears = [self.build_linear_parameter(f'{name}/linear.{index}' if qat else original_name,
                                              linear, up_bank[index][1] if up_bank else None)
                   for index, (original_name, linear) in enumerate(bank)]
        parameter = {'linears': linears, 'inputMode': input_mode}
        if fused:
            parameter['fusedActivation'] = fusedActivation
        return [{
            'type': 'IndexedLinear',
            'main_type': 'IndexedLinearParam',
            'main': parameter,
            'name': name,
            'inputIndexes': op['inputIndexes'],
            'outputIndexes': op['outputIndexes'],
            'defaultDimentionFormat': 'NCHW',
        }]

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
        layer_index = -1
        kv_shared_layer_index = -1
        for attr in attrs:
            if attr['key'] == 'name':
                name = attr['s']
            elif attr['key'] == 'kv_cache':
                kv_cache = attr['i']
            elif attr['key'] == 'layer_index':
                layer_index = attr.get('i', -1)
            elif attr['key'] == 'kv_shared_layer_index':
                kv_shared_layer_index = attr.get('i', -1)
        origin_input = op['inputIndexes']
        origin_output = op['outputIndexes']
        main_dict = {
            "kv_cache": bool(kv_cache),
            "layer_index": layer_index,
            "kv_shared_layer_index": kv_shared_layer_index,
        }
        fused_attention = {
            "inputIndexes": origin_input,
            "main_type": "AttentionParam",
            "main": main_dict,
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
        gate_fold = False
        gate_coef = None
        gate_bias = None
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
            elif attr['key'] == 'gate_fold':
                gate_fold = bool(attr['i'])
            elif attr['key'] == 'gate_coef':
                # ONNX FLOATS attrs serialize as Attribute.list.f in MNN JSON
                gate_coef = attr.get('list', {}).get('f', [])
            elif attr['key'] == 'gate_bias':
                gate_bias = attr.get('list', {}).get('f', [])

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
        if gate_fold:
            # The folded graph has no softplus/exp chain left, so a runtime that
            # finds these missing cannot fall back — it would read the raw `a`
            # projection as the decay gate. Refuse to emit such a model.
            if not gate_coef or not gate_bias or \
               len(gate_coef) != num_v_heads or len(gate_bias) != num_v_heads:
                raise RuntimeError(
                    f'LinearAttention "{name}": gate_fold needs {num_v_heads} gate_coef and gate_bias '
                    f'values, got {len(gate_coef or [])} and {len(gate_bias or [])}')
            linear_attention_param["gate_fold"] = True
            linear_attention_param["gate_coef"] = gate_coef
            linear_attention_param["gate_bias"] = gate_bias

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

    def get_extra_attr(self, op, key, default=None):
        for attr in op.get('main', {}).get('attr', []):
            if attr.get('key') != key:
                continue
            if 's' in attr:
                return attr['s']
            if 'i' in attr:
                return attr['i']
            if 'f' in attr:
                return attr['f']
        return default

    def linear_quantization(self, name, linear):
        qat_loader = getattr(self.exporter, 'qat_loader', None)
        if qat_loader is not None:
            source_name = getattr(linear, '_mnn_qat_source_weight_name', None)
            if qat_loader.has_qparams(source_name):
                bits, block, symmetric = qat_loader.expected_config(source_name)
                if bits != 16:
                    block = block or linear.in_features
                    if block <= 0 or linear.in_features % block:
                        raise ValueError(f'Invalid QAT block size {block} for {source_name}')
                    return bits, block, symmetric
                return 16, linear.in_features, False

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

        block_size = linear.in_features if quant_block == 0 else quant_block
        if block_size <= 0:
            raise ValueError(f'Invalid quantization block size for {name}: {quant_block}')
        while linear.in_features % block_size != 0:
            block_size //= 2
        return quant_bit, block_size, quant_sym

    def build_linear_parameter(self, name, linear, up_linear=None):
        ic, oc = linear.in_features, linear.out_features + (up_linear.out_features if up_linear is not None else 0)
        quant_bit, block_size, quant_sym = self.linear_quantization(name, linear)
        is_lm = 'lm_head' in name
        qat_loader = getattr(self.exporter, 'qat_loader', None)
        if qat_loader is not None:
            source_name = getattr(linear, '_mnn_qat_source_weight_name', None)
            is_lm = source_name is not None and source_name == qat_loader.lm_head_weight
            if quant_bit != 16 and any(
                    qat_loader.has_qparams(getattr(item, '_mnn_qat_source_weight_name', None))
                    for item in (linear, up_linear) if item is not None):
                self.qat_weights[name] = (linear,) if up_linear is None else (linear, up_linear)
                # Validate the final layout above, but encode QAT weights only after fusion.
                quant_bit = 16
        if is_lm and self.lm_weight is not None:
            external, q_min, shape_int32, header_len = self.lm_weight
        else:
            external, q_min, shape_int32, header_len = self.build_weight(linear, quant_bit, block_size, quant_sym,
                                                                      up_linear)
        if is_lm and self.lm_weight is None:
            self.lm_weight = [external, q_min, shape_int32, header_len]
        if is_lm and self.args.tie_word_embeddings:
            weight_offset = external[0] + header_len
            alpha_offset = external[0] + external[1]
            alpha_size = external[2]
            self.tie_embeddings_info = {
                "weight_offset": weight_offset,
                "alpha_offset": alpha_offset,
                "alpha_size": alpha_size,
                "quant_bit": quant_bit,
                "quant_block": block_size,
                "alpha_dtype": "fp16" if self.args.scale_bit == 16 else "fp32",
            }

        if quant_bit == 16:
            quanParameter = { "type": 3 }
        else:
            if quant_sym:
                aMin = 0
                readType = 0
            else:
                aMin = q_min
                readType = oc * (ic // block_size)

            quanParameter = {
                "quantScale": 1.0, "scaleIn": 0.0, "scaleOut": 0.0,
                "useInt32": False, "has_scaleInt": False, "shapeInt32": shape_int32,
                "type": 1, "aMaxOrBits": quant_bit, "aMin": aMin, "readType": readType, "weightSize": 0,
                "scaleStorage": "FP16" if self.args.scale_bit == 16 else "FP32",
            }
        return {
            'common': {
                'dilateX': 1, 'dilateY': 1, 'strideX': 1, 'strideY': 1,
                'kernelX': 1, 'kernelY': 1, 'padX': 0, 'padY': 0, 'group': 1,
                'outputCount': oc, 'relu': False, 'padMode': 'CAFFE',
                'relu6': False, 'inputCount': ic, 'hasOutputShape': False
            },
            "quanParameter": quanParameter,
            "external": external
        }

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

        parameter = self.build_linear_parameter(name, linear)

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

        conv_op = {
            "name": conv_name,
            "inputIndexes": pre_convert_output,
            "outputIndexes": conv_output,
            "type": "Convolution",
            "main_type": "Convolution2D",
            "main": parameter,
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
