import glob
import json
import math
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

import safetensors
import torch
from torch.overrides import TorchFunctionMode

from utils.torch_utils import repack_low_bits


active_qat_loader = ContextVar('active_qat_loader', default=None)


def mark_qat_sources(model, args, source_names=None):
    """Record checkpoint keys before the common export wrappers remap modules."""
    if args is None or not getattr(args, 'qat', False):
        return
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param.ndim < 2:
                continue
            key = f'{name}.{param_name}' if name else param_name
            source_name = source_names.get(key, key) if source_names is not None else key
            param._mnn_qat_source_weight_name = source_name
            if param_name == 'weight':
                module._mnn_qat_source_weight_name = source_name


def filter_qat_unexpected_keys(unexpected, state_dict, args):
    if args is None or not getattr(args, 'qat', False):
        return unexpected
    return [name for name in unexpected if not (
        name.endswith(('.scale', '.zero_point')) and
        (name.rsplit('.', 1)[0] + '.weight' in state_dict or name.rsplit('.', 1)[0] in state_dict))]


def copy_qat_source(source, target, transform=None, preserve_groups=True):
    """Copy QAT metadata after weight assignment, without modifying tensor data."""
    name = getattr(source, '_mnn_qat_source_weight_name', None)
    if name is None:
        return
    target._mnn_qat_source_weight_name = name
    weight = getattr(source, 'weight', source)
    shape = weight.shape
    if isinstance(source, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        shape = (shape[0], math.prod(shape[1:]))
    target._mnn_source_weight_shape = getattr(source, '_mnn_source_weight_shape', shape)
    target._mnn_weight_transforms = getattr(source, '_mnn_weight_transforms', ())
    if transform is not None:
        target._mnn_weight_transforms += ((transform, tuple(target.weight.shape), preserve_groups),)


class SafetensorStore:
    def __init__(self, model_path):
        self.tensor_files = {}
        files = sorted(glob.glob(os.path.join(model_path, '*.safetensors')))
        if not files:
            raise ValueError(f'No safetensors files found in QAT checkpoint: {model_path}')
        for filename in files:
            with safetensors.safe_open(filename, framework='pt', device='cpu') as handle:
                for key in handle.keys():
                    if key in self.tensor_files:
                        raise ValueError(f'Duplicate tensor {key} in QAT checkpoint')
                    self.tensor_files[key] = filename

    def has_tensor(self, name):
        return name in self.tensor_files

    def get_tensor(self, name):
        filename = self.tensor_files.get(name)
        if filename is None:
            raise KeyError(f'Tensor not found in QAT checkpoint: {name}')
        with safetensors.safe_open(filename, framework='pt', device='cpu') as handle:
            return handle.get_tensor(name)


def pack_quantized_values(values, bits, block_size):
    values = values.to(torch.uint8).contiguous().flatten()
    if bits == 8:
        return values
    if bits in (2, 4):
        values_per_byte = 8 // bits
        if values.numel() % values_per_byte != 0:
            raise ValueError(f'Cannot pack {values.numel()} {bits}-bit values into complete bytes')
        grouped = values.reshape(-1, values_per_byte)
        packed = torch.zeros(grouped.shape[0], dtype=torch.uint8)
        for index in range(values_per_byte):
            packed |= grouped[:, index] << (bits * (values_per_byte - 1 - index))
        return packed
    if bits != 3:
        raise ValueError(f'Unsupported QAT quantization bits: {bits}')
    if block_size * bits % 8 != 0:
        raise ValueError(f'3-bit block size must produce whole bytes, got {block_size}')
    return repack_low_bits(values.reshape(-1, block_size), bits, block_size).flatten()


class QATLoader:
    def __init__(self, model_path, args):
        config_path = os.path.join(model_path, 'qat_config.json')
        if not os.path.isfile(config_path):
            raise ValueError(f'--qat requires qat_config.json under --path: {config_path}')
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = json.load(file)

        self.bits = int(self.config['bits'])
        self.group_size = int(self.config['group_size'])
        self.symmetric = bool(self.config.get('symmetric', False))
        self.quant_lm_head = bool(self.config.get('quant_lm_head', False))
        self.lm_head_bits = int(self.config.get('lm_head_bits') or self.bits)
        lm_head_group_size = self.config.get('lm_head_group_size')
        self.lm_head_group_size = int(
            lm_head_group_size if lm_head_group_size is not None else self.group_size)
        for bits in (self.bits, self.lm_head_bits):
            if bits not in (2, 3, 4, 8):
                raise ValueError(f'Unsupported QAT bits: {bits}')
        if self.group_size < 0 or self.lm_head_group_size < 0:
            raise ValueError('QAT group sizes must be non-negative')

        conflicting_options = []
        for name in ('awq', 'hqq', 'omni', 'smooth'):
            if getattr(args, name, False):
                conflicting_options.append(f'--{name}')
        for name in ('gptq_path', 'lora_path'):
            if getattr(args, name, None) is not None:
                conflicting_options.append(f'--{name}')
        if conflicting_options:
            raise ValueError(
                '--qat cannot be combined with ' + ', '.join(conflicting_options))
        if getattr(args, 'skip_weight', False):
            raise ValueError('--qat cannot be combined with --skip_weight')

        self.store = SafetensorStore(model_path)
        self.lm_head_weight = None
        self.used_qparams = set()
        self.exported_weights = {}

    @contextmanager
    def export_context(self):
        token = active_qat_loader.set(self)
        try:
            yield
            self.validate_export()
        finally:
            active_qat_loader.reset(token)

    @contextmanager
    def export_weights(self, model, onnx_path):
        import onnx
        from torch.onnx import symbolic_opset9

        path = os.path.abspath(onnx_path)
        self.exported_weights.pop(path, None)
        # Some export wrappers keep rebuilt blocks in ordinary lists/dicts.
        seen = set()
        def visit(value):
            if not isinstance(value, (torch.nn.Module, list, tuple, dict)) or id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, torch.nn.Module):
                children = list(value.children()) + [
                    child for key, child in vars(value).items()
                    if key not in ('_modules', '_parameters', '_buffers')]
                yield value
            else:
                children = value.values() if isinstance(value, dict) else value
            for child in children:
                yield from visit(child)

        linears = [module for module in visit(model) if isinstance(module, torch.nn.Linear)]
        weight_sources = {id(module.weight): module for module in linears}
        registered = {id(param) for param in model.parameters()}
        constants = {id(param): param.detach() for module in linears
                     for param in (module.weight, module.bias) if param is not None and id(param) not in registered}
        parameters = []
        # Shared floating weights can still have different per-module QAT parameters.
        active_linear = []
        loader = self

        class QATLinear(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, weight, bias, index):
                return torch.nn.functional.linear(input, weight, bias)

            @staticmethod
            def symbolic(g, input, weight, bias, index):
                if bias is None:
                    result = symbolic_opset9.matmul(g, input, symbolic_opset9.t(g, weight))
                else:
                    result = symbolic_opset9.linear(g, input, weight, bias)
                node = result.node()
                node.i_('mnn_qat', index)
                if node.kind() == 'onnx::Add':
                    node = next(value.node() for value in node.inputs()
                                if value.node().kind() == 'onnx::MatMul')
                    node.i_('mnn_qat', index)
                return result

        class LinearExportMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func is torch.nn.functional.linear:
                    input = args[0] if args else kwargs['input']
                    weight = args[1] if len(args) > 1 else kwargs['weight']
                    bias = args[2] if len(args) > 2 else kwargs.get('bias')
                    source = weight_sources.get(id(weight), weight)
                    if active_linear and active_linear[-1].weight is weight:
                        source = active_linear[-1]
                    if loader.has_qparams(getattr(source, '_mnn_qat_source_weight_name', None)):
                        linear = source
                        if not isinstance(source, torch.nn.Module):
                            linear = SimpleNamespace(weight=weight)
                            copy_qat_source(source, linear)
                        if loader.expected_config(linear._mnn_qat_source_weight_name)[0] == 16:
                            return func(*args, **kwargs)
                        parameters.append(linear)
                        return QATLinear.apply(input, constants.get(id(weight), weight),
                                               constants.get(id(bias), bias), len(parameters) - 1)
                return func(*args, **kwargs)

        hooks = []
        def leave_linear(module, inputs, output):
            active_linear.pop()
        try:
            for module in linears:
                hooks.append(module.register_forward_pre_hook(lambda module, inputs: active_linear.append(module)))
                hooks.append(module.register_forward_hook(leave_linear, always_call=True))
            with LinearExportMode():
                yield
        finally:
            for hook in hooks:
                hook.remove()
        if not parameters:
            return
        # Keep ONNX unchanged apart from removing temporary source annotations.
        graph = onnx.load(onnx_path, load_external_data=False)
        names = [[] for _ in parameters]
        def annotate(graph, constants=()):
            constants = set(constants) | {tensor.name for tensor in graph.initializer}
            owners = {}
            for node in graph.node:
                for attr in list(node.attribute):
                    if attr.name == 'mnn_qat':
                        owners[node.output[0]] = attr.i
                        names[attr.i].append(node.output[0] + '__matmul_converted')
                        node.attribute.remove(attr)
                    elif attr.type == onnx.AttributeProto.GRAPH:
                        annotate(attr.g, constants)
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        for subgraph in attr.graphs:
                            annotate(subgraph, constants)
                if node.op_type == 'Constant' or (node.input and all(name in constants for name in node.input)):
                    constants.update(node.output)
                # MNN may use an added constant bias's output name for the converted Linear.
                if node.op_type == 'Add' and node.output[0] not in owners:
                    for source, bias in (node.input, reversed(node.input)):
                        if source in owners and bias in constants:
                            index = owners[source]
                            owners[node.output[0]] = index
                            names[index].append(node.output[0] + '__matmul_converted')
                            break
        annotate(graph.graph)
        if not all(names):
            raise ValueError('QAT Linear annotations were lost during ONNX export')
        onnx.save(graph, onnx_path)
        self.exported_weights[path] = list(zip(parameters, names))

    def apply(self, graph_path, converter):
        path = os.path.abspath(converter.onnx_model_path)
        records = [((linear,), names) for linear, names in self.exported_weights.get(path, [])] + [
            (linears, [name]) for name, linears in converter.qat_weights.items()]
        aliases = {}
        for index, (_, names) in enumerate(records):
            for name in names:
                if name in aliases:
                    raise ValueError(f'Ambiguous QAT weight mapping: {name}')
                aliases[name] = index
        with open(graph_path, encoding='utf-8') as file:
            graph = json.load(file)
        targets, consumed = {}, set()
        op_lists = [graph['oplists']] + [subgraph['nodes'] for subgraph in graph.get('subgraphs', [])]
        for ops in op_lists:
            for op in ops:
                if op['type'] == 'Convolution':
                    members = [(op['name'], op['main'])]
                elif op['type'] == 'FusedLinear':
                    names, convs = op['name'].split('+'), op['main']['convs']
                    if len(names) != len(convs):
                        raise ValueError(f'Missing FusedLinear member names: {op["name"]}')
                    members = zip(names, convs)
                elif op['type'] == 'IndexedLinear':
                    members = [(f'{op["name"]}/linear.{index}', main)
                               for index, main in enumerate(op['main']['linears'])]
                else:
                    continue
                for name, main in members:
                    if name not in aliases:
                        continue
                    index = aliases[name]
                    linears = records[index][0]
                    ic = linears[0].weight.shape[1]
                    oc = sum(linear.weight.shape[0] for linear in linears)
                    common = main['common']
                    if (common['inputCount'], common['outputCount']) != (ic, oc) or any(
                            common.get(key, 1) != 1 for key in ('kernelX', 'kernelY', 'group')):
                        raise ValueError(f'QAT Linear shape mismatch: {name}')
                    targets[id(main)] = linears
                    consumed.add(index)
        missing = set(range(len(records))) - consumed
        if missing:
            raise ValueError('QAT Linear weights were not found after conversion: ' + ', '.join(
                records[index][1][0] for index in sorted(missing)))
        if not records:
            self.exported_weights.pop(path, None)
            return

        # Repack referenced ranges so replacing float/native weights leaves no unused payload.
        weight_path = converter.mnn_weight_path
        file_size = os.path.getsize(weight_path)
        copied = {}
        tied = converter.tie_embeddings_info
        with tempfile.TemporaryDirectory(dir=os.path.dirname(weight_path)) as temp:
            new_path = os.path.join(temp, 'weights')
            with open(weight_path, 'rb') as source, open(new_path, 'wb') as output:
                converter.mnn_weight = output
                converter.mnn_weight_offset = 0

                def read(offset, size):
                    if offset < 0 or size < 0 or offset + size > file_size:
                        raise ValueError(f'Invalid external weight range: {offset}, {size}')
                    source.seek(offset)
                    data = source.read(size)
                    if len(data) != size:
                        raise ValueError('Truncated external weights')
                    return data

                def rewrite(value):
                    if isinstance(value, list):
                        for item in value:
                            rewrite(item)
                        return
                    if not isinstance(value, dict):
                        return
                    linears = targets.get(id(value))
                    if linears is not None:
                        ic = linears[0].weight.shape[1]
                        oc = sum(linear.weight.shape[0] for linear in linears)
                        external = value.get('external', [])
                        bias = value.get('bias', [])
                        if external:
                            bias_index = 3 if value.get('quanParameter') else 2
                            bias = torch.frombuffer(bytearray(read(
                                external[0] + sum(external[1:bias_index]), external[bias_index])),
                                dtype=torch.float32) if external[bias_index] else None
                        else:
                            bias = torch.tensor(bias, dtype=torch.float32) if bias else None
                        if bias is not None and bias.numel() != oc:
                            raise ValueError(f'QAT Linear bias size mismatch: {linears[0]._mnn_qat_source_weight_name}')
                        weights, row = [], 0
                        for linear in linears:
                            rows = linear.weight.shape[0]
                            end = row + rows
                            weight = SimpleNamespace(weight=linear.weight, bias=bias[row:end] if bias is not None else None,
                                                     in_features=ic, out_features=rows)
                            copy_qat_source(linear, weight)
                            weights.append(weight)
                            row = end
                        source_name = weights[0]._mnn_qat_source_weight_name
                        bits, block, symmetric = self.expected_config(source_name)
                        converter.mnn_weight_offset = output.tell()
                        external, q_min, shape_int32, header = converter.build_weight(
                            weights[0], bits, block, symmetric, weights[1] if len(weights) > 1 else None)
                        value['external'] = external
                        value['quanParameter'] = {
                            'type': 1, 'aMaxOrBits': bits, 'aMin': 0 if symmetric else q_min,
                            'readType': 0 if symmetric else oc * (ic // (block or ic)),
                            'shapeInt32': shape_int32, 'quantScale': 1.0,
                            'scaleIn': 0.0, 'scaleOut': 0.0, 'useInt32': False,
                            'has_scaleInt': False, 'weightSize': 0,
                            'scaleStorage': 'FP16' if converter.args.scale_bit == 16 else 'FP32',
                        }
                        value.pop('weight', None)
                        value.pop('bias', None)
                        if converter.args.tie_word_embeddings and source_name == self.lm_head_weight:
                            converter.tie_embeddings_info = {
                                'weight_offset': external[0] + header,
                                'alpha_offset': external[0] + external[1], 'alpha_size': external[2],
                                'quant_bit': bits, 'quant_block': block,
                                'alpha_dtype': 'fp16' if converter.args.scale_bit == 16 else 'fp32',
                            }
                        return
                    external = value.get('external')
                    if external:
                        key = tuple(external)
                        if key not in copied:
                            copied[key] = output.tell()
                            offset, size = external[0], sum(external[1:])
                            while size:
                                chunk = min(size, 8 * 1024 * 1024)
                                output.write(read(offset, chunk))
                                offset, size = offset + chunk, size - chunk
                        value['external'] = [copied[key]] + external[1:]
                        if tied is not None and converter.tie_embeddings_info is tied and external[0] == tied['weight_offset']:
                            converter.tie_embeddings_info = dict(tied, weight_offset=copied[key],
                                alpha_offset=tied['alpha_offset'] + copied[key] - external[0])
                    for key, item in value.items():
                        if key != 'external':
                            rewrite(item)

                rewrite(graph)
                converter.mnn_weight_offset = output.tell()
            os.replace(new_path, weight_path)
        with open(graph_path, 'w', encoding='utf-8') as file:
            json.dump(graph, file, ensure_ascii=False, indent=4)
        self.exported_weights.pop(path, None)

    def bind_tied_embeddings(self, linear, embedding, tied):
        self.lm_head_weight = getattr(linear, '_mnn_qat_source_weight_name', None)
        head_quantized = self.has_qparams(self.lm_head_weight) and self.quant_lm_head
        source_name = getattr(embedding, '_mnn_qat_source_weight_name', None)
        if not self.has_qparams(source_name):
            # Equal floating weights do not make an unquantized embedding
            # interchangeable with a fake-quantized output head.
            return tied and not head_quantized
        if not self.config.get('tie_word_embeddings', False) or not tied or not self.quant_lm_head:
            raise ValueError('Embedding QAT parameters require a tied, quantized lm_head')
        if head_quantized and self.lm_head_weight != source_name:
            for suffix in ('scale',) if self.symmetric else ('scale', 'zero_point'):
                head_param = self.store.get_tensor(f'{self._tensor_prefix(self.lm_head_weight)}.{suffix}')
                embed_param = self.store.get_tensor(f'{self._tensor_prefix(source_name)}.{suffix}')
                if not torch.equal(head_param.float().flatten(), embed_param.float().flatten()):
                    raise ValueError(f'Tied embedding and lm_head have different QAT {suffix}')
            self.used_qparams.add(self._tensor_prefix(self.lm_head_weight))
        copy_qat_source(embedding, linear)
        self.lm_head_weight = source_name
        return True

    @staticmethod
    def _tensor_prefix(source_weight_name):
        if not source_weight_name:
            return None
        return source_weight_name[:-len('.weight')] if source_weight_name.endswith('.weight') else source_weight_name

    def has_qparams(self, source_weight_name):
        prefix = self._tensor_prefix(source_weight_name)
        if prefix is None:
            return False
        has_scale = self.store.has_tensor(f'{prefix}.scale')
        has_zero = self.store.has_tensor(f'{prefix}.zero_point')
        if not has_scale and not has_zero:
            return False
        if not has_scale or (not self.symmetric and not has_zero):
            missing = 'scale' if not has_scale else 'zero_point'
            raise ValueError(f'Missing QAT tensor: {prefix}.{missing}')
        return True

    def expected_config(self, source_weight_name):
        if source_weight_name is not None and source_weight_name == self.lm_head_weight:
            if not self.quant_lm_head:
                self.used_qparams.add(self._tensor_prefix(source_weight_name))
                return 16, self.lm_head_group_size, self.symmetric
            return self.lm_head_bits, self.lm_head_group_size, self.symmetric
        return self.bits, self.group_size, self.symmetric

    def validate_export(self):
        sources = set()
        for name in self.store.tensor_files:
            if name.endswith('.zero_point'):
                sources.add(name.rsplit('.', 1)[0])
            elif name.endswith('.scale'):
                prefix = name.rsplit('.', 1)[0]
                # Modules may also have ordinary scale parameters, e.g. routers.
                if self.store.has_tensor(prefix + '.weight') or self.store.has_tensor(prefix):
                    sources.add(prefix)
        for source in sorted(sources):
            self.has_qparams(source + '.weight')
        unused = sources - self.used_qparams
        if unused:
            raise ValueError('QAT parameters were not exported: ' + ', '.join(sorted(unused)))
        if not sources:
            raise ValueError('No QAT scale/zero_point tensors found in checkpoint')

    def get_qparams(self, linear, quant_bit, quant_block, symmetric):
        source_weight_name = getattr(linear, '_mnn_qat_source_weight_name', None)
        if not self.has_qparams(source_weight_name):
            raise ValueError(f'quantize() called without QAT params for {source_weight_name}')

        expected = self.expected_config(source_weight_name)
        actual = (quant_bit, quant_block, symmetric)
        if actual != expected:
            raise ValueError(
                f'QAT export config mismatch for {source_weight_name}: '
                f'checkpoint={expected}, exporter={actual}')

        prefix = self._tensor_prefix(source_weight_name)
        scale_name = f'{prefix}.scale'
        zero_name = f'{prefix}.zero_point'
        oc, ic = linear.weight.shape
        block_size = ic if quant_block == 0 else quant_block
        if block_size <= 0 or ic % block_size != 0:
            raise ValueError(
                f'Invalid QAT block size {block_size} for {source_weight_name} '
                f'with input size {ic}')
        block_num = ic // block_size
        source_shape = tuple(getattr(linear, '_mnn_source_weight_shape', linear.weight.shape))
        source_block = source_shape[-1] if quant_block == 0 else quant_block
        if source_shape[-1] % source_block:
            raise ValueError(f'Invalid QAT block size {source_block} for source {source_weight_name}')
        param_shape = source_shape[:-1] + (source_shape[-1] // source_block,)

        def load_param(name):
            param = self.store.get_tensor(name).float().cpu()
            if param.numel() != math.prod(param_shape):
                raise ValueError(f'QAT shape mismatch for {name}: expected {param_shape}, got {tuple(param.shape)}')
            param = param.reshape(param_shape)
            shape = source_shape
            for transform, output_shape, preserve_groups in getattr(linear, '_mnn_weight_transforms', ()):
                block = shape[-1] if quant_block == 0 else quant_block
                output_block = output_shape[-1] if quant_block == 0 else quant_block
                if output_shape[-1] % output_block:
                    raise ValueError(f'QAT transform for {name} splits a quantization block')
                output_param_shape = output_shape[:-1] + (output_shape[-1] // output_block,)
                if preserve_groups:
                    param = transform(param)
                else:
                    expanded = transform(param.repeat_interleave(block, dim=-1))
                    groups = expanded.reshape(*output_param_shape, output_block)
                    param = groups[..., 0]
                    if not torch.equal(groups, param.unsqueeze(-1).expand_as(groups)):
                        raise ValueError(f'QAT transform for {name} mixes input-channel quantization groups')
                if tuple(param.shape) != output_param_shape:
                    raise ValueError(
                        f'QAT transform shape mismatch for {name}: '
                        f'expected {output_param_shape}, got {tuple(param.shape)}')
                shape = output_shape
            if tuple(param.shape) != (oc, block_num):
                raise ValueError(
                    f'QAT transform for {name} does not preserve input-channel groups: '
                    f'expected {(oc, block_num)}, got {tuple(param.shape)}')
            return param

        scale = load_param(scale_name)
        if not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError(f'QAT scale must be finite and positive: {scale_name}')
        zero_point = None if symmetric else load_param(zero_name)
        if zero_point is not None and not torch.isfinite(zero_point).all():
            raise ValueError(f'QAT zero_point must be finite: {zero_name}')
        self.used_qparams.add(prefix)
        return scale, zero_point

    def quantize(self, linear, quant_bit, quant_block, symmetric):
        scale, zero_point = self.get_qparams(linear, quant_bit, quant_block, symmetric)
        weight = linear.weight.detach().float().cpu()
        if not torch.isfinite(weight).all():
            raise ValueError(f'QAT weight must be finite: {linear._mnn_qat_source_weight_name}')
        oc, ic = weight.shape
        block_size = ic if quant_block == 0 else quant_block
        block_num = ic // block_size
        offset = 1 << (quant_bit - 1)
        max_code = (1 << quant_bit) - 1
        weight_blocks = weight.reshape(oc, block_num, block_size)
        if symmetric:
            zero_point = None
            signed = torch.round(weight_blocks / scale.unsqueeze(-1))
            signed = torch.clamp(signed, -offset, offset - 1)
            codes = (signed + offset).to(torch.uint8)
            alpha = scale.flatten()
        else:
            codes = torch.round(
                weight_blocks / scale.unsqueeze(-1) + zero_point.unsqueeze(-1))
            codes = torch.clamp(codes, 0, max_code).to(torch.uint8)
            bias = (offset - zero_point) * scale
            alpha = torch.stack([bias, scale], dim=-1).flatten()

        packed = pack_quantized_values(codes, quant_bit, block_size)
        return packed, alpha
