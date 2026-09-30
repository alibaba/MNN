import os
import onnx
from .custom_op import IndexedLinear

class OnnxRebuilder:
    def __init__(self, onnx_path, weight_ops):
        self.weight_ops = weight_ops
        self.onnx_model = onnx.load(onnx_path)
        self.dst_path = onnx_path
        self.onnx_weight_path = f'{onnx_path}.data'
        self.onnx_weight_offset = 0

    def make_external(self, name, data, shape):
        # write to external weight
        if isinstance(data, list):
            length = sum(self.onnx_weight.write(item.tobytes()) for item in data)
        else:
            length = self.onnx_weight.write(data.tobytes())
        location = os.path.basename(self.onnx_weight_path)
        offset = self.onnx_weight_offset
        self.onnx_weight_offset += length
        tensor = onnx.TensorProto()
        tensor.name = name
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.dims.extend(shape)
        # external info
        tensor.data_location = onnx.TensorProto.EXTERNAL
        for k, v in { "location": location, "offset": offset, "length": length }.items():
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)
        self.onnx_model.graph.initializer.append(tensor)

    def build_weight(self, name, has_bias, ic, oc):
        assert(name in self.weight_ops)
        linear = self.weight_ops[name]
        assert(linear.in_features == ic and
               linear.out_features == oc and
               (linear.bias is not None) == has_bias)
        weight_name, bias_name = f'{name}_weight', f'{name}_bias'
        weight = linear.weight.data.transpose(1, 0).flatten().float().numpy()
        self.make_external(weight_name, weight, [ic, oc])
        if has_bias:
            bias = linear.bias.data.flatten().float().numpy()
            self.make_external(bias_name, bias, [oc])
        return weight_name, bias_name

    def indexed_bank_nodes(self, name, inputs, input_mode, target):
        """Standard ONNX reference; preserve every original projection weight."""
        import numpy as np
        from onnx import helper
        bank = self.weight_ops[name]
        ic, oc = bank[0][1].in_features, bank[0][1].out_features
        weight_name = f'{name}_weight'
        weights = [linear.weight.detach().transpose(0, 1).float().numpy() for _, linear in bank]
        self.make_external(weight_name, weights, [len(bank), ic, oc])
        ids = f'{name}/ids'
        selected_weight, expanded_input = f'{name}/weight', f'{name}/input'
        product, squeezed = f'{name}/product', f'{name}/squeezed'
        axes_name, squeeze_axes_name = f'{name}/input_axes', f'{name}/output_axes'
        axes = [-2, -3] if input_mode == 0 else [-2]
        nodes = [
            helper.make_node('Cast', [inputs[1]], [ids], to=onnx.TensorProto.INT64),
            helper.make_node('Gather', [weight_name, ids], [selected_weight], axis=0),
            helper.make_node('Constant', [], [axes_name], value=helper.make_tensor(
                axes_name, onnx.TensorProto.INT64, [len(axes)], axes)),
            helper.make_node('Unsqueeze', [inputs[0], axes_name], [expanded_input]),
            helper.make_node('MatMul', [expanded_input, selected_weight], [product]),
            helper.make_node('Constant', [], [squeeze_axes_name], value=helper.make_tensor(
                squeeze_axes_name, onnx.TensorProto.INT64, [1], [-2])),
        ]
        has_bias = any(linear.bias is not None for _, linear in bank)
        nodes.append(helper.make_node('Squeeze', [product, squeeze_axes_name], [squeezed if has_bias else target]))
        if has_bias:
            bias_name, selected_bias = f'{name}_bias', f'{name}/bias'
            biases = [linear.bias.detach().float().numpy() if linear.bias is not None
                      else np.zeros(oc, dtype=np.float32) for _, linear in bank]
            self.make_external(bias_name, biases, [len(bank), oc])
            nodes += [helper.make_node('Gather', [bias_name, ids], [selected_bias], axis=0),
                      helper.make_node('Add', [squeezed, selected_bias], [target])]
        return nodes

    def rebuild(self):
        from onnx import helper
        new_nodes = []
        self.onnx_weight = open(self.onnx_weight_path, 'wb')
        for node in self.onnx_model.graph.node:
            if node.op_type == 'FakeLinear':
                attributes = {a.name: a for a in node.attribute}
                name = attributes.get('name').s.decode('utf-8')
                has_bias = attributes.get('has_bias').i
                ic = attributes.get('in_features').i
                oc = attributes.get('out_features').i
                weight, bias = self.build_weight(name, has_bias, ic, oc)
                if has_bias:
                    # fakelinear -> matmul + add
                    middle_tensor = f'{name}_matmul'
                    new_nodes.append(helper.make_node('MatMul', [node.input[0], weight], [middle_tensor], name))
                    new_nodes.append(helper.make_node('Add', [middle_tensor, bias], node.output, f'{name}/Add'))
                else:
                    # fakelinear -> matmul
                    new_nodes.append(helper.make_node('MatMul', [node.input[0], weight], node.output, name))
            elif node.op_type == 'IndexedLinear':
                from onnx import helper
                attributes = {a.name: a for a in node.attribute}
                name = attributes['name'].s.decode('utf-8')
                input_mode = attributes['input_mode'].i
                activation = attributes['fusedActivation'].i if 'fusedActivation' in attributes else -1
                if activation not in (-1, IndexedLinear.SILU):
                    raise ValueError(f'Unsupported IndexedLinear UnaryOpOperation: {activation}')
                fused = activation != -1
                up_name = attributes['up_name'].s.decode('utf-8') if 'up_name' in attributes else ''
                gate, up = f'{name}/gate', f'{name}/up'
                target = f'{name}/gate_up' if fused else node.output[0]
                if up_name:
                    target = gate
                new_nodes.extend(self.indexed_bank_nodes(name, node.input, input_mode, target))
                if fused:
                    if up_name:
                        new_nodes.extend(self.indexed_bank_nodes(up_name, node.input, input_mode, up))
                    else:
                        split_size = self.weight_ops[name][0][1].out_features // 2
                        split = f'{name}/split'
                        new_nodes += [helper.make_node('Constant', [], [split], value=helper.make_tensor(
                            split, onnx.TensorProto.INT64, [2], [split_size, split_size])),
                            helper.make_node('Split', [target, split], [gate, up], axis=-1)]
                    sigmoid, silu = f'{name}/sigmoid', f'{name}/silu'
                    new_nodes += [helper.make_node('Sigmoid', [gate], [sigmoid]),
                                  helper.make_node('Mul', [gate, sigmoid], [silu]),
                                  helper.make_node('Mul', [silu, up], node.output)]
            else:
                new_nodes.append(node)
        self.onnx_weight.close()
        del self.onnx_model.graph.node[:]
        self.onnx_model.graph.node.extend(new_nodes)
        onnx.save(self.onnx_model, self.dst_path)
        return self.onnx_weight_path
