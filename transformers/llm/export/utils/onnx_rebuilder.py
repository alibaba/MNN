import os
import onnx

class OnnxRebuilder:
    def __init__(self, onnx_path, weight_ops):
        self.weight_ops = weight_ops
        self.onnx_model = onnx.load(onnx_path)
        self.dst_path = onnx_path
        self.onnx_weight_path = f'{onnx_path}.data'
        self.onnx_weight_offset = 0

    def make_external(self, name, data, shape):
        # write to external weight
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
            else:
                new_nodes.append(node)
        self.onnx_weight.close()
        del self.onnx_model.graph.node[:]
        self.onnx_model.graph.node.extend(new_nodes)
        onnx.save(self.onnx_model, self.dst_path)
        return self.onnx_weight_path