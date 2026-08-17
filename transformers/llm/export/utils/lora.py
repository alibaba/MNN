import os
import json
from safetensors import safe_open

class LoRA:
    def __init__(self, lora_path, scale = 4.0):
        self.lora_A = {}
        self.lora_B = {}
        self.lora_keys = set()
        self.scale = scale
        self.load(lora_path)

    def __str__(self):
        return str(self.lora_keys)

    def has_lora(self, op_name):
        if op_name[0] != '/':
            return False
        for key in self.lora_keys:
            if key in op_name:
                return True
        return False

    def has_tag(self, tag):
        # lora_keys is global across layers; a LoRA file may cover only some
        # layers, so per-op targeting must check the exact layer+name tag.
        return tag in self.lora_A and tag in self.lora_B

    def get_lora(self, tag):
        lora_a, lora_b = self.lora_A[tag], self.lora_B[tag]
        return lora_a, lora_b

    def load(self, path):
        if os.path.isdir(path):
            base_dir = path
            config = json.load(open(os.path.join(base_dir, 'adapter_config.json'), 'rt'))
            lora_alpha = config['lora_alpha']
            r = config['r']
            self.scale = float(lora_alpha) / r
            path = os.path.join(base_dir, 'adapter_model.safetensors')
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                names = k.split('.')
                ni = next((i for i, n in enumerate(names) if n.startswith('lora_')), -1)
                if ni < 1 or names[ni] not in ('lora_A', 'lora_B'):
                    continue

                name, key = names[ni], names[ni - 1]
                layer = next(n for n in reversed(names[:ni - 1]) if n.isdigit())
                tag = layer + key
                tensor = f.get_tensor(k).float()
                self.lora_keys.add(key)
                if 'lora_A' == name:
                    self.lora_A[tag] = tensor
                else:
                    self.lora_B[tag] = tensor * self.scale

    def build_tensor(self, name):
        self.base_model['tensorName'].append(name)
        return len(self.base_model['tensorName']) - 1

    def build_conv(self, input_index, output_name, dims, weight):
        output_index = self.build_tensor(output_name)
        oc, ic = dims
        bias = [0.0 for i in range(oc)]
        op = {
            'type': 'Convolution',
            'name': output_name,
            'inputIndexes': [input_index],
            'outputIndexes': [ output_index ],
            'main_type': 'Convolution2D',
            'main': {
                'common': {
                    'dilateX': 1, 'dilateY': 1, 'strideX': 1, 'strideY': 1,
                    'kernelX': 1, 'kernelY': 1, 'padX': 0, 'padY': 0, 'group': 1,
                    'outputCount': oc, 'relu': False, 'padMode': 'CAFFE',
                    'relu6': False, 'inputCount': ic, 'hasOutputShape': False
                },
                "weight": weight,
                "bias": bias
            },
            'defaultDimentionFormat': 'NC4HW4'
        }
        self.new_ops.append(op)
        return output_index

    def build_binary(self, op_type, input_indexes, output_name, output_index = None):
        # 0: Add, 2: Mul
        if output_index is None:
            output_index = self.build_tensor(output_name)
        op = {
            "type": "BinaryOp",
            "name": output_name,
            "inputIndexes": input_indexes,
            "outputIndexes": [ output_index ],
            "main_type": "BinaryOp",
            "main": { "opType": op_type, "T": "DT_FLOAT", "activationType": 0 },
            "defaultDimentionFormat": "NC4HW4"
        }
        self.new_ops.append(op)
        return output_index

    def build_unary_silu(self, input_index, output_name):
        # json2mnn resolves enum fields by name, so opType must stay a string.
        output_index = self.build_tensor(output_name)
        op = {
            "type": "UnaryOp",
            "name": output_name,
            "inputIndexes": [ input_index ],
            "outputIndexes": [ output_index ],
            "main_type": "UnaryOp",
            "main": { "opType": "SILU", "T": "DT_FLOAT" },
            "defaultDimentionFormat": "NC4HW4"
        }
        self.new_ops.append(op)
        return output_index

    def replace_input(self, origin_idx, new_idx):
        # Rewire every consumer of the base projection output to read the LoRA
        # sum instead. Replace only the matching index (consumers may have other
        # inputs, e.g. the residual LayerNorm / MulSilu BinaryOp that consume a
        # projection output directly when FuseTransformerC4 is on). This iterates
        # the ORIGINAL oplists; the freshly built A/B/ADD ops are still in
        # self.new_ops, so the ADD (which legitimately consumes origin_idx) is
        # never rewired and no cycle forms. The injected ops are NC4HW4 so the
        # ADD keeps the C4 layout ([seq, C, 1, 1]) of the patched tensor
        # instead of broadcasting mismatched NHWC shapes into it.
        for op in self.base_model['oplists']:
            idxs = op.get('inputIndexes', [])
            if origin_idx in idxs:
                op['inputIndexes'] = [new_idx if x == origin_idx else x for x in idxs]

    def build_lora_chain(self, input_index, output_index, tag):
        # lora_B @ lora_A @ x -> lora_B @ (lora_A @ x), added onto output_index
        lora_a, lora_b = self.get_lora(tag)
        a_out = self.build_conv(input_index, f'{tag}_A', list(lora_a.shape), lora_a.flatten().tolist())
        b_out = self.build_conv(a_out, f'{tag}_B', list(lora_b.shape), lora_b.flatten().tolist())
        return self.build_binary('ADD', [output_index, b_out], f'{tag}_add')

    def apply_lora(self, op):
        names = op['name'].split('/')
        tag = names[1].split('.')[1] + names[3]
        if not self.has_tag(tag):
            return
        outpt_index = op['outputIndexes'][0]
        n_out = self.build_lora_chain(op['inputIndexes'][0], outpt_index, tag)
        self.replace_input(outpt_index, n_out)

    def fused_members(self, op):
        # FuseTransformerC4 records the member projection names in convs order
        # in the op name, joined by '+':
        # /layers.0/self_attn/q_proj/Linear+.../k_proj/Linear+.../v_proj/Linear
        parts = op['name'].split('+')
        layer = parts[0].split('/')[1].split('.')[1]
        members = [part.split('/')[-2] for part in parts]
        return layer, members

    def unfold_fused_ln(self, op):
        # The folded binary RMSNorm normalized the hidden inline; restore it as
        # a LayerNorm op so LoRA A/B can read the normalized value too. The
        # restored LayerNorm keeps multiple consumers, so json2mnn's fold pass
        # will not fold it back.
        main = op['main']
        ln = main.pop('ln')
        main['has_ln'] = False
        base_name = op['name'].split('+')[0]
        norm_out = self.build_tensor(f'{base_name}_ln_norm')
        ln_op = {
            'type': 'LayerNorm',
            'name': f'{base_name}_ln',
            'inputIndexes': list(op['inputIndexes']),
            'outputIndexes': [op['outputIndexes'][-1], norm_out],
            'main_type': 'LayerNorm',
            'main': ln,
            'defaultDimentionFormat': 'NC4HW4'
        }
        # The producer must precede its consumer in oplists: the fused op (and
        # every appended LoRA chain) reads norm_out, and apply() has already
        # appended the fused op as the last element, so slot the LayerNorm in
        # right before it. A trailing producer makes the scheduler shape-compute
        # the fused op from a stale tensor and the members come out as
        # [1, C, 2, 2] garbage.
        self.new_ops.insert(len(self.new_ops) - 1, ln_op)
        op['inputIndexes'] = [norm_out]
        op['outputIndexes'] = op['outputIndexes'][:-1]
        return norm_out

    def unfold_silu_mul(self, op, layer, members, targeted):
        # SiLU(gate) * up is computed inline by the fused op, so the gate/up
        # outputs do not exist as tensors. Re-materialize them as plain fused
        # outputs (the fused GEMM is kept) and rebuild the elementwise tail; the
        # Mul re-produces the original fused output tensor so downstream
        # consumers stay wired.
        main = op['main']
        main['act_silu_mul'] = False
        prefix = op['name'].split('+')[0].rsplit('/', 1)[0]
        member_out = [self.build_tensor(f'{prefix}/{name}_fused_out') for name in members]
        swiglu_out = op['outputIndexes'][0]
        op['outputIndexes'] = member_out
        input_index = op['inputIndexes'][0]
        finals = []
        for i, name in enumerate(members):
            if i in targeted:
                finals.append(self.build_lora_chain(input_index, member_out[i], layer + name))
            else:
                finals.append(member_out[i])
        silu_out = self.build_unary_silu(finals[0], f'{prefix}/{members[0]}_silu')
        self.build_binary('MUL', [silu_out, finals[1]], f'{prefix}/{members[0]}_silu_mul', output_index=swiglu_out)

    def apply_fused(self, op):
        layer, members = self.fused_members(op)
        targeted = [i for i, name in enumerate(members) if self.has_tag(layer + name)]
        if not targeted:
            return
        main = op['main']
        convs = main.get('convs', [])
        if len(members) != len(convs):
            raise RuntimeError(
                f'FusedLinear "{op["name"]}" has {len(convs)} convs but {len(members)} member names; '
                'rebuild MNNConvert so FuseTransformerC4 records member names in the op name')
        if main.get('has_ln'):
            self.unfold_fused_ln(op)
        if main.get('act_silu_mul'):
            self.unfold_silu_mul(op, layer, members, targeted)
            return
        input_index = op['inputIndexes'][0]
        for i in targeted:
            tag = layer + members[i]
            n_out = self.build_lora_chain(input_index, op['outputIndexes'][i], tag)
            self.replace_input(op['outputIndexes'][i], n_out)

    def apply(self, base_path, out):
        self.base_model = json.load(open(base_path, 'rt'))
        self.new_ops = []
        for i in range(len(self.base_model['oplists'])):
            op = self.base_model['oplists'][i]
            self.new_ops.append(op)
            if op['type'] == 'Convolution':
                if self.has_lora(op['name']):
                    self.apply_lora(op)
            elif op['type'] == 'FusedLinear':
                self.apply_fused(op)
        self.base_model['oplists'] = self.new_ops
        with open(out, 'w', encoding='utf-8') as file:
            json.dump(self.base_model, file, ensure_ascii=False, indent=4)
        return out
