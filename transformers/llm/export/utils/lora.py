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
                layer, key, name = names[4], names[6], names[7]
                tag = layer + key
                tensor = f.get_tensor(k).float()
                self.lora_keys.add(key)
                if 'lora_A' == name:
                    self.lora_A[tag] = tensor
                else:
                    self.lora_B[tag] = tensor * self.scale

    def build_conv(self, input_index, output_name, dims, weight):
        output_index = len(self.base_model['tensorName'])
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
            'defaultDimentionFormat': 'NHWC'
        }
        self.new_ops.append(op)
        self.base_model['tensorName'].append(output_name)
        return output_index

    def build_binary(self, op_type, input_indexes, output_name):
        # 0: Add, 2: Mul
        output_index = len(self.base_model['tensorName'])
        op = {
            "type": "BinaryOp",
            "name": output_name,
            "inputIndexes": input_indexes,
            "outputIndexes": [ output_index ],
            "main_type": "BinaryOp",
            "main": { "opType": op_type, "T": "DT_FLOAT", "activationType": 0 },
            "defaultDimentionFormat": "NHWC"
        }
        self.new_ops.append(op)
        self.base_model['tensorName'].append(output_name)
        return output_index

    def replace_input(self, origin_idx, new_idx):
        for op in self.base_model['oplists']:
            if op['type'] == 'ConvertTensor' and origin_idx in op['inputIndexes']:
                op['inputIndexes'] = [new_idx]

    def apply_lora(self, op):
        names = op['name'].split('/')
        tag = names[1].split('.')[1] + names[3]
        lora_a, lora_b = self.get_lora(tag)
        input_index = op['inputIndexes'][0]
        outpt_index = op['outputIndexes'][0]
        # lora_B @ lora_A @ x -> lora_B @ (lora_A @ x)
        a_out = self.build_conv(input_index, f'{tag}_A', list(lora_a.shape), lora_a.flatten().tolist())
        b_out = self.build_conv(a_out, f'{tag}_B', list(lora_b.shape), lora_b.flatten().tolist())
        n_out = self.build_binary(0, [outpt_index, b_out], f'{tag}_add')
        self.replace_input(outpt_index, n_out)

    def apply(self, base_path, out):
        self.base_model = json.load(open(base_path, 'rt'))
        self.new_ops = []
        for i in range(len(self.base_model['oplists'])):
            op = self.base_model['oplists'][i]
            self.new_ops.append(op)
            if op['type'] == 'Convolution':
                if self.has_lora(op['name']):
                    self.apply_lora(op)
        self.base_model['oplists'] = self.new_ops
        with open(out, 'w', encoding='utf-8') as file:
            json.dump(self.base_model, file, ensure_ascii=False, indent=4)
        return out
