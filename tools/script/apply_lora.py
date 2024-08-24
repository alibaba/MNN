import os
import json
import argparse

class Base:
    def __init__(self, path, fuse_lora):
        self.fuse_lora = fuse_lora
        self.load(path)

    def __str__(self):
        return str(self.lora_keys)

    def load(self, path):
        self.base_model = json.load(open(path, 'rt'))

    def build_conv(self, input_index, output_name, dims, weight, mul_scale = 1.0):
        output_index = len(self.base_model['tensorName'])
        oc, ic = dims
        bias = [0.0 for i in range(oc)]
        if mul_scale != 1.0:
            weight = [w * mul_scale for w in weight]
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
        self.base_model['oplists'].insert(self.idx, op)
        self.idx += 1
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
            "main": { "opType": 0, "T": "DT_FLOAT", "activationType": 0 },
            "defaultDimentionFormat": "NHWC"
        }
        self.base_model['oplists'].insert(self.idx, op)
        self.idx += 1
        self.base_model['tensorName'].append(output_name)
        return output_index

    def replace_input(self, origin_idx, new_idx):
        for op in self.base_model['oplists']:
            if op['type'] == 'ConvertTensor' and origin_idx in op['inputIndexes']:
                op['inputIndexes'] = [new_idx]

    def apply_lora(self, op, lora):
        names = op['name'].split('/')
        mul_scale = lora.scale
        tag = names[1].split('.')[1] + names[3]
        lora_a, lora_b = lora.get_lora(tag)
        input_index = op['inputIndexes'][0]
        outpt_index = op['outputIndexes'][0]
        if self.fuse_lora:
            w = (lora_a @ lora_b)
            weight = w.reshape(-1).tolist()
            b_out = self.build_conv(input_index, f'{tag}_B', w.shape, weight, mul_scale)
            n_out = self.build_binary(0, [outpt_index, b_out], f'{tag}_add')
            self.replace_input(outpt_index, n_out)
            return
        # lora_B @ lora_A @ x -> lora_B @ (lora_A @ x)
        a_out = self.build_conv(input_index, f'{tag}_A', list(lora_a.shape), lora_a.flatten().tolist())
        b_out = self.build_conv(a_out, f'{tag}_B', list(lora_b.shape), lora_b.flatten().tolist(), mul_scale)
        n_out = self.build_binary(0, [outpt_index, b_out], f'{tag}_add')
        self.replace_input(outpt_index, n_out)

    def apply(self, lora, out):
        ops = []
        for i in range(len(self.base_model['oplists'])):
            op = self.base_model['oplists'][i]
            if op['type'] == 'Convolution':
                if lora.has_lora(op['name']):
                    self.idx = i + 1
                    self.apply_lora(op, lora)
        with open(out, 'w', encoding='utf-8') as file:
            json.dump(self.base_model, file, ensure_ascii=False, indent=4)

class LoRA:
    def __init__(self, path, scale):
        self.lora_A = {}
        self.lora_B = {}
        self.lora_keys = set()
        self.scale = scale
        self.load(path)

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
            print(self.scale)
            path = os.path.join(base_dir, 'adapter_model.safetensors')
        from safetensors import safe_open
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
                    self.lora_B[tag] = tensor

def main(args):
    base = Base(args.base, args.fuse)
    lora = LoRA(args.lora, args.scale)
    base.apply(lora, args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='apply_lora', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--base', type=str, required=True, help='base model json path.')
    parser.add_argument('--lora', type=str, required=True, help='lora dir path or *.safetensors path.')
    parser.add_argument('--scale', type=float, default=4.0, help='lora scale: `alpha/r`.')
    parser.add_argument('--fuse', type=bool, default=False, help='fuse A and B.')
    parser.add_argument('--out', type=str, default='lora.json', help='out file name.')
    args = parser.parse_args()
    main(args)
