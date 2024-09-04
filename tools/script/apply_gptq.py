import json
import torch
import argparse

class MNNWeight:
    def __init__(self, name, external, weight_elements):
        self.name = name
        self.external = external
        self.quant_bits = 4
        if round(weight_elements / external[1]) == 2:
            self.quant_bits = 4
            self.a_min = -8
        else:
            self.quant_bits = 8
            self.a_min = -128
        self.parse_name()

    def __repr__(self) -> str:
        return f'{self.layer_id}.{self.op_id}.{self.block_id}, {self.external}'

    def parse_name(self):
        parts = self.name.split('/')
        if len(parts) > 4:
            self.layer_id = parts[1].split('.')[1]
            self.op_id = parts[2] + '.' + parts[3]
            self.block_id = parts[-1].split('__')[-1]
        else:
            self.layer_id = -1
            self.op_id = parts[2]
            self.block_id = parts[-1].split('__')[-1]

    def key(self):
        if self.layer_id == -1: return self.op_id
        return f'{self.layer_id}.{self.op_id}'
    def idx(self): return int(self.block_id)
    def offset(self): return self.external[0]
    def weight_size(self): return self.external[1]
    def scale_size(self): return self.external[2]

def weight_reorder(qweight, bits=4, group_size=128):
    oc = qweight.shape[-1]
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight, (2 ** bits) - 1, out=weight)
    weight = weight.reshape(-1, oc).transpose(1, 0)
    if bits == 8:
        weight = weight.to(torch.uint8)
        return weight
    weight = weight.reshape(-1, 2).to(torch.uint8)
    weight = weight[:, 0] * 16 + weight[:, 1]
    return weight

class MNNModel:
    def __init__(self, model, weight):
        self.mnn_graph = json.load(open(model, 'rt'))
        self.external_weight = weight
        self.parse_conv()

    def parse_conv(self):
        self.weights = []
        for op in self.mnn_graph['oplists']:
            if op['type'] == 'Convolution':
                name = op['name']
                external = op['main']['external']
                weight_elements = op['main']['common']['outputCount'] * op['main']['common']['inputCount']
                self.weights.append(MNNWeight(name, external, weight_elements))

    def apply_weight_split(self, gptq_tensor):
        bin_file = open(self.external_weight, 'r+b')
        for mnn_weight in self.weights:
            idx = mnn_weight.idx()
            gptq_weight = gptq_tensor.get(mnn_weight.key())
            if gptq_weight is None: continue
            print(f'write {mnn_weight.key()}.{idx} ... ', end='')
            weight = gptq_weight.weight(idx)
            scale = gptq_weight.scale(idx).float()
            # write weight data
            weight = weight_reorder(weight, mnn_weight.quant_bits)
            weight_bytes = weight.numpy().tobytes()
            weight_size = mnn_weight.weight_size()
            header_len = weight_size - len(weight_bytes)
            assert(header_len > 0)
            bin_file.seek(mnn_weight.offset() + header_len)
            bin_file.write(weight_bytes)
            scale_size = mnn_weight.scale_size()
            is_asy = scale.numel() * scale.element_size() < scale_size
            # write scale data
            if is_asy:
                zeros = mnn_weight.a_min * scale
                scale = torch.stack([zeros, scale], axis=-1)
            scale_bytes = scale.numpy().tobytes()
            assert(scale_size == len(scale_bytes))
            bin_file.write(scale_bytes)
            print('Done!')
            # break
        bin_file.close()

    def apply_weight(self, gptq_tensor):
        bin_file = open(self.external_weight, 'r+b')
        for mnn_weight in self.weights:
            gptq_weight = gptq_tensor.get(mnn_weight.key())
            if gptq_weight is None: continue
            print(f'write {mnn_weight.key()} ... ', end='')
            # print(f'mnn_weight.quant_bits = {mnn_weight.quant_bits}')
            weight = gptq_weight.qweight
            scale = gptq_weight.scales.float().transpose(1, 0)
            # write weight data
            weight = weight_reorder(weight, mnn_weight.quant_bits)
            weight_bytes = weight.numpy().tobytes()
            weight_size = mnn_weight.weight_size()
            header_len = weight_size - len(weight_bytes)
            assert(header_len > 0)
            bin_file.seek(mnn_weight.offset() + header_len)
            bin_file.write(weight_bytes)
            scale_size = mnn_weight.scale_size()
            is_asy = scale.numel() * scale.element_size() < scale_size
            # write scale data
            if is_asy:
                zeros = mnn_weight.a_min * scale
                scale = torch.stack([zeros, scale], axis=-1)
            scale_bytes = scale.numpy().tobytes()
            assert(scale_size == len(scale_bytes))
            bin_file.write(scale_bytes)
            print('Done!')
        bin_file.close()

    def apply(self, gptq_tensor):
        if self.weights[0].block_id.isdigit():
            self.apply_weight_split(gptq_tensor)
        else:
            self.apply_weight(gptq_tensor)

class GPTQWeight:
    def __init__(self, name):
        self.name = name

    def __repr__(self) -> str:
        if hasattr(self, 'qweight'):
            return f'{self.name}, {self.qweight.shape}, {self.scales.shape}'
        return 'None'

    def add(self, name, tensor):
        setattr(self, name, tensor)

    def weight(self, idx):
        shape = self.qweight.shape
        if len(shape) == 2:
            ic, oc = shape
            self.qweight = self.qweight.reshape(ic//16, 16, oc)
        return self.qweight[idx]

    def scale(self, idx):
        return self.scales[idx]

class GPTQTensor:
    def __init__(self, file):
        self.file = file
        self.load()

    def prefix(self, name):
        splits = name.split('.')
        if 'lm_head' in splits[0] and len(splits) == 2:
            return splits[0], splits[1]
        if len(splits) < 5:
            return None, None
        pre = f'{splits[2]}.{splits[3]}.{splits[4]}'
        suf = splits[-1]
        return pre, suf

    def __repr__(self) -> str:
        return self.weight_dict.__repr__()

    def get(self, key : str):
        if key in self.weight_dict:
            return self.weight_dict[key]
        return None

    def load(self):
        self.weight_dict = dict()
        from safetensors import safe_open
        with safe_open(self.file, framework="pt") as f:
            for k in f.keys():
                p, s = self.prefix(k)
                if p is None: continue
                if s not in ['qweight', 'scales']: continue
                if p not in self.weight_dict:
                    self.weight_dict[p] = GPTQWeight(p)
                self.weight_dict[p].add(s, f.get_tensor(k))

def main(args):
    mnn_model = MNNModel(args.mnn_graph, args.mnn_weight)
    gptq_weight = GPTQTensor(args.gptq_tensor)
    mnn_model.apply(gptq_weight)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='apply_gptq', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mnn_graph', type=str, required=True, help='mnn graph json path.')
    parser.add_argument('--mnn_weight', type=str, required=True, help='mnn weight file path.')
    parser.add_argument('--gptq_tensor', type=str, required=True, help='gptq tensor path.')
    args = parser.parse_args()
    main(args)
