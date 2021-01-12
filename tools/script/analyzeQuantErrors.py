import json
import os
import re
import math
import shutil

_SPACE_PATTERN = re.compile(r'\s+')

def _load_tensor(tensor_file):
    text = open(tensor_file).read().strip()
    parts = [v for v in _SPACE_PATTERN.split(text)]
    
    numbers_per_line = len(parts)
    index = text.find('\n')
    if index != -1:
        numbers_per_line = len(_SPACE_PATTERN.split(text[:index].strip()))

    return [float(v) for v in parts], numbers_per_line

def _normalize(a):
    sum = 0
    for v in a:
        sum += v * v
    sum = math.sqrt(sum)
    if not sum:
        return a
    return [v/sum for v in a]

def norm(a):
    sum = 0
    for v in a:
        sum += v * v
    return math.sqrt(sum)

def _normalized_distance(a, b):
    assert len(a) == len(b)
    a = _normalize(a)
    b = _normalize(b)

    sum = 0
    for i, va in enumerate(a):
        vb = b[i]
        diff = va - vb
        sum += diff * diff
    return math.sqrt(sum)

def _distance(a, b):
    assert len(a) == len(b)
    # a = _normalize(a)
    # b = _normalize(b)

    sum = 0
    for i, va in enumerate(a):
        vb = b[i]
        diff = va - vb
        sum += diff * diff
    return math.sqrt(sum)

def _cos_distance(a, b):
    norm_a = norm(a)
    norm_b = norm(b)
    sum = 0
    for i, va in enumerate(a):
        vb = b[i]
        sum += va * vb
    return 1 - sum / (norm_a * norm_b)

class Analyzer(object):
    # @param: scale_file - 量化时生成的scale_file
    # @param: normal_output - 运行正常模型时生成的tensor dump目录
    # @param: quant_output - 运行量化模型时生成的tensor dump目录
    def __init__(self, scale_file, normal_output, quant_output):
        self.scale_file = scale_file
        self.normal_output = normal_output
        self.quant_output = quant_output
        self.scales = self._load_scale()

    def _load_scale(self):
        return json.loads(open(self.scale_file).read())

    def execute(self):
        dequant_output = self.quant_output + '-dequant'
        shutil.rmtree(dequant_output, ignore_errors=True)
        os.makedirs(dequant_output, exist_ok=True)

        for op in self.scales:
            if not op['outputs']:
                continue
            outputs = op['outputs']

            for i, output in enumerate(outputs):
                if not output['scales']:
                    continue

                file_name = '%s_%d' % (op['name'].replace('/', '_'), i)
                normal_tensor_file = os.path.join(
                    self.normal_output, file_name)
                quant_tensor_file = os.path.join(self.quant_output, file_name)
                if not os.path.exists(normal_tensor_file):
                    continue
                if not os.path.exists(quant_tensor_file):
                    continue

                normal_tensor, numbers_per_line = _load_tensor(normal_tensor_file)
                quant_tensor, _ = _load_tensor(quant_tensor_file)

                if len(normal_tensor) != len(quant_tensor):
                    print('error: normal tensor file: %s count: %d' % (normal_tensor_file, len(normal_tensor)))
                    print('error: quant tensor file: %s count: %d' % (quant_tensor_file, len(quant_tensor)))
                    sys.exit(1)

                scales = output['scales']
                assert len(normal_tensor) % len(scales) == 0

                plane_size = len(normal_tensor) // len(scales)
                
                dequant_tensor = []
                non_zero_count = 0
                max_value_count = 0
                min_value_count = 0
                max_value = 127
                min_value = -127
                for i, scale in enumerate(scales):
                    plane = quant_tensor[i*plane_size:(i+1)*plane_size]
                    for v in plane:
                        dequant_tensor.append(v * scale)
                        if v:
                            non_zero_count += 1
                            if v == max_value:
                                max_value_count += 1
                            if v == min_value:
                                min_value_count += 1
                
                print(file_name)
                d = _distance(normal_tensor, dequant_tensor)
                normalized_d = _normalized_distance(normal_tensor, dequant_tensor)
                cos_d = _cos_distance(normal_tensor, dequant_tensor)
                print('max rate: %.06f%%' % (max_value_count / non_zero_count * 100))
                print('min rate: %.06f%%' % (min_value_count / non_zero_count * 100))
                print('norm of normal: %.06f' % (norm(normal_tensor)))
                print('norm of quant: %.06f' % (norm(dequant_tensor)))
                print('cos distance: %.08f' % (cos_d))
                print('normalized distance: %.08f' % (normalized_d))
                print('distance: %.08f\n' % (d))

                # Output dequant tensor
                lines = []
                col = numbers_per_line
                row = len(dequant_tensor) // col

                assert len(dequant_tensor) % col == 0
                for i in range(row):
                    parts = []
                    for j in range(col):
                        parts.append(('%f' % dequant_tensor[col*i+j]).rstrip('0').rstrip('.'))
                    lines.append('\t'.join(parts))

                dequant_tensor_file = os.path.join(dequant_output, file_name)
                open(dequant_tensor_file, 'w').write('\n'.join(lines))                    

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scale-file",
                        required=True, help="量化时输出的scale文件")
    parser.add_argument("-n", "--normal-output",
                        required=True, help="运行正常模型时生成的tensor dump目录")
    parser.add_argument("-q", "--quant-output", required=True,
                        help="运行量化模型时生成的tensor dump目录")
    args = parser.parse_args()

    Analyzer(args.scale_file, args.normal_output, args.quant_output).execute()
