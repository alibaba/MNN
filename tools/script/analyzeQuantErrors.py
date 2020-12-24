import json
import os
import re
import math

_SPACE_PATTERN = re.compile(r'\s+')


def _load_tensor(tensor_file):
    text = open(tensor_file).read()
    parts = [v for v in _SPACE_PATTERN.split(text) if v]
    
    return [float(v) for v in parts]

def _normalize(a):
    sum = 0
    for v in a:
        sum += v * v
    sum = math.sqrt(sum)
    if not sum:
        return a
    return [v/sum for v in a]

def _distance(a, b):
    assert len(a) == len(b)
    a = _normalize(a)
    b = _normalize(b)

    sum = 0
    for i, va in enumerate(a):
        vb = b[i]
        diff = va - vb
        sum += diff * diff
    return math.sqrt(sum)


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

                normal_tensor = _load_tensor(normal_tensor_file)
                quant_tensor = _load_tensor(quant_tensor_file)

                assert len(normal_tensor) == len(quant_tensor)

                scales = output['scales']
                assert len(normal_tensor) % len(scales) == 0

                plane_size = len(normal_tensor) // len(scales)
                
                dequant_tensor = []
                for i, scale in enumerate(scales):
                    plane = quant_tensor[i*plane_size:(i+1)*plane_size]
                    for v in plane:
                        dequant_tensor.append(v * scale)
                
                print(file_name)
                d = _distance(normal_tensor, dequant_tensor)
                print('distance: %.08f\n' % (d))

                # for i in range(3):
                #     print('%f, %f, %f' % (normal_tensor[i], dequant_tensor[i], quant_tensor[i]))
                

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
