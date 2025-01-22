import json
import sys

def treatQuant(quantalgo, limitsize):
    if 'layer' not in quantalgo:
        return
    layers = quantalgo['layer']
    for layer in layers:
        if 'opName' not in layer:
            continue
        name = layer['opName']
        if 'conv' not in layer:
            continue
        conv = layer['conv']
        inputChannel = conv['inputChannel']
        outputChannel = conv['outputChannel']
        kernelSize = conv['kernelSize']
        weight = layer['weight'][0]
        size = inputChannel * outputChannel * kernelSize[0] * kernelSize[1]
        if weight['blockSize'] <= 0:
            print('Set ' + name + " block size to 128")
            weight['blockSize'] = 128
        if size < limitsize:
            print('Skip ' + name + " quant, becuase size < %d" %limitsize)
            weight['bits'] = 0
        elif weight['blockSize'] != 0 and kernelSize[0] * kernelSize[1] != 1:
            print('Skip ' + name + " quant, because not 1x1 conv with block")
            weight['bits'] = 0

    return


def mainFunciton():
    if len(sys.argv) < 3:
        print("Usage: python user_quant_modify_demo.py input.json output.json")
        return
    compress = {}
    with open(sys.argv[1]) as f:
        compress = json.loads(f.read())
    if 'algo' not in compress:
        return
    algos = compress['algo']
    for algo in algos:
        if 'type' not in algo:
            continue
        if algo['type'] == 'QUANTIZE':
            if 'quantParams' in algo:
                treatQuant(algo['quantParams'], 1024)
            break
    with open(sys.argv[2], 'w') as f:
        f.write(json.dumps(compress, indent=4))

if __name__ == '__main__':
    mainFunciton()
