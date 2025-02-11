import sys
import os
import json
import functools
mnnconvert = './MNNConvert'
if os.path.exists('MNNConvert'):
    mnnconvert = './MNNConvert'
elif os.path.exists('MNNConvert.exe'):
    mnnconvert = 'MNNConvert.exe'
else:
    mnnconvert = 'mnnconvert'

print('Use MNN Convert: ', mnnconvert)
import argparse
class QuantInfo:
    def __init__(self, jsonPath):
        compress = None
        with open(jsonPath) as f:
            compress = json.loads(f.read())
        if 'algo' not in compress:
            print("Invalid Json")
            return
        self.dstjson = jsonPath
        algos = compress['algo']
        self.compress = compress
        self.conv_1x1 = []
        self.conv_other = []

        for algo in algos:
            if 'type' not in algo:
                continue
            if algo['type'] == 'QUANTIZE':
                if 'quantParams' in algo:
                    quantalgo = algo['quantParams']
                    if 'layer' in quantalgo:
                        layers = quantalgo['layer']
                        for layer in layers:
                            if 'opName' not in layer:
                                continue
                            if 'conv' not in layer:
                                continue
                            conv = layer['conv']
                            kernelSize = conv['kernelSize']
                            if kernelSize[0] * kernelSize[1] == 1:
                                self.conv_1x1.append(layer)
                            else:
                                self.conv_other.append(layer)
                break
        def computeConvSize(layer):
            conv = layer['conv']
            kernelSize = conv['kernelSize']
            inputChannel = conv['inputChannel']
            outputChannel = conv['outputChannel']
            return kernelSize * inputChannel * outputChannel
        def sortconv(A, B):
            return computeConvSize(A) < computeConvSize(B)
        self.conv_1x1.sort(key=functools.cmp_to_key(sortconv))
    def update(self):
        with open(self.dstjson, 'w') as f:
            f.write(json.dumps(self.compress, indent=4))
    def setBlock(self, block):
        for c in self.conv_1x1:
            c['weight'][0]['blockSize'] = block
        for c in self.conv_other:
            c['weight'][0]['bits'] = 0
    def mutableSize(self):
        return len(self.conv_1x1)
    def setMultiBlock(self, seprate, block0, block1):
        if seprate >= len(self.conv_1x1):
            return
        for i in range(0, seprate):
            self.conv_1x1[i]['weight'][0]['blockSize'] = block0
        for i in range(seprate, len(self.conv_1x1)):
            self.conv_1x1[i]['weight'][0]['blockSize'] = block1
    def setBits(self, pos, bits):
        if pos >= len(self.conv_1x1):
            return
        self.conv_1x1[pos]['weight'][0]['bits'] = bits


def getRate(loginfo):
    lines = loginfo.split('\n')
    lines = list(filter(lambda x:x.find('TESTERROR')>=0, lines))
    if len(lines) == 0:
        return 0.0
    rate = 0.0
    for line in lines:
        content = line.split('absMaxV:')[1].replace(' ', '')
        value = content.split('-DiffMax')
        maxv = float(value[0])
        diffmax = float(value[1])
        if maxv > 0.01 and diffmax / maxv > rate:
            rate = diffmax / maxv
    return rate

forwardJson = '.tmp.json'
def initDynamicQuantJson(memory):
    v = {}
    v['backend'] = 0
    v['mode'] = 4
    v['memory'] = memory
    v['precision'] = 1
    v['hints'] = [5, 1]
    print('Create test config: ', forwardJson, ", memory=", memory)
    with open(forwardJson, 'w') as f:
        f.write(json.dumps(v))

def testJson(model, dstjson, testdir, dstmodel):
    cmd = mnnconvert + ' -f MNN --modelFile ' + model + ' --MNNModel ' + dstmodel
    cmd += ' --weightQuantBits=8 --weightQuantAsymmetric=0 '
    cmd += ' --compressionParamsFile ' + dstjson
    cmd += ' --testdir ' + testdir
    cmd += ' --thredhold 0.001'
    cmd += ' --testconfig ' + forwardJson
    cmd += ' --alignDenormalizedValue 0 '
    info = os.popen(cmd).read()
    return getRate(info)

class TestSuit:
    def __init__(self, model, dstjson, testdir, dstmodel):
        self.model = model
        self.dstjson = dstjson
        self.testdir = testdir
        self.dstmodel = dstmodel
    def test(self):
        return testJson(self.model, self.dstjson, self.testdir, self.dstmodel)

def findBestBits(info, test, targetRate):
    info.setBlock(64)
    info.update()
    rate = test.test()
    if rate > targetRate:
        return rate
    length = info.mutableSize()
    tested = False
    for i in range(length):
        info.setBits(i, 4)
        info.update()
        rate = test.test()
        tested = True
        if rate > targetRate:
            # roll back to 8
            info.setBits(i, 8)
            tested = False
        else:
            print('Set %d layer to 4 bits' %i, ', rate=%f' %rate)
    if not tested:
        info.update()
        rate = test.test()
    return rate

def findBestBlock(info, test, targetRate):
    validBlock = 0
    bestBlock = 256
    bestRate = 1.0
    rate = 1.0
    for block in (256, 128, 64, 32):
        info.setBlock(block)
        info.update()
        rate = test.test()
        print('block=%d,' %block + ' rate=%f' %rate)
        if rate < bestRate:
            bestRate = rate
            bestBlock = block
        if rate < targetRate:
            validBlock = block
            break
    if validBlock < 256 and rate < targetRate:
        largeBlock = validBlock * 2
        length = info.mutableSize()
        # 2-div check
        sta = 0
        pos = length // 2
        fin = length
        while pos > sta and pos < fin:
            info.setMultiBlock(pos, largeBlock, validBlock)
            info.update()
            rate = test.test()
            print('len:%d' %pos, ', rate:%f' %rate)
            if rate < targetRate:
                sta = pos
            else:
                fin = pos
            pos = (sta + fin + 1) // 2
        if sta != pos:
            info.setMultiBlock(sta, largeBlock, validBlock)
            info.update()
            rate = test.test()
    else:
        info.setBlock(bestBlock)
        info.update()
        rate = test.test()
    return rate

def skipMoreOps(info, test, targetRate, current):
    length = info.mutableSize()
    for il in range(length):
        i = length - il - 1
        info.setBits(i, 0)
        info.update()
        rate = test.test()
        if rate < current:
            print("Skip quant for ", i, ", rate=", rate)
            current = rate
        else:
            info.setBits(i, 8)
        if rate <= targetRate:
            break
    return rate


def mainFunction():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)    
    parser.add_argument('--model', type=str, required=True,help='src float mnn model')
    parser.add_argument('--quant_model', type=str, required=True,help='dst quant mnn model')
    parser.add_argument('--test_dir', type=str, required=True,help='test dir')
    parser.add_argument('--rate', type=float, default=0.05,help='test rate')
    parser.add_argument('--select_bits', type=int, default=1,help='Try set layer as 4 bits')
    parser.add_argument('--select_block', type=int, default=1,help='Try select blocks')
    args = parser.parse_args()
    rate = args.rate
    model = args.model
    dstmodel = args.quant_model
    testdir = args.test_dir
    print("Target Rate is %f" %rate)
    targetRate = rate
    print('src: ', model, ", dst:", dstmodel)
    dstjson = dstmodel + '.json'
    initDynamicQuantJson(2)
    if os.path.exists(dstjson):
        print("Erase old file: ", dstjson)
        os.remove(dstjson)
    if not os.path.exists(model):
        print(model, " not exist")
        return
    if not os.path.exists(testdir) or not os.path.isdir(testdir):
        print(testdir, " not exist or is not dir")
        return
    if not os.path.exists(os.path.join(testdir, "input.json")):
        print(testdir, " not has input.json")
        return
    test = TestSuit(model, dstjson, testdir, dstmodel)
    rate = test.test()
    print('Init rate: %f' %rate)
    info = QuantInfo(dstjson)
    if args.select_bits > 0:
        rate = findBestBits(info, test, targetRate)
    if args.select_block > 0:
        rate = findBestBlock(info, test, targetRate)
    if rate > targetRate:
        initDynamicQuantJson(0)
        rate = test.test()
    if rate > targetRate:    
        rate = skipMoreOps(info, test, targetRate, rate)
    print("rate=%f" %rate, ", save to " + dstmodel)
    originSize = os.path.getsize(model) / 1024.0 / 1024.0
    dstSize = os.path.getsize(dstmodel) / 1024.0 / 1024.0

    print("Compress From %f MB " %originSize, ' to %f MB' %dstSize)
 

if __name__ == '__main__':
    mainFunction()
