#!/usr/bin/python3
import sys
import os
import re

total_num = 0

def run_cmd(args):
    cmd = args[0]
    for i in range(1, len(args)):
        cmd += ' ' + args[i]
    stdout = os.popen(cmd).read()
    return stdout

def parseRes(res):
    pattern = re.compile(r'(\d+, \d+\.\d+)\s')
    idxs = set()
    avgp = 0
    items = pattern.findall(res)
    for item in items:
        splitIdx = item.find(',')
        idx = int(item[:splitIdx])
        point = float(item[splitIdx+1:])
        idxs.add(idx)
        avgp += point
    avgp /= len(items) 
    return idxs, avgp

def compare(origin, quant):
    img_dir = '../resource/images'
    for name in os.listdir(img_dir):
        origin_res = run_cmd(['./pictureRecognition.out', origin, img_dir + '/' + name])
        quant_res = run_cmd(['./pictureRecognition.out', quant, img_dir + '/' + name])
        # print(origin_res, quant_res)
        originIdx, originPoint = parseRes(origin_res)
        quantIdx, quantPoint = parseRes(quant_res)
        idxRate = len(originIdx & quantIdx) / max(len(originIdx), len(quantIdx))    
        pointRate = quantPoint / originPoint
        print(name, idxRate, pointRate)
        if idxRate < 0.5 or pointRate < 0.5 or pointRate > 2.0:
            print('False')
            return False
    return True

def test(path):
    global total_num
    total_num += 1
    originModel = path + '/test.mnn'
    quantModel  = './__quantModel.mnn'
    message = run_cmd(['./quantized.out', originModel, quantModel, path + '/test.json'])
    res = True
    try:
        res = compare(originModel, quantModel)
    except:
        print('Quant Error!')
        res = False
    return res 
    
if __name__ == '__main__':
    model_root_dir = sys.argv[1]
    root_dir = os.path.join(model_root_dir, 'TestPTQ')
    print('root: ' + root_dir + '\n')
    gWrong = []
    for name in os.listdir(root_dir):
        if name == '.DS_Store':
            continue
        print(name)
        res = test(root_dir + '/' + name)
        if not res:
            gWrong.append(name)
    print('Wrong: %d' %len(gWrong))
    for w in gWrong:
        print(w)
    print('### Wrong/Total: %d / %d ###'%(len(gWrong), total_num))
