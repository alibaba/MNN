#!/usr/bin/python3
#-- coding:utf8 --
import sys
import os
import re
import json

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

def compare(origin, quant, jsonFile):
    img_dir = '../resource/images'
    for name in os.listdir(img_dir):
        origin_res = run_cmd(['./pictureRecognition_module.out', origin, jsonFile, img_dir + '/' + name])
        quant_res = run_cmd(['./pictureRecognition_module.out', quant, jsonFile, img_dir + '/' + name])

        originIdx, originPoint = parseRes(origin_res)
        quantIdx, quantPoint = parseRes(quant_res)
        print(originIdx, originPoint)
        print(quantIdx, quantPoint)
        idxRate = len(originIdx & quantIdx) / max(len(originIdx), len(quantIdx))    
        pointRate = quantPoint / originPoint
        print(name, idxRate, pointRate)
        if idxRate < 0.5:
            print('False')
            return False
    return True

def compareAcc(origin, quant, imagepath, labelpath, quantized_json):
    # img_dir = '../resource/batchimgs'
    img_dir = imagepath
    groundtruth = labelpath

    batchsize, totalimgs = "5", "50"

    origin_res = run_cmd(['./pictureRecognition_batch.out', origin, img_dir + '/', groundtruth, quantized_json, batchsize, totalimgs])
    quant_res = run_cmd(['./pictureRecognition_batch.out', quant, img_dir + '/', groundtruth, quantized_json, batchsize, totalimgs])

    pattern = re.compile(r"\d+\.?\d*")
    acc_origin = float(pattern.findall(origin_res.split("acc: ")[1])[0])
    acc_quant = float(pattern.findall(quant_res.split("acc: ")[1])[0])
    print("Original model accuracy: ", acc_origin)
    print("Quantized model accuracy: ", acc_quant)

    if acc_origin - acc_quant > 10.0:
        print("Accuracy lose too much...")
        return False
    return True

def test(modelpath, path):
    global total_num
    total_num += 1
    jsonFile = path + '/test.json'
    jsonObj = {}
    with open(jsonFile) as f:
        jsonObj = json.loads(f.read())
    originModel = modelpath + jsonObj['model']
    quantModel  = './__quantModel.mnn'
    message = run_cmd(['./quantized.out', originModel, quantModel, path + '/test.json'])
    res = True
    try:
        res = compare(originModel, quantModel, jsonFile)
    except:
        print('Quant Error!')
        res = False

    message = run_cmd(['rm -f ' + quantModel])
    return res

def testacc(modelpath, imagepath, path, labelpath):
    res = True
    jsonFile = path + '/quantized.json'
    jsonObj = {}
    with open(jsonFile) as f:
        jsonObj = json.loads(f.read())
    originModel = modelpath + jsonObj['model']
    quantModel  = './__quantModel.mnn'
    message = run_cmd(['./quantized.out', originModel, quantModel, jsonFile])
    res = True
    try:
        res = compareAcc(originModel, quantModel, imagepath, labelpath, jsonFile)
    except:
        print('Quant Error!')
        res = False

    message = run_cmd(['rm -f ' + quantModel])
    return res

if __name__ == '__main__':
    model_root_dir = sys.argv[1]
    root_dir = os.path.join(model_root_dir, 'TestPTQ')
    print('root: ' + root_dir + '\n')
    
    gWrong = []
    for name in os.listdir(root_dir + '/json'):
        if '.DS_Store' in name:
            continue
        print(name)
        res = test(root_dir + '/model/', root_dir + '/json/' + name)
        if not res:
            gWrong.append(name)
    print('Single picture test wrong: %d' %len(gWrong))
    for w in gWrong:
        print(w)
    print('TEST_NAME_PTQ: PTQ测试\nTEST_CASE_AMOUNT_PTQ: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n'%(len(gWrong), total_num - len(gWrong)))
    if len(gWrong) > 0:
        exit(1)

    gWrong = []
    print("Batch pictures test...")
    for name in os.listdir(root_dir + '/json'):
        if '.DS_Store' in name:
            continue
        print(name)
        res = testacc(root_dir + '/model/', root_dir + '/batchimgs', root_dir + '/json/' + name, root_dir + '/trueval.txt')
        if not res:
            gWrong.append(name)
    print('Batch pictures test wrong: %d' %len(gWrong))
    for w in gWrong:
        print(w)
    print('BATCH_TEST_NAME_PTQ: PTQ测试\nTEST_CASE_AMOUNT_PTQ: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n'%(len(gWrong), total_num - len(gWrong)))
    if len(gWrong) > 0:
        exit(1)
