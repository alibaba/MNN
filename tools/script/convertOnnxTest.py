#!/usr/bin/python
#-- coding:utf8 --
import sys

config_file = ""
model_root_dir = sys.argv[1]
if len(sys.argv) > 2:
    config_file = sys.argv[2]
total_num = 0
total_size = 0.0
correct_num = 0
import os
def run_cmd(args):
    cmd = args[0]
    for i in range(1, len(args)):
        cmd += ' ' + args[i]
    stdout = os.popen(cmd).read()
    global total_num
    total_num += 1
    return stdout

gWrong = []

root_dir = os.path.join(model_root_dir, 'TestOnnx')
print('root: ' + root_dir + '\n')

for name in os.listdir(root_dir):
    if name == '.DS_Store' or name == 'ops':
        continue
    print(name)
    message = run_cmd(['./TestConvertResult', 'Onnx', root_dir + '/' + name, config_file])
    if (message.find('TEST_SUCCESS') == -1):
        gWrong.append(name)
    else:
        mnnsize = (os.path.getsize("convert_cache.mnn") + os.path.getsize("convert_cache.mnn.weight")) / 1024.0 / 1024.0
        correct_num +=1
        total_size += mnnsize
    print(message)
    print("Model Size:", mnnsize, " MB")

print('Wrong: %d' %len(gWrong))
for w in gWrong:
    print(w)
print('TEST_NAME_MODULE: 模型测试\nTEST_CASE_AMOUNT_MODULE: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n'%(len(gWrong), total_num - len(gWrong)))
print('TEST_CASE={\"name\":\"Onnx转换测试\",\"failed\":%d,\"passed\":%d}\n'%(len(gWrong), total_num - len(gWrong)))
print('Total Size: ', total_size,' MB, convert ', correct_num," model")
if len(gWrong) > 0:
    exit(1)
