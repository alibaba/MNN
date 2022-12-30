#!/usr/bin/python
#-- coding:utf8 --
import sys

model_root_dir = sys.argv[1]
total_num = 0

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

root_dir = os.path.join(model_root_dir, 'TestTf')
print('root: ' + root_dir + '\n')

for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    print(name)
    message = run_cmd(['./TestConvertResult', 'Tf', root_dir + '/' + name])
    if (message.find('TEST_SUCCESS') == -1):
        gWrong.append(name)
    print(message)

print('Wrong: %d' %len(gWrong))
for w in gWrong:
    print(w)
print('TEST_NAME_TF: TFConvert测试\nTEST_CASE_AMOUNT_TF: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n'%(len(gWrong), total_num - len(gWrong)))
if len(gWrong) > 0:
    exit(1)
