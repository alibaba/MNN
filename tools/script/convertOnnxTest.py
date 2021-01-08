#!/usr/bin/python
import sys

model_root_dir = sys.argv[1]
total_num = 0
import os
def run_cmd(args):
    from subprocess import Popen, PIPE, STDOUT
    stdout, _ = Popen(args, stdout=PIPE, stderr=STDOUT).communicate()
    global total_num
    total_num += 1
    return stdout

gWrong = []

root_dir = os.path.join(model_root_dir, 'TestOnnx')
print('root: ' + root_dir + '\n')

for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    print(name)
    message = run_cmd(['./TestConvertResult', 'Onnx', root_dir + '/' + name])
    if (message.find('TEST_SUCCESS') == -1):
        gWrong.append(name)
    print(message)

print('Wrong: %d' %len(gWrong))
for w in gWrong:
    print(w)
print '### Wrong/Total: %d / %d ###'%(len(gWrong), total_num)
