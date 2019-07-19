#!/usr/bin/python
import sys

model_root_dir = sys.argv[1]

root_dir = model_root_dir + '/TestResource'
print 'root: ', root_dir

forwardType = " 0 "
if len(sys.argv) > 2:
    forwardType = ' ' + sys.argv[2] + ' '
thredhold = ' 0.001 '
if len(sys.argv) > 3:
    thredhold = ' ' + sys.argv[3] + ' '

gWrong = []
import os
for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    outputName = os.path.join(root_dir, name, 'output.txt')
    print modelName
    os.popen("adb push " + modelName + " /data/local/tmp/MNN/temp.bin ").read()
    os.popen("adb push " + inputName + " /data/local/tmp/MNN/input_0.txt ").read()
    os.popen("adb push " + outputName + " /data/local/tmp/MNN/output.txt ").read()

    message = os.popen('adb shell \"cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && ./testModel.out temp.bin input_0.txt output.txt ' + forwardType + thredhold + "\"").read()
    print message
    if (message.find('Correct') < 0):
        gWrong.append(modelName)

root_dir = model_root_dir + '/OpTestResource'
print 'root: ', root_dir

forwardType = " 0 "
if len(sys.argv) > 2:
    forwardType = ' ' + sys.argv[2] + ' '
thredhold = ' 0.001 '
if len(sys.argv) > 3:
    thredhold = ' ' + sys.argv[3] + ' '

import os
for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    outputName = os.path.join(root_dir, name, 'output_0.txt')
    print modelName
    os.popen("adb push " + modelName + " /data/local/tmp/MNN/temp.bin ").read()
    os.popen("adb push " + inputName + " /data/local/tmp/MNN/input_0.txt ").read()
    os.popen("adb push " + outputName + " /data/local/tmp/MNN/output_0.txt ").read()
    
    message = os.popen('adb shell \"cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && ./testModel.out temp.bin input_0.txt output_0.txt ' + forwardType + thredhold + "\"").read()
    print message
    if (message.find('Correct') == -1):
        gWrong.append(modelName)

print 'Wrong: ', len(gWrong)
for w in gWrong:
    print w
