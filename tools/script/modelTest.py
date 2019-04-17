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

import os
gWrong = []
for name in os.listdir(root_dir):
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    outputName = os.path.join(root_dir, name, 'output.txt')
    print modelName

    message = os.popen('./testModel.out ' + modelName + ' ' + inputName + ' ' + outputName + forwardType + thredhold).read()
    if (message.find('Correct') == -1):
        gWrong.append(modelName)
    print message

root_dir = model_root_dir + '/TestWithDescribe'
print('Model Root Path: ' + root_dir + '\n')

for name in os.listdir(root_dir):
    modelName = os.path.join(root_dir, name, 'temp.bin')
    if not os.path.exists(modelName):
        continue
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Running...")
    print(modelName)
    config = os.path.join(root_dir, name, 'config.txt')
    message = os.popen('./testModelWithDescrisbe.out ' + modelName + ' ' + config + ' ' + '0' + ' 0.0001').read()
    if (message.find('Correct') == -1):
        gWrong.append(modelName)
    print message
print 'Wrong: ', len(gWrong)
for w in gWrong:
    print w
