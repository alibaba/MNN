#!/usr/bin/python
import sys

root_dir = sys.argv[1]
print 'root: ', root_dir

forwardType = " 0 "
if len(sys.argv) > 2:
    forwardType = ' ' + sys.argv[2] + ' '
thredhold = ' 0.001 '
if len(sys.argv) > 3:
    thredhold = ' ' + sys.argv[3] + ' '

import os
for name in os.listdir(root_dir):
    if name.startswith("."):
        continue
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    outputName = os.path.join(root_dir, name, 'output.txt')
    print modelName

    print os.popen('./testModel.out ' + modelName + ' ' + inputName + ' ' + outputName + forwardType + thredhold).read()
