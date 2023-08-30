#!/usr/bin/python
import sys
import os
from subprocess import Popen, PIPE, STDOUT

root_dir = sys.argv[1]
print('root: ', root_dir)

forwardType = " 0 "
if len(sys.argv) > 2:
    forwardType = ' ' + sys.argv[2] + ' '
    
thredhold = ' 0.001 '
if len(sys.argv) > 3:
    thredhold = ' ' + sys.argv[3] + ' '

def run_cmd(args):
    process = Popen(args, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
    return stdout, stderr

for name in os.listdir(root_dir):
    if name.startswith("."):
        continue
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    outputName = os.path.join(root_dir, name, 'output.txt')
    print(modelName)

    stdout, stderr = run_cmd(['./testModel.out', modelName, inputName, outputName, forwardType, thredhold])
    print(stdout)
    if stderr:
        print("Error:", stderr)
