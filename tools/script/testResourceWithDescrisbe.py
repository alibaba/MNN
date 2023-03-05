#!/usr/bin/python
import sys

root_dir = sys.argv[1]
print('Model Root Path: ' + root_dir + '\n')

import os
def run_cmd(args):
    from subprocess import Popen, PIPE, STDOUT
    stdout, _ = Popen(args, stdout=PIPE, stderr=STDOUT).communicate()
    return stdout

for name in os.listdir(root_dir):
    modelName = os.path.join(root_dir, name, 'temp.bin')
    if not os.path.exists(modelName):
        continue
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Running...")
    print(modelName)
    config = os.path.join(root_dir, name, 'config.txt')
    print(run_cmd(['./testModelWithDescribe.out', modelName, config, '0', '0.0001']))
