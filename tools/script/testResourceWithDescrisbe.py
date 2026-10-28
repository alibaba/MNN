#!/usr/bin/python
import sys

root_dir = sys.argv[1]
print('Model Root Path: ' + root_dir + '\n')

import os
def run_cmd(args):
    from subprocess import Popen, PIPE, STDOUT
    process = Popen(args, stdout=PIPE, stderr=STDOUT)
    stdout, _ = process.communicate()
    return stdout, process.returncode

failed = False
for name in os.listdir(root_dir):
    modelName = os.path.join(root_dir, name, 'temp.bin')
    if not os.path.exists(modelName):
        continue
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Running...")
    print(modelName)
    config = os.path.join(root_dir, name, 'config.txt')
    try:
        output, returncode = run_cmd(['./testModelWithDescribe.out', modelName, config, '0', '0.0001'])
    except OSError as error:
        print('Failed to run %s: %s' % (modelName, error))
        failed = True
        continue
    print(output)
    if returncode != 0:
        print('Failed model %s: exit status %d' % (modelName, returncode))
        failed = True

sys.exit(1 if failed else 0)
