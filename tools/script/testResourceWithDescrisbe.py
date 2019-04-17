#!/usr/bin/python
import sys

root_dir = sys.argv[1]
print('Model Root Path: ' + root_dir + '\n')

import os
for name in os.listdir(root_dir):
    modelName = os.path.join(root_dir, name, 'temp.bin')
    if not os.path.exists(modelName):
        continue
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Running...")
    print(modelName)
    config = os.path.join(root_dir, name, 'config.txt')
    print(os.popen('./testModelWithDescrisbe.out ' + modelName + ' ' + config + ' ' + '0' + ' 0.0001').read())
