#!/usr/bin/python
import sys

model_root_dir = sys.argv[1]

import os
gWrong = []

root_dir = os.path.join(model_root_dir, 'TestOnnx')
print('root: ' + root_dir + '\n')

for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    print(name)

    message = os.popen("./TestConvertResult Onnx " + root_dir + '/' + name).read()
    if (message.find('TEST_SUCCESS') == -1):
        gWrong.append(name)
    print(message)

print('Wrong: %d' %len(gWrong))
for w in gWrong:
    print(w)
