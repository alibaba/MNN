#!/usr/bin/python
import sys

model_root_dir = sys.argv[1]

root_dir = model_root_dir + '/TestResource'
print('root: ', root_dir)

forwardType = " 0 "
if len(sys.argv) > 2:
    forwardType = ' ' + sys.argv[2] + ' '
thredhold = ' 0.001 '
if len(sys.argv) > 3:
    thredhold = ' ' + sys.argv[3] + ' '
if len(sys.argv) > 4:
    thredhold += (' ' + sys.argv[4] + ' ')

gWrong = []
import os
def run_cmd(args):
    from subprocess import Popen, PIPE, STDOUT
    stdout, _ = Popen(args, stdout=PIPE, stderr=STDOUT).communicate()
    return stdout

for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    outputName = os.path.join(root_dir, name, 'output.txt')
    print modelName
    run_cmd(['adb', 'push', modelName, '/data/local/tmp/MNN/temp.bin'])
    run_cmd(['adb', 'push', inputName, '/data/local/tmp/MNN/input_0.txt'])
    run_cmd(['adb', 'push', outputName, '/data/local/tmp/MNN/output.txt'])

    message = run_cmd(['adb', 'shell', 'cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && ./testModel.out temp.bin input_0.txt output.txt %s %s' %(forwardType, thredhold)])
    print(message)
    if (message.find('Correct') < 0):
        gWrong.append(modelName)

root_dir = model_root_dir + '/OpTestResource'
print('root: ', root_dir)

for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    outputName = os.path.join(root_dir, name, 'output_0.txt')
    print(modelName)
    run_cmd(['adb', 'push', modelName, '/data/local/tmp/MNN/temp.bin'])
    run_cmd(['adb', 'push', inputName, '/data/local/tmp/MNN/input_0.txt'])
    run_cmd(['adb', 'push', outputName, '/data/local/tmp/MNN/output_0.txt'])

    message = run_cmd(['adb', 'shell', 'cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && ./testModel.out temp.bin input_0.txt output_0.txt %s %s' % (forwardType, thredhold)])
    print(message)
    if (message.find('Correct') == -1):
        gWrong.append(modelName)

# total model test
root_dir = os.path.join(model_root_dir, 'TestWithDescribe')
print('Model Root Path: ' + root_dir + '\n')

for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    modelName = os.path.join(root_dir, name, 'temp.bin')
    if not os.path.exists(modelName):
        continue
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Running...")
    print(modelName)
    print(os.popen('adb push ' + root_dir + '/' + name + '/* ' + '/data/local/tmp/MNN/').read())
    message = run_cmd(['adb', 'shell', 'cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && ./testModelWithDescrisbe.out temp.bin config.txt %s %s' %(forwardType, thredhold)])
    if (message.find('Correct') == -1):
        gWrong.append(modelName)
    print(message)

print('Wrong: ', len(gWrong))
for w in gWrong:
    print(w)
