#!/usr/bin/python
import sys
import os

model_root_dir = sys.argv[1]

forwardType = " 0 "
if len(sys.argv) > 2:
    forwardType = ' ' + sys.argv[2] + ' '
thredhold = ' 0.001 '
if len(sys.argv) > 3:
    thredhold = ' ' + sys.argv[3] + ' '
if len(sys.argv) > 4:
    thredhold += (' ' + sys.argv[4] + ' ')
runStatic = False
if len(sys.argv) > 5:
    runStatic = True
gWrong = []

convert = './MNNConvert -f MNN --bizCode MNN --saveStaticModel --modelFile '
tmpModel = '__tmpModel__.mnn'
dynamic_size = 0
static_size = 0
total_num = 0
# total model test
command = 'testModel.out.exe' if os.name == 'nt' else './testModel.out'
root_dir = os.path.join(model_root_dir, 'TestResource')
print('root: ' + root_dir + '\n')

# subprocess.Popen is intended to replace os.popen, which is more easy to release resource and safer.
# communicate function will close process automatically
def run_cmd(args):
    from subprocess import Popen, PIPE, STDOUT
    stdout, _ = Popen(args, stdout=PIPE, stderr=STDOUT).communicate()
    global total_num
    total_num += 1
    return stdout

for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    outputName = os.path.join(root_dir, name, 'output.txt')
    print modelName

    if runStatic:
        os.rename(modelName, tmpModel)
        message = os.popen(convert + tmpModel + ' ' + ' --MNNModel ' + modelName).read()
        if (message.find('Done') == -1):
            gWrong.append(modelName)
        print message
        dynamic_size += os.path.getsize(tmpModel)/1024.0
        static_size += os.path.getsize(modelName)/1024.0
    message = run_cmd([command, modelName, inputName, outputName, forwardType, thredhold])
    if (message.find('Correct') == -1):
        gWrong.append(modelName)
    print message
    if runStatic and os.path.exists(tmpModel):
        os.rename(tmpModel, modelName)

# model test for op
command = 'testModel.out.exe' if os.name == 'nt' else './testModel.out'
root_dir = os.path.join(model_root_dir, 'OpTestResource')
print('Model Root Path For OpTest: ' + root_dir + '\n')

for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    outputName = os.path.join(root_dir, name, 'output_0.txt')
    print modelName

    if runStatic:
        os.rename(modelName, tmpModel)
        message = os.popen(convert + tmpModel + ' ' + ' --MNNModel ' + modelName).read()
        if (message.find('Done') == -1):
            gWrong.append(modelName)
        print message
        dynamic_size += os.path.getsize(tmpModel)/1024.0
        static_size += os.path.getsize(modelName)/1024.0
    message = run_cmd([command, modelName, inputName, outputName, forwardType, thredhold])
    if (message.find('Correct') == -1):
        gWrong.append(modelName)
    print message
    if runStatic and os.path.exists(tmpModel):
        os.rename(tmpModel, modelName)

# total model test
command = 'testModelWithDescrisbe.out.exe' if os.name == 'nt' else './testModelWithDescrisbe.out'
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
    config = os.path.join(root_dir, name, 'config.txt')

    if runStatic:
        os.rename(modelName, tmpModel)
        message = os.popen(convert + tmpModel + ' ' + ' --MNNModel ' + modelName + ' --inputConfigFile ' + config).read()
        if (message.find('Done') == -1):
            gWrong.append(modelName)
        print message
        dynamic_size += os.path.getsize(tmpModel)/1024.0
        static_size += os.path.getsize(modelName)/1024.0
    message = run_cmd([command, modelName, config, forwardType, thredhold])
    if (message.find('Correct') == -1):
        gWrong.append(modelName)
    print message
    if runStatic and os.path.exists(tmpModel):
        os.rename(tmpModel, modelName)
print 'Wrong: ', len(gWrong)
for w in gWrong:
    print w
print '### Wrong/Total: %d / %d ###'%(len(gWrong), total_num)

if runStatic:
    print 'Total Dynamic Model Size: ', dynamic_size/1024.0, 'M'
    print 'Total Static Model Size: ', static_size/1024.0, 'M'
