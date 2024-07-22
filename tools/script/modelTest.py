#!/usr/bin/python
#-- coding:utf8 --
import sys
import os

model_root_dir = sys.argv[1]

forwardType = " 0 "
if len(sys.argv) > 2:
    forwardType = ' ' + sys.argv[2] + ' '
thredhold = ' 0.001 '
if len(sys.argv) > 3:
    thredhold = ' ' + sys.argv[3] + ' '
precision = ' 1 '
if len(sys.argv) > 4:
    precision = ' ' + sys.argv[4] + ' '

runStatic = False
if len(sys.argv) > 5:
    runStatic = True

input_dims = ''
if len(sys.argv) > 6:
    input_dims = ' ' + sys.argv[6] + ' '
parameters = forwardType + thredhold + precision + input_dims
print("all parameters: ", parameters)

gWrong = []

convert = ('MNNConvert.exe' if os.name == 'nt' else './MNNConvert') + ' -f MNN --bizCode MNN --saveStaticModel --modelFile '
tmpModel = '__tmpModel__.mnn'
dynamic_size = 0
static_size = 0
total_num = 0
# total model test
command = 'testModel.out.exe' if os.name == 'nt' else './testModel.out'
root_dir = os.path.join(model_root_dir, 'TestResource')
print('root: ' + root_dir + '\n')

message = ""
def run_cmd(args):
    cmd = args[0]
    for i in range(1, len(args)):
        cmd += ' ' + args[i]
    print(cmd)
    stdout = os.popen(cmd).read()
    global total_num
    total_num += 1
    return stdout

for name in os.listdir(root_dir):
    if name == '.DS_Store':
        continue
    modelName = os.path.join(root_dir, name, 'temp.bin')
    inputName = os.path.join(root_dir, name, 'input_0.txt')
    outputName = os.path.join(root_dir, name, 'output.txt')
    print(modelName)

    if runStatic:
        cmd = convert + modelName + ' ' + ' --MNNModel ' + tmpModel
        print(cmd)
        message = os.popen(cmd).read()
        if (message.find('Converted Success') == -1):
            gWrong.append(modelName)
            continue
        print(message)
        dynamic_size += os.path.getsize(modelName)/1024.0
        static_size += os.path.getsize(tmpModel)/1024.0
        if name == 'mobilenetv1quan' or name == 'overflowaware':
            parameters_specific = forwardType + ' 0.1 ' + precision + input_dims
            message = run_cmd([command, tmpModel, inputName, outputName, parameters_specific])
        else:
            message = run_cmd([command, tmpModel, inputName, outputName, parameters])
    else:
        if name == 'mobilenetv1quan' or name == 'overflowaware':
            parameters_specific = forwardType + ' 0.1 ' + precision + input_dims
            message = run_cmd([command, modelName, inputName, outputName, parameters_specific])
        else:
            message = run_cmd([command, modelName, inputName, outputName, parameters])
    if (message.find('Correct') == -1):
        gWrong.append(modelName)
    print(message)

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
    print(modelName)

    if runStatic:
        cmd = convert + modelName + ' ' + ' --MNNModel ' + tmpModel
        print(cmd)
        message = os.popen(cmd).read()
        if (message.find('Converted Success') == -1):
            gWrong.append(modelName)
            continue
        print(message)
        dynamic_size += os.path.getsize(modelName)/1024.0
        static_size += os.path.getsize(tmpModel)/1024.0
        message = run_cmd([command, tmpModel, inputName, outputName, parameters])
        print(message)
    else:
        message = run_cmd([command, modelName, inputName, outputName, parameters])
        print(message)
    if (message.find('Correct') == -1):
        gWrong.append(modelName)

# total model test
command = 'testModelWithDescribe.out.exe' if os.name == 'nt' else './testModelWithDescribe.out'
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
        cmd = convert + modelName + ' ' + ' --MNNModel ' + tmpModel + ' --inputConfigFile ' + config
        print(cmd)
        message = os.popen(cmd).read()
        if (message.find('Converted Success') == -1):
            gWrong.append(modelName)
            continue
        print(message)
        dynamic_size += os.path.getsize(modelName)/1024.0
        static_size += os.path.getsize(tmpModel)/1024.0
        message = run_cmd([command, tmpModel, config, parameters])
    else:
        message = run_cmd([command, modelName, config, parameters])
    if (message.find('Correct') == -1):
        gWrong.append(modelName)
    print(message)

# model test for train
command = 'testTrain.out.exe' if os.name == 'nt' else './testTrain.out'
root_dir = os.path.join(model_root_dir, 'TestTrain')
print('Model Root Path For Train: ' + root_dir + '\n')

for name in os.listdir(root_dir):
    print(name)
    if name == '.DS_Store':
        continue
    jsonName = os.path.join(root_dir, name, 'train.json')
    message = run_cmd([command, jsonName, os.path.join(root_dir, name)])
    if (message.find('Correct') == -1):
        gWrong.append("Train:" + name)
    print(message)

# model test for module
command = 'testModel_expr.out.exe' if os.name == 'nt' else './testModel_expr.out'
root_dir = os.path.join(model_root_dir, 'TestModule')
print('Model Root Path For Train: ' + root_dir + '\n')

for name in os.listdir(root_dir):
    print(name)
    if name == '.DS_Store':
        continue
    if name == 'saodubi':
        continue
    moduleName = os.path.join(root_dir, name, 'model.mnn')
    inputName = os.path.join(root_dir, name, 'input.mnn')
    outputName = os.path.join(root_dir, name, 'output.mnn')
    message = run_cmd([command, moduleName, inputName, outputName])
    if (message.find('Correct') == -1):
        gWrong.append("Module:" + name)
    print(message)

print('Wrong: ', len(gWrong))
for w in gWrong:
    print(w)

flag = ''
if runStatic:
    flag = 'STATIC'
print('TEST_NAME_MODEL%s: 模型测试%s\nTEST_CASE_AMOUNT_MODEL%s: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n'%(flag, flag, flag, len(gWrong), total_num - len(gWrong)))
if len(gWrong) > 0:
    exit(1)

if runStatic:
    print('Total Dynamic Model Size: ', dynamic_size/1024.0, 'M')
    print('Total Static Model Size: ', static_size/1024.0, 'M')
