#!/usr/bin/python
import os
import sys

def run_cmd(args):
    from subprocess import Popen, PIPE, STDOUT
    stdout, _ = Popen(args, stdout=PIPE, stderr=STDOUT).communicate()
    return stdout.decode('utf-8')

gWrong = []
gRight = 0

def test(model_root_dir, forwardType, thredhold):
    global gRight
    root_dir = model_root_dir + '/TestResource'
    print('root: ', root_dir)
    for name in os.listdir(root_dir):
        if name == '.DS_Store':
            continue
        modelName = os.path.join(root_dir, name, 'temp.bin')
        inputName = os.path.join(root_dir, name, 'input_0.txt')
        outputName = os.path.join(root_dir, name, 'output.txt')
        print(modelName)
        run_cmd(['adb', 'push', modelName, '/data/local/tmp/MNN/temp.bin'])
        run_cmd(['adb', 'push', inputName, '/data/local/tmp/MNN/input_0.txt'])
        run_cmd(['adb', 'push', outputName, '/data/local/tmp/MNN/output.txt'])
        message = run_cmd(['adb', 'shell', 'cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && ./testModel.out temp.bin input_0.txt output.txt %s %s' %(forwardType, thredhold)])
        print(str(message))
        if (message.find('Correct') < 0):
            gWrong.append(modelName)
        else:
            gRight += 1
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
        else:
            gRight += 1
    # total model test
    root_dir = os.path.join(model_root_dir, 'TestWithDescribe')
    print('Model Root Path: ' + root_dir + '\n')
    for name in os.listdir(root_dir):
        if name == '.DS_Store':
            continue
        modelDir = os.path.join(root_dir, name)
        modelName = os.path.join(modelDir, 'temp.bin')
        if not os.path.exists(modelName):
            continue
        print(run_cmd(['adb', 'push', modelDir, '/data/local/tmp/MNN/']))
        print(modelDir)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Running...")
        message = run_cmd(['adb', 'shell', 'cd /data/local/tmp/MNN&&export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && ./testModelWithDescrisbe.out %s/temp.bin %s/config.txt %s %s' %(name, name, forwardType, thredhold)])
        run_cmd(['adb', 'shell', 'rm -rf /data/local/tmp/MNN/%s'%(name)])
        if (message.find('Correct') == -1):
            gWrong.append(modelDir)
        else:
            gRight += 1
        print(message)
    print('Wrong: ', len(gWrong))
    for w in gWrong:
        print(w)

def android_test(root_dir, forwardType, thredhold, bits):
    test(root_dir, forwardType, thredhold)
    print("TEST_NAME_MODEL%d: 模型测试%d\nTEST_CASE_AMOUNT_MODEL%d: {\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n"%(bits, bits, bits, len(gWrong), gRight))

if __name__ == '__main__':
    root_dir = sys.argv[1]
    forwardType = " 0 "
    if len(sys.argv) > 2:
        forwardType = ' ' + sys.argv[2] + ' '
    thredhold = ' 0.001 '
    if len(sys.argv) > 3:
        thredhold = ' ' + sys.argv[3] + ' '
    if len(sys.argv) > 4:
        thredhold += (' ' + sys.argv[4] + ' ')
    test(root_dir, forwardType, thredhold)
