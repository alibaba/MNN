#-*-coding:utf-8-*-
#coding by: yuangu(lifulinghan@aol.com)

import os
import sys
import shutil
import platform

def p():
    frozen = "not"
    if getattr(sys, 'frozen',False):
        frozen = "ever so"
        return os.path.dirname(sys.executable)

    return os.path.split(os.path.realpath(__file__))[0]

currentWorkPath = p()
os.chdir(currentWorkPath)

if '-lazy' in sys.argv and os.path.isdir("current"):
    print("*** done ***")
    exit(0)

# check is flatbuffer installed or not
FLATC = '../3rd_party/flatbuffers/tmp/flatc' + ('.exe' if "Windows" ==  platform.system() else '')
FLATC = os.path.realpath(FLATC)

if not os.path.isfile(FLATC):
    print("*** building flatc ***")
    tmpDir = os.path.realpath('../3rd_party/flatbuffers/tmp')
    
    if os.path.isdir(tmpDir):
        shutil.rmtree(tmpDir)
    
    os.mkdir(tmpDir)
    os.chdir(tmpDir)

    os.system('cmake  -DCMAKE_BUILD_TYPE=Release ..')
    if "Windows" ==  platform.system():
        os.system('cmake --build . --target flatc --config Release')
        if os.path.isfile( os.path.join(tmpDir, 'Release/flatc.exe') ):
            shutil.move(os.path.join(tmpDir, 'Release/flatc.exe'), FLATC)
    else:
        os.system('cmake --build . --target flatc')


    # dir recover
    os.chdir(currentWorkPath)

# determine directory to use
DIR='default'
if os.path.isdir('private'):
    DIR = 'private'
DIR = os.path.realpath(DIR)

# clean up
print('*** cleaning up ***')
if os.path.isdir('current'):
    shutil.rmtree('current')
os.mkdir('current')

# flatc all fbs
os.chdir('current')
listFile = os.listdir(DIR)
for fileName in listFile:
    tmpFileName = os.path.join(DIR, fileName)
    cmd = "%s -c -b --gen-object-api --reflect-names %s" %(FLATC, tmpFileName)
    os.system(cmd)

os.chdir(currentWorkPath)
print( "*** done ***")