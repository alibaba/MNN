#!/usr/bin/python
import os

def generateGradFile(rootDir):
    geoDir = os.path.join(rootDir, "source", "grad")
    regFile = os.path.join(geoDir, "GradOPRegister.cpp")
    fileNames = os.listdir(geoDir)
    print(fileNames)
    if len(fileNames) <= 1:
        # Error dirs
        return
    funcNames = []
    for fi in fileNames:
        if ".cpp" not in fi:
            continue
        f = os.path.join(geoDir, fi)
        if os.path.isdir(f):
            continue
        with open(f) as fileC:
            c = fileC.read().split('\n')
            c = list(filter(lambda l:l.find('REGISTER_GRAD')>=0, c))
            for l in c:
                l = l.split('(')[1]
                l = l.split(')')[0]
                l = l.replace(' ', '')
                l = l.split(',')
                funcName = '___' + l[0] + '__' + l[1] + '__'
                funcNames.append(funcName)

    with open(regFile, 'w') as f:
        f.write('// This file is generated by Shell for ops register\n')
        f.write('#include \"OpGrad.hpp\"\n')
        f.write('namespace MNN {\n')
        for l in funcNames:
            f.write("extern void " + l + '();\n')
        f.write('\n')
        f.write('void registerGradOps() {\n')
        for l in funcNames:
            f.write(l+'();\n')
        f.write("}\n}\n")


import sys
generateGradFile(sys.argv[1])