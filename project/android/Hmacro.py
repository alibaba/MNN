#!/usr/bin/python
import os

def getMacro(fileName):
    #Delete '/' '.'
        sta =0
        while sta <len(fileName):
            if (fileName[sta]=='/'):
                break
            sta+=1
        fin = sta+1
        while fin < len(fileName):
            if (fileName[fin]=='.'):
                break
            fin+=1
        fileName = fileName[sta+1 : fin]

        dirs = fileName.split('/')
        macro = ''
        for d in dirs:
            macro += d.upper() + '_'
        macro+= 'H'
        return macro

def replaceMacro(fileName, macro):
    f = open(fileName)
    lines = f.read().split('\n')
    f.close()
    #Find if the file has macro
    hasmacro = False
    for line in lines:
        if (line.find('_H')!=-1 or line.find('_h')!=-1) and line.find('#ifndef')!=-1:
            hasmacro = True
            break
        if (line.find('#pragma')!=-1):
            hasmacro = True
            break
    if hasmacro == False:
        print(fileName)
        fc = ''
        fc += '#ifndef ' + macro + '\n'
        fc += '#define ' + macro + '\n'
        for line in lines:
            if len(line) <=0:
                continue
            fc += line + '\n'
        fc += '#endif'+'\n'
        f = open(fileName, 'w')
        f.write(fc);
        f.close()
    return

if __name__=='__main__':
    cmd = 'find . -name \"*.h\"'
    files = os.popen(cmd).read().split('\n')
    for file in files:
        if len(file) <=0:
            continue
        macro = getMacro(file);
        replaceMacro(file, macro);

