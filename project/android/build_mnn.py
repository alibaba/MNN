#-*-coding:utf-8-*
# code by yuangu(lifulinghan@aol.com)

import os
import platform
import shutil
import sys
import getopt

def p():
    frozen = "not"
    if getattr(sys, 'frozen',False):
        frozen = "ever so"
        return os.path.dirname(sys.executable)

    return os.path.split(os.path.realpath(__file__))[0]

def checkPath(path):
    if not os.path.isdir(path):      
        parent = os.path.dirname(path)
        if os.path.isdir(parent):
            os.mkdir(path)
        else:
            checkPath(parent)

SUPPER_ABI_LIST = [
    'armeabi-v7a',
    "arm64-v8a",
    "x86",
    'x86_64'
] 

class MNN_Builder:
    def __init__(self, argv):
        try:
            opts, args = getopt.getopt(argv,"hs:o:a:",['help', 'sdk=','out=','abi='])
        except getopt.GetoptError:
            self.usage()
            sys.exit(-1)

        androidSDKPath = None
        outDir = os.path.join(p(), 'out')
        abiList = [abi for abi in SUPPER_ABI_LIST]
        extOption = []

        for opt, arg in opts:
            if opt in ("-h","--help"):
                self.usage()
                sys.exit()
           
            elif opt in ("-s","--sdk"):
                androidSDKPath = arg
            
            elif opt in ("-o","--out"):
                outDir =  arg
            elif opt in ('-a', '--abi'):
                if arg == 'all':
                    pass
                elif arg in SUPPER_ABI_LIST:
                    abiList.append(arg)
                else:
                    print('abi not support')
                    self.usage()
                    sys.exit(-1)
            elif opt in ('-c', '--core'):
                if arg == 'opencl':
                    extOption.append("-DMNN_OPENCL=true")
                    extOption.append("-DANDROID_PLATFORM=android-16")
                elif arg == 'opengl':
                    extOption.append("-DMNN_OPENGL=true")
                    extOption.append("-DANDROID_PLATFORM=android-21")
                elif arg == 'vulkan':
                    extOption.append("-DMNN_VULKAN=true")
                    extOption.append("-DANDROID_PLATFORM=android-21")
                elif arg == 'cpu':
                    pass
                else:
                    print('the core not support')
                    self.usage()
                    sys.exit(-1)
            elif opt in ('-d', '--enable_debug'):
                extOption.append('-DMNN_DEBUG=true')
        
        if not '-DMNN_DEBUG=true' in extOption:
            extOption.append('-DMNN_DEBUG=false')


        if androidSDKPath == None:
            androidSDKPath = self.getAndroidSDKPath()
        
        #check android sdk
        if androidSDKPath == None or not os.path.isdir(androidSDKPath):
            print("not found android sdk")
            sys.exit(-1)

        androidNDKPath = self.getNDKPath(androidSDKPath)
        # check android ndk
        if androidNDKPath == None:
            print('not found android ndk')
            sys.exit(-1)
        
        cmakeDir = self. getCmakeDir(androidSDKPath)
        if cmakeDir == None:
            print("please install cmake in android sdk")
            exit(-1)

        outDir = os.path.realpath(outDir)
        checkPath(outDir)

        cmakeBin = os.path.join(cmakeDir,'bin/cmake')   + ( '.exe' if  'Windows' ==  platform.system() else '' )
        ninjaBin = os.path.join(cmakeDir,'bin/ninja')  + ( '.exe' if  'Windows' ==  platform.system() else '' )
        print(abiList)
        for abi in  abiList:
            build_path = self.build(abi, androidNDKPath, cmakeBin, ninjaBin, extOption)
            self.copySoToOut(build_path, abi, outDir)
        
        print('****done****')
        
    def usage(self):
        print('usage: python build_mnn.py [-s <Android SDK>] [-o <*.so out dir>] [-a <aib name>]')
        print("-h, --help  print this message")
        print("-s, -sdk  Android SDK dir path, default from system variables of ANDROID_HOME or ANDROID_SDK_ROOT ")
        print("-o, --out  *.so out dir default './out' ")
        print("-a, --abi  all,armeabi-v7a,arm64-v8a,x86,x86_64, default all")
        print("-c, --core cpu, opencl,opengl,vulkan, default cpu")
        print("-d, --enable_debug, default close")

    def getAndroidSDKPath(self):
        environ_names = [
            'ANDROID_HOME', 
            'ANDROID_SDK_ROOT'
        ]

        for name in environ_names:            
            #环境变量里不存在
            if name not  in os.environ.keys():
                continue

            android_sdk_path = os.environ[name]
            #验证如果不存在此目录 
            if not  os.path.isdir(android_sdk_path):
                continue
            
            return android_sdk_path
        
        #没有找到相应的sdk路径
        return None
    
    def getCmakeDir(self, androidSDKPath):
        ndk_cmake_dir  = os.path.join(androidSDKPath,  "cmake")
        if  not  os.path.isdir(ndk_cmake_dir):
            return None
        
        cmake_dir_list = os.listdir(ndk_cmake_dir)
        list_len = len(cmake_dir_list)
        if list_len <= 0:
            return  None
        
        return os.path.join(ndk_cmake_dir, cmake_dir_list[0] )


    # get ndk path from android sdk or NDK_ROOT or ANDROID_NDK
    def  getNDKPath(self, androidSDKPath):
        #通过系统变量来寻找
        environ_names = [
            'NDK_ROOT', 
            'ANDROID_NDK'
        ]

        for name in environ_names:            
            #环境变量里不存在
            if name not  in os.environ.keys():
                continue

            android_ndk_path = os.environ[name]
            #验证如果不存在此目录 
            if not  os.path.isdir(android_ndk_path):
                continue
            
            return android_ndk_path
        
        ndk_bundle_dir  = os.path.join(androidSDKPath,  "ndk-bundle/toolchains")
        if os.path.isdir(ndk_bundle_dir):
            return  os.path.join(androidSDKPath, "ndk-bundle")
    
    def build(self, abi, androidNDKPath,cmakeBin ,ninjaBin, extOption):
        rootPath = p()
        build_path = os.path.join(rootPath, 'build/' + abi)
        checkPath(build_path)

        if os.path.isdir(build_path):
            shutil.rmtree(build_path)

        os.mkdir(build_path)
        os.chdir(build_path)
        
        cmd = '''%s -DANDROID_ABI=%s   \
                %s \
                -DCMAKE_BUILD_TYPE=Release   \
                -DANDROID_NDK=%s    \
                -DANDROID_STL=c++_static \
                -DCMAKE_CXX_FLAGS=-std=c++11 -frtti -fexceptions   \
                -DCMAKE_TOOLCHAIN_FILE=%s/build/cmake/android.toolchain.cmake    \
                -DCMAKE_MAKE_PROGRAM=%s -G "Ninja"    \
                -DMNN_BUILD_FOR_ANDROID_COMMAND=true   \
                -DMNN_DEBUG=false -DNATIVE_LIBRARY_OUTPUT=. ../../../../'''%(cmakeBin,abi, ' '.join(extOption), androidNDKPath,androidNDKPath,ninjaBin) 

        if (os.system(cmd) != 0 or  os.system("%s --build ."%(cmakeBin, )) != 0):
            print("build failed")
            sys.exit(-1)
        
        os.chdir(rootPath)
        return build_path

    def copySoToOut(self, build_path, abi, outDir):
        copyList = ['libMNN.so']
        for v in copyList:
            if os.path.exists(os.path.join( build_path, v )):
                checkPath(os.path.join(outDir, abi))
                shutil.copy( os.path.join( build_path, v ), os.path.join(outDir, abi, v ))



if __name__ == "__main__":
    MNN_Builder(sys.argv[1:])

    