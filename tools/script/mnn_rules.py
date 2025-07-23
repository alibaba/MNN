#!/usr/bin/env python3
#
# This script analyses Cppcheck dump files to global static issues
# - warn about static global objects
#

import sys
# 尝试导入 cppcheckdata 模块
try:
    # append cppcheck addones dir
    sys.path.append('/usr/lib/x86_64-linux-gnu/cppcheck/addons')
    import cppcheckdata
except ImportError as e:
    print('Warning: cppcheckdata module is not available:', e, file=sys.stderr)
    print('Skipping static check.')
    sys.exit(0)

def reportError(token, msg, id):
    cppcheckdata.reportError(token, 'warning', msg, 'mnn-rules', id)

def mnn_rules(vari, arg):
    if 'opencl' in arg:
        return
    # 1. global static variable: Class, not Pointer, Static, Global
    if var.isClass and not var.isPointer and var.isStatic and (var.isGlobal or var.access == 'Namespace'):
        reportError(var.typeStartToken, 'Global static variable \'' + var.nameToken.str + '\', dangerous on iOS', 'global-static')

if __name__ == '__main__':
    for arg in sys.argv[1:]:
        if arg.startswith('-'):
            continue
        print('Checking ' + arg + '...')
        data = cppcheckdata.parsedump(arg)
        for cfg in data.configurations:
            if len(data.configurations) > 1:
                print('Checking ' + arg + ', config "' + cfg.name + '"...')
            for var in cfg.variables:
                mnn_rules(var, arg)
