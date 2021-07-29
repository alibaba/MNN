import cases

import MNN
F = MNN.expr

import os
from os.path import join, exists

try:
    import MNNCV
except:
# invoke under work_path (import cases directly) on official python vm on PC
# > cd pymnn/test/playground && python main.py
    work_path = os.getcwd()
else:
# invoke on alinnpython vm on mobile by MNN WorkBench
    kit = MNNCV.Kit()
    work_path = kit.getEnvVars()['workPath']

models_dir = join(work_path, 'models')

models_path = []
with open(join(work_path, 'model_names.txt')) as f:
    for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        model_path = join(models_dir, line)
        models_path.append(model_path)

F.set_config(backend=F.Backend.CPU)
#F.set_config(backend=F.Backend.HIAI)
#F.set_config(backend=F.Backend.OPENCL)

#cases.dynamic_module_test(models_path)
cases.static_module_test(models_path)
