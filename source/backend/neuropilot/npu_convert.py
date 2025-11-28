#!/usr/bin/python
import sys
import json
import os
post_treat = {}
# neuropilot position
sdk_path = os.environ["NEURON_SDK"]
print('NEURON_SDK: ', sdk_path)
ncccomiple = sdk_path + '/host/bin/ncc-tflite '
extractshared = sdk_path + '/host/bin/extract-shared '
archoptions = ' --arch=mdla5.1 --l1-size-kb=7168 --num-mdla=4 '
print(archoptions)
clean_tmp = True
with open(sys.argv[1]) as f:
    post_treat = json.load(f)
if (len(sys.argv)>=3):
    archoptions = sys.argv[2]
print("archoptions: ", archoptions)
merges = post_treat["merge"]
for key in post_treat["merge"]:
    srcs = merges[key]
    dst = key
    #options = " --opt=3 --opt-footprint --opt-aggressive --stable-linearize --gno=LTS --gno-exp --gno-non-4d-tiling --fc-to-conv --broadcast-act-wgt --split-large-conv-ic=1536 --mdla-int16-lut --suppress-input --suppress-output --show-memory-summary"
    options = " --stable-linearize --gno=LTS --gno-exp --gno-non-4d-tiling --fc-to-conv --broadcast-act-wgt "
    options += " --mdla-int16-lut "
    options += " --split-large-conv-ic=2048"
    options += ' --opt=3 --opt-accuracy --opt-footprint '
    options += " --suppress-input --suppress-output "
    IN_DLA = ""
    OUT_DLA = ""
    dlas = []
    for i,src in enumerate(srcs):
        fullcmd = ncccomiple + " " + src + options + archoptions
        print(fullcmd)
        print(os.popen(fullcmd).read())
        dlas.append(src.replace('tflite', 'dla'))
        IN_DLA += " -i " + src.replace('tflite', 'dla') + ' '
        OUT_DLA += " -o " + key + ".shared" + "_%d" %i
        if clean_tmp:
            os.remove(src)
    if len(srcs) > 1:
        fullcmd = extractshared + " " + IN_DLA + ' ' + OUT_DLA + " -s " + key + '.weight'
        print(fullcmd)
        print(os.popen(fullcmd).read())
        if clean_tmp:
            for dla in dlas:
                os.remove(dla)

    
