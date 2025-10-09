#!/usr/bin/python
import sys
import json
import os
post_treat = {}
# neuropilot position
# eg: third/mtk/neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk/
cmd = sys.argv[1]
ncccomiple = cmd + '/host/bin/ncc-tflite '
extractshared = cmd + '/host/bin/extract-shared '
with open(sys.argv[2]) as f:
    post_treat = json.load(f)
merges = post_treat["merge"]
for key in post_treat["merge"]:
    srcs = merges[key]
    dst = key
    #options = " --opt=3 --opt-footprint --opt-aggressive --stable-linearize --gno=LTS --gno-exp --gno-non-4d-tiling --fc-to-conv --broadcast-act-wgt --split-large-conv-ic=1536 --mdla-int16-lut --suppress-input --suppress-output --show-memory-summary"
    options = " --stable-linearize --gno=LTS --gno-exp --gno-non-4d-tiling --fc-to-conv --broadcast-act-wgt "
    options += " --mdla-int16-lut "
    options += " --split-large-conv-ic=2048"
    options += ' --opt=3 --opt-accuracy --opt-footprint '
    # options += " --suppress-input --suppress-output "
    archoptions = ' --l1-size-kb=7168 --num-mdla=4'
    IN_DLA = ""
    OUT_DLA = ""
    for i,src in enumerate(srcs):
        fullcmd = ncccomiple + " " + src + " --arch=mdla5.1 " + options + archoptions
        print(fullcmd)
        print(os.popen(fullcmd).read())
        IN_DLA += " -i " + src.replace('tflite', 'dla') + ' '
        OUT_DLA += " -o " + key + ".shared" + "_%d" %i
    if len(srcs) > 1:
        fullcmd = extractshared + " " + IN_DLA + ' ' + OUT_DLA + " -s " + key + '.weight'
        print(fullcmd)
        print(os.popen(fullcmd).read())
    
