#!/usr/bin/python
import os
import sys

def run(path):
    cmd = "find " + path + " -name \"*.pb\" | xargs -I {} ./MNNConvert -f TF --modelFile {} --MNNModel temp.mnn --bizCode test"
    # print(cmd)
    print(os.popen(cmd).read())
    return 0

if __name__ == "__main__":
    path = sys.argv[1]
    run(path)