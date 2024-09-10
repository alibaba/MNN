import math
import numpy as np

max_diff = 0.
num_diff = 0
contents1 = open("test1.txt", "rt").read().split("\n")
contents2 = open("test2.txt", "rt").read().split("\n")
log0 = [content.split(' ') for content in contents1]
log1 = [content.split(' ') for content in contents2]
for layer, ln in enumerate(zip(log0, log1)):
    ln0, ln1 = ln
    for idx, n in enumerate(zip(ln0,ln1)):
        n0, n1 = n
        if n0 == "" or n1 == "":
            continue
        n = float(n0) - float(n1)
        if abs(n) >= 1e-9:
            max_diff = max(abs(round(n,10)), max_diff)
            num_diff += 1
            print(layer, idx, n0, n1, round(n,10))
print(max_diff)
print(num_diff)