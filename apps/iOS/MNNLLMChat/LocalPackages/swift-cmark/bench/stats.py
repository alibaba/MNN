#!/usr/bin/env python3

import sys
import statistics

def pairs(l, n):
        return zip(*[l[i::n] for i in range(n)])

# data comes in pairs:
#    n - time for running the program with no input
#    m - time for running it with the benchmark input
# we measure (m - n)

values = [ float(y) - float(x) for (x,y) in pairs(sys.stdin.readlines(),2)]

print("mean = %.4f, median = %.4f, stdev = %.4f" %
    (statistics.mean(values), statistics.median(values),
      statistics.stdev(values)))

