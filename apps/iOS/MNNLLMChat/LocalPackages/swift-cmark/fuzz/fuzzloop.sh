#!/bin/bash

# Stop when an error is found
set -e

# Create a corpus sub-directory if it doesn't already exist.
mkdir -p corpus

# The memory and disk usage grows over time, so this loop restarts the
# fuzzer every 4 hours. The `-merge=1` option is used to minimize the
# corpus on each iteration.
while :
do
    date
    echo restarting loop

    # Minimize the corpus
    mv corpus/ corpus2
    mkdir corpus
    echo minimizing corpus
    ./fuzz/fuzz_quadratic -merge=1 corpus ../bench corpus2/ -max_len=1024
    rm -r corpus2

    # Run the fuzzer for 4 hours
    date
    echo start fuzzer
    ./fuzz/fuzz_quadratic corpus -dict=../test/fuzzing_dictionary -jobs=$(nproc) -workers=$(nproc) -max_len=1024 -max_total_time=14400
done
