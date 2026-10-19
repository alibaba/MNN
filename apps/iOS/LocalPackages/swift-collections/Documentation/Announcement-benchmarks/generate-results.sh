#!/bin/sh
#===----------------------------------------------------------------------===//
#
# This source file is part of the Swift Collections open source project
#
# Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
#
#===----------------------------------------------------------------------===//

set -eux

../../Utils/run-benchmarks.sh library run results.json --library Library.json --max-size 16M --cycles 20 --mode replace-all
../../Utils/run-benchmarks.sh library render results.json --library Library.json --max-time 10us --min-time 1ns --theme-file Theme.json --percentile 90 --output .
