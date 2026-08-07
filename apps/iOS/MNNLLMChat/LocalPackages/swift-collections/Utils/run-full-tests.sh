#!/bin/sh
#===----------------------------------------------------------------------
#
# This source file is part of the Swift Collections open source project
#
# Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
#
#===----------------------------------------------------------------------

# Build & test this package in as many configurations as possible.
# It is a good idea to run this script before each release to uncover
# potential problems not covered by the usual CI runs.
#
# Note that this is not a fully automated solution -- some manual editing will
# sometimes be required when e.g. testing the package on older Xcode releases.

set -eu

cd "$(dirname $0)/.."

build_dir="$(mktemp -d "/tmp/$(basename $0).XXXXX")"

bold_on="$(tput bold)"
bold_off="$(tput sgr0)"

red_on="$(tput setaf 1)"
red_off="$(tput sgr0)"

spm_flags=""

if [ "$(uname)" = "Darwin" ]; then
    swift="xcrun swift"
else
    swift="swift"
    spm_flags="$spm_flags -j 1"

    if $swift --version | grep 'Swift version 5\.3\.' > /dev/null; then
        spm_flags="$spm_flags --enable-test-discovery"
    fi
fi

echo "Build output logs are saved in $bold_on$build_dir$bold_off"
$swift --version

_count=0
try() {
    label="$1"
    shift
    _count=$(($_count + 1))
    count="$(printf "%02d" $_count)"
    output="$build_dir/$count.$label.log"
    echo "$bold_on[$count $label]$bold_off $@"
    start="$(date +%s)"
    if "$@" >"$output" 2>&1; then
        end="$(date +%s)"
        echo "  Completed in $(($end - $start))s"
    else
        end="$(date +%s)"
        echo "  ${red_on}${bold_on}Failed in $(($end - $start))s.${bold_off}${red_off}" \
             "${red_on}See $output for full console output.${red_off}"
        tail -10 "$output" | sed 's/^/  /'
    fi
}

# Build using SPM
try "spm.debug" $swift test -c debug $spm_flags --build-path "$build_dir/spm.debug"
try "spm.release" $swift test -c release $spm_flags --build-path "$build_dir/spm.release"

# Build with CMake
cmake_build_dir="$build_dir/cmake"
try "cmake.generate" cmake -S . -B "$cmake_build_dir" -G Ninja
try "cmake.build-with-ninja" ninja -C "$cmake_build_dir"

# Build using xcodebuild
try_xcodebuild() {
    label="$1"
    destination="$2"
    shift 2
    
    try "$label" \
        xcrun xcodebuild -scheme swift-collections-Package \
        -configuration Release \
        -destination "$destination" \
        -derivedDataPath "$build_dir/xcodebuild" \
        "$@"
}

if [ "$(uname)" = "Darwin" ]; then
    # Build using xcodebuild
    try_xcodebuild "xcodebuild.build.macOS" "generic/platform=macOS" build
    try_xcodebuild "xcodebuild.build.macCatalyst" "generic/platform=macOS,variant=Mac Catalyst" build
    try_xcodebuild "xcodebuild.build.iOS" "generic/platform=iOS" build
    try_xcodebuild "xcodebuild.build.iOS-simulator" "generic/platform=iOS Simulator" build
    try_xcodebuild "xcodebuild.build.watchOS" "generic/platform=watchOS" build
    try_xcodebuild "xcodebuild.build.watchOS-simulator" "generic/platform=watchOS Simulator" build
    try_xcodebuild "xcodebuild.build.tvOS" "generic/platform=tvOS" build
    try_xcodebuild "xcodebuild.build.tvOS-simulator" "generic/platform=tvOS Simulator" build

    try_xcodebuild "xcodebuild.test.macOS" "platform=macOS" test
    try_xcodebuild "xcodebuild.test.macCatalyst" "platform=macOS,variant=Mac Catalyst" test
    try_xcodebuild "xcodebuild.test.iOS-simulator" "platform=iOS Simulator,name=iPhone 12" test
    try_xcodebuild "xcodebuild.test.watchOS-simulator" "platform=watchOS Simulator,name=Apple Watch Series 6 - 44mm" test
    try_xcodebuild "xcodebuild.test.tvOS-simulator" "platform=tvOS Simulator,name=Apple TV 4K (at 1080p) (2nd generation)" test
fi

# Build benchmarks
cd Benchmarks
try "benchmarks.debug" $swift run -c debug $spm_flags --build-path "$build_dir/benchmarks.debug" benchmark --help
try "benchmarks.release" $swift run -c release $spm_flags --build-path "$build_dir/benchmarks.release" benchmark --help
cd ..
if [ "$(uname)" = "Darwin" ]; then
    # FIXME: This should just work on Linux, too
    try "run-benchmarks" ./Utils/run-benchmarks.sh run --max-size 128 "$build_dir/benchmark.results" --cycles 3
fi

# Build with custom configurations
try "spm.internal-checks" $swift test -c debug -Xswiftc -DCOLLECTIONS_INTERNAL_CHECKS $spm_flags --build-path "$build_dir/spm.internal-checks"
try "spm.deterministic-hashing" $swift test -c debug -Xswiftc -DCOLLECTIONS_DETERMINISTIC_HASHING $spm_flags --build-path "$build_dir/spm.deterministic-hashing"
try "spm.internal-checks+deterministic-hashing" $swift test -c debug -Xswiftc -DCOLLECTIONS_INTERNAL_CHECKS -Xswiftc -DCOLLECTIONS_DETERMINISTIC_HASHING $spm_flags --build-path "$build_dir/spm.internal-checks+deterministic-hashing"

# Some people like to build their dependencies in warnings-as-errors mode.
# (This won't work with tests on Linux as we have some tests for deprecated APIs that will produce a warning there.)
try "spm.warnings-as-errors.debug"  $swift build -c debug -Xswiftc -warnings-as-errors $spm_flags --build-path "$build_dir/spm.warnings-as-errors"
try "spm.warnings-as-errors.release"  $swift build -c release -Xswiftc -warnings-as-errors $spm_flags --build-path "$build_dir/spm.warnings-as-errors"

# Build with library evolution enabled. (This configuration is
# unsupported, but let's make some effort not to break it.)
if [ "$(uname)" = "Darwin" ]; then
    try "spm.library-evolution" \
        $swift build \
        -c release \
        $spm_flags \
        --build-path "$build_dir/spm.library-evolution"  \
        -Xswiftc -enable-library-evolution
    try "xcodebuild.library-evolution" \
        xcodebuild -scheme swift-collections-Package \
        -destination "generic/platform=macOS" \
        -destination "generic/platform=iOS" \
        BUILD_LIBRARY_FOR_DISTRIBUTION=YES
fi

$swift package clean
