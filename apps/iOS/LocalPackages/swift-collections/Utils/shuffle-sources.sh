#!/bin/sh
#
# Randomly reorder Swift source files in the specified directory, then
# try rebuilding the package, in a loop.
#
# This script is useful to help reproducing nondeterministic issues
# with the Swift compiler's MergeModules phase, as in
# https://github.com/apple/swift-collections/issues/7
#

set -eu


function shuffle() {
    input="$1"

    tmp="$(dirname "$input")/tmp"
    mkdir "$tmp"
    local i=0
    ls "$input"/*.swift | sort -R | while read file; do
        mv "$file" "$tmp/$i.swift"
        i=$(($i + 1))
    done

    mv "$tmp"/*.swift "$input"
    rmdir "$tmp"
}

if [[ -z "${1+set}" || ! -d "$1" ]]; then
    echo "Usage: $0 <directory> <build-arguments>..." >&2
    exit 1
fi

dir="$1"
shift

cd "$(dirname "$0")/.."

i=0
while true; do
    echo
    echo "$i"
    
    rm -rf .build
    shuffle "$dir"
    xcrun swift build "$@"
    i="$(($i + 1))"
done
