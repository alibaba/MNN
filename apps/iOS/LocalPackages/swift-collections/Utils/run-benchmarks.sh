#!/bin/sh

set -eu

srcroot="$(dirname "$0")/../Benchmarks"
results="batch.results"

command="${1:-help}"
shift || :

run() {
  local flags
  flags="-c release"
  flags="$flags -Xswiftc -Xllvm -Xswiftc -align-module-to-page-size"
  swift run --package-path "$srcroot" $flags benchmark "$@"
}

case "$command" in
    batch-run)
        revision="$(git rev-parse HEAD)"
        run library run "$results" \
              --source-url "https://github.com/apple/swift-collections/tree/$revision" \
              --mode replace-all \
              --max-size 4M --cycles 3 \
              --amortized-cutoff 10us \
              "$@"
        ;;
    batch-render)
        run library render "$results" \
              --min-time 100ps --max-time 10us \
              --min-size 1 --max-size 4M \
              "$@"
        ;;
    batch-clean)
        rm -rf "$out"/results
        ;;
    *)
        run "$command" "$@"
        ;;
esac
