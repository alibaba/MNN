#!/usr/bin/env bash
# Validate commit message format: [Module:Type] Description
# Example: [LLM:Feature] Add streaming support

MSG=$(head -1 "$1")

# Allow merge commits
if printf '%s\n' "$MSG" | grep -qE "^Merge "; then
    exit 0
fi

# Check format: [Module:Type] Description
if ! printf '%s\n' "$MSG" | grep -qE '^\[[A-Za-z0-9]+:(Feature|Bugfix|Perf|Refact|Style|Doc|Test|Chore)\] .+'; then
    echo "ERROR: Commit message format must be:"
    echo "  [Module:Type] Description"
    echo ""
    echo "  Module: LLM, CPU, Metal, CUDA, OpenCL, Vulkan, Core, Infra, Doc, etc."
    echo "  Type:   Feature, Bugfix, Perf, Refact, Style, Doc, Test, Chore"
    echo ""
    echo "  Example: [LLM:Feature] Add streaming support"
    echo "  Got:     $MSG"
    exit 1
fi

exit 0
