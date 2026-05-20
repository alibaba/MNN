#!/usr/bin/env bash
# Only check format of changed lines (not whole files).
# Requires: git-clang-format (usually comes with clang-format)

if ! command -v git-clang-format &> /dev/null; then
    echo "Error: git-clang-format is not installed."
    echo "Install it via: brew install clang-format (macOS) or apt install clang-format (Linux)"
    exit 1
fi

output=$(git clang-format --diff --staged --extensions cpp,c,h,hpp,cc,m,mm 2>&1)

if [ "$output" = "no modified files to format" ] || \
   [ "$output" = "clang-format did not modify any files" ]; then
    exit 0
fi

if echo "$output" | grep -q "^diff"; then
    echo "The following changed lines have format issues:"
    echo ""
    echo "$output"
    echo ""
    echo "To fix, run:"
    echo "  git clang-format --staged --extensions cpp,c,h,hpp,cc,m,mm"
    echo "  git add -u"
    exit 1
fi

# Unknown output, treat as error
if [ -n "$output" ]; then
    echo "Unexpected output from git-clang-format:"
    echo "$output"
    exit 1
fi

exit 0
