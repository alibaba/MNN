#!/usr/bin/env bash
# Prevent committing files larger than 1MB.

MAX_SIZE=1048576
failed=0

git diff --cached --name-only --diff-filter=ACMR -z | while IFS= read -r -d '' f; do
    size=$(wc -c < "$f" 2>/dev/null || echo 0)
    if [ "$size" -gt "$MAX_SIZE" ]; then
        echo "ERROR: $f is $((size / 1024))KB, exceeds 1MB limit"
        # Write to temp file since subshell can't set parent variable
        echo 1 > /tmp/.pre-commit-large-file-failed
    fi
done

if [ -f /tmp/.pre-commit-large-file-failed ]; then
    rm -f /tmp/.pre-commit-large-file-failed
    exit 1
fi

exit 0
