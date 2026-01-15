# AI Agent Instructions

## Compilation
To compile this project, please use the provided `build.sh` script in the root directory.

```bash
./build.sh
```

This script handles the two-stage build process:
1. Building the MNN static library with appropriate optimizations (Metal, LLM support, etc.).
2. Building the `mnncli` executable.

Do not attempt to run `cmake` manually unless you have a specific reason to bypass the standard build flow.
