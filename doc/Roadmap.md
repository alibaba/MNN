# Roadmap
=========

We plan to release a stable version every two months.

## Model Optimization
1. Adds support for PyTorch (Caffe2) and MXNet in converter
2. Improve converter graph optimization
3. Improve support for quantification and add support for sparse calculation

## Scheduling optimization
1. Add model flops statistics
2. Add dynamic scheduling policies according to hardware features

## Calculation Optimization
1. Continuous optimization on existing backends (CPU/OpenGL/OpenCL/Vulkan/Metal)
2. Optimize Arm v8.2 backend to support quantitative models
3. Add NPU backend using NNAPI
4. Apply fast matrix multiplication and Winograd algorithm to better performance

## Other
1. Documentation and examples
2. Improve test and benchmark related tools
3. Support more Op
