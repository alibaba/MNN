# MNN Hexagon Backend

This directory contains the Hexagon backend implementation for MNN. It supports accelerating model inference on Qualcomm Hexagon DSPs.

## Prerequisites

- **Hexagon SDK**: Ensure you have downloaded and installed the Qualcomm Hexagon SDK.
- **Environment Variable**: Set the `HEXAGON_SDK_ROOT` environment variable pointing to the root of your Hexagon SDK installation.

## Compilation Steps

### 1. Compile the Custom HTP Ops Library

Before compiling MNN, you need to build the custom Hexagon Tensor Processor (HTP) operator library located in `htp-ops-lib/`.

```bash
cd source/backend/hexagon/htp-ops-lib

# Pass the target DSP architecture version (e.g., v73, v75, v79)
# Ensure your Hexagon SDK environment is properly set up before running this script
bash build.sh v79
```

This script will generate two essential libraries in the `htp-ops-lib/outputs/` directory:
- `libMNN_htpops.so`: The Android AArch64 stub library that runs on the CPU.
- `libMNN_htpops_skel.so`: The Hexagon DSP skeleton library that runs on the Hexagon NPU/DSP.

### 2. Compile MNN with Hexagon Backend Enabled

When configuring the MNN build, you need to enable the Hexagon backend by passing the `-DMNN_HEXAGON=ON` flag to CMake. Don't need Hexagon SDK.

```bash
cd /path/to/MNN
mkdir build && cd build

cmake .. \
  -DMNN_HEXAGON=ON \
  # ... other MNN compilation flags (e.g., cross-compiling for Android)

make -j8
```

*Note: If you have already exported `HEXAGON_SDK_ROOT` in your environment, you can omit the `-DHEXAGON_SDK_ROOT` CMake argument.*

### 3. Deployment and Execution

To run the compiled MNN with the Hexagon backend on an Android device:

1. Push your compiled MNN executable and libraries to the device.
2. Push the generated `libMNN_htpops.so` and `libMNN_htpops_skel.so` libraries to the device.
3. Configure your `LD_LIBRARY_PATH` so the system can find `libMNN_htpops.so`.
4. Configure the `ADSP_LIBRARY_PATH` environment variable to include the directory containing `libMNN_htpops_skel.so` so the DSP can load the skeleton library.

```bash
# Example on device:
export ADSP_LIBRARY_PATH="/data/local/tmp/hexagon_libs;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp"
export LD_LIBRARY_PATH="/data/local/tmp/hexagon_libs:$LD_LIBRARY_PATH"
```
