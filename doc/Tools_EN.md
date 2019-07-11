[中文版本](Tools_CN.md)

# Tools

When compiling with cmake, build products also contain tools used for testing, as explained below.

## MNNV2Basic.out
### Usage
Test performance and dump tensor data.

### Parameters
``` bash
./MNNV2Basic.out temp.mnn 10 0 0 1x3x224x224 4
```

- The first parameter specifies the file name of MNN model.
- The second parameter specifies the number of loop times for the performance test, and 10 indicates that it will run 10 times to test the performance.
- The third parameter specifies whether to dump the the inference intermediate tensor. 0: don't dump, 1: dump output of all operators, 2: dump the input and output for all operators. If dump, the directory is "output".
- The fourth parameter specifies the computing device that performs the inference. Valid values are 0 (floating point CPU), 1 (Metal), 3 (OpenCL), 6 (OpenGL), 7 (Vulkan).
- The fifth parameter specifies the size of the input tensor, which generally does not need to be specified.
- The sixth parameter specifies the thread number, default set as 4, only valid for CPU.

### Default input and output
Only single input, single output is supported. The input is read from "input_0.txt" and the output is dumped to "output.txt".


## checkFile.out
### Usage
Check if the two tensor text files are consistent.

### Parameters
``` bash
./checkFile.out XXX.txt YYY.txt 0.1.
```

- 0.1 means absolute threshold, 0.0001 if not specified.
- When the comparison value exceeds the absolute threshold, it will be output directly to the console.


## checkDir.out
### Usage
Compare the files with the same name in the two folders.

### Parameters
``` bash
./checkDir.out output android_output 1
```

- 1 means absolute threshold, 0.0001 if not specified.
- When the comparison value exceeds the absolute threshold, it will be output directly to the console.


## timeProfile.out
### Usage
Operators' total time-consuming statistic.

### Parameters
``` bash
./timeProfile.out temp.mnn 10 0 1x3x224x224
```

- The first parameter is filename of model.
- The second parameter is run times, default 100.
- The third parameter is the forward type, default 0.
- The fourth parameter is input tensor size, generally needn't be specified.

### Outputs
- The first column is the operator's type.
- The second column is the average time consuming.
- The third column is time-consuming percent.
- Example:

```
Node Type                Avg(ms)       %             Called times
Softmax                  0.018100      0.022775      1.000000
Pooling                  0.080800      0.101671      1.000000
ConvolutionDepthwise     14.968399     18.834826     13.000000
Convolution              64.404617     81.040726     15.000000
total time : 79.471924 ms, total mflops : 2271.889404
```

## backendTest.out
### Usage
This tool compares the results of the inferences performed by the specified computing device and CPU. Read input_0.txt in the current directory as input by default.

### Parameters
``` bash
./backendTest.out temp.mnn 3 0.15
```

- First parameter: model file.
- Second parameter: computing device that performs inference.
- Third parameter: error tolerance.
