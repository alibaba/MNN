# MNN cv 

MNN cv is a warpper of MNN's expr functions and some cv functions, which provides the APIs like OpenCV. 

## Usage
Compile MNN with cv, using below command:
```bash
cmake -DMNN_BUILD_OPENCV=ON .. && make -j8
```

## Macros
### MNN_IMGPROC_COLOR
`MNN_IMGPROC_COLOR` default is ON, this macro control the color function like: cvtColor.
### MNN_IMGPROC_GEOMETRIC
`MNN_IMGPROC_GEOMETRIC` default is ON, this macro control the geometric function like: resize.
### MNN_IMGPROC_DRAW
`MNN_IMGPROC_DRAW` default is ON, this macro control the draw function like: line.
### MNN_IMGPROC_FILTER
`MNN_IMGPROC_FILTER` default is ON, this macro control the filter function like: blur.
### MNN_IMGPROC_MISCELLANEOUS
`MNN_IMGPROC_MISCELLANEOUS` default is ON, this macro control the miscellaneous function like: threshold.
### MNN_IMGPROC_STRUCTRAL
`MNN_IMGPROC_STRUCTRAL` default is ON, this macro control the structral function like: findContours.
### MNN_IMGCODECS
`MNN_IMGCODECS` default is OFF, this macro control the imgcodecs function like: imread.
### MNN_OPENCV_TEST
`MNN_OPENCV_TEST` default is OFF, this macro control the unit test.
### MNN_OPENCV_BENCH
`MNN_OPENCV_BENCH` default is OFF, this macro control the benchmark of MNN cv with OpenCV.
 
