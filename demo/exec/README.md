[TOC]

# Build this Demo on Linux or Mac

Use [Top CMakeLists.txt](../../CMakeLists.txt) to construct demo like this:

```bash
cd path/to/MNN
mkdir build && cd build
cmake -DMNN_BUILD_DEMO=ON ..
make -j8
```

# Build this Demo on Windows

Use [Top CMakeLists.txt](../../CMakeLists.txt) to construct demo like this:
```powershell
cd path/to/MNN
mkdir build
cd build
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_DEMO=ON ..
nmake
```

# Demo Programs

## pictureRecognition

Basic image classification with batch inference support.

```bash
./pictureRecognition.out model.mnn input0.jpg input1.jpg input2.jpg ...
```

Arguments:
- `model.mnn`: MNN model file
- `input*.jpg`: Input images (batch size = number of images)

This demo requests the automatic backend and reports the selected backend,
memory usage, and FLOPS. Available backends depend on the build and device.

Example:
```bash
./pictureRecognition.out mobilenet_v1.mnn cat.jpg dog.jpg
```

## pictureRecognition_batch

Image classification with batch processing.

```bash
./pictureRecognition_batch.out model.mnn imagedir/ groundtruth.txt quantized.json [batchsize] [total_imgs]
```

The image directory must include its trailing separator. `groundtruth.txt`
contains one image filename and label list per line, separated by a space;
multiple labels are comma-separated. `quantized.json` configures image
preprocessing and runtime options. Batch size defaults to 10 and total images
defaults to 50; the image list must contain enough entries for the requested run.

## pictureRecognition_module

Image classification using the Module API.

```bash
./pictureRecognition_module.out model.mnn config.json input0.jpg input1.jpg ...
```

`config.json` provides image preprocessing settings such as `format`, `width`,
`height`, `mean`, and `normal`, and the optional `CPU` runtime setting.

## MultiPose

Human pose estimation using MobileNet-based models.

1. Download [pose model](https://github.com/czy2014hust/posenet-python/raw/master/models/model-mobilenet_v1_075.pb)
2. [Convert](../../tools/converter/README.md) tensorflow model to MNN model
3. Run:

```bash
./multiPose.out model.mnn input.jpg pose.png
```

Arguments:
- `model.mnn`: Converted pose model
- `input.jpg`: Input image
- `pose.png`: Output with detected keypoints

## Segment

Semantic segmentation using DeepLabv3.

Download the model from:
https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite

```bash
./segment.out model.mnn input.jpg result.png
```

Arguments:
- `model.mnn`: Converted DeepLabv3 model
- `input.jpg`: Input image
- `result.png`: Segmentation mask output

## transformerDemo

Transformer model inference (e.g., Neural Machine Translation).

```bash
./transformerDemo.out model.mnn
```

The model must have specific input/output names: `NmtModel/Placeholder`, `NmtModel/Placeholder_1`, `NmtModel/transpose_2`.

Outputs are written to `0_output.txt` and `1_output.txt`. The program runs 10 iterations for benchmarking.

Note: Input sequences are hardcoded. Modify the source for custom inputs.

## nluDemo

Natural Language Understanding inference for BERT-like models.

```bash
./nluDemo.out data_dir model.mnn [num_threads] [seq_length]
```

Arguments:
- `data_dir`: Directory with input files (see below)
- `model.mnn`: Converted NLU model
- `num_threads`: CPU threads (default: 1)
- `seq_length`: Sequence length (default: 128)

Required files in `data_dir`:
- `input_ids.txt`: Token IDs
- `input_mask.txt`: Attention mask
- `segment_ids.txt`: Segment IDs

Outputs `mnn.out` and `mnn.out2` for verification.

Example:
```bash
./nluDemo.out ./bert_data bert_model.mnn 4 128
```

## expressDemo

MNN Express API usage example.

```bash
./expressDemo.out model.mnn [backend_type] [test_time]
```

Loads an existing model with the Express API, reads input values from
`input_0.txt` in the working directory, and writes output values to `output.txt`.
`backend_type` defaults to CPU (0), and `test_time` defaults to 10.

## expressMakeModel

Model creation with the Express API.

```bash
./expressMakeModel.out
```

Writes a procedurally generated model to `pos.mnn`, then loads `seg.mnn` from
the working directory to build `mask.mnn`. The segmentation model must provide
the `sub_7` input and `ResizeBilinear_3` output expected by the source.

## multithread_imgrecog

Multi-threaded image recognition for performance testing.

```bash
./multithread_imgrecog.out model.mnn input0.jpg input1.jpg ...
```

Runs one CPU inference thread per input image using a cloned module.

## pictureRotate

Image preprocessing and rotation using MNN's image utilities.

```bash
./pictureRotate.out input.jpg angle output.png
```

The angle in degrees is required and precedes the output filename. The demo
writes PNG data.

---

## Notes

### Model Conversion

Model inference demos require MNN format (`.mnn`). Use the
[MNN Converter](../../tools/converter/README.md) to convert supported source
models. `pictureRotate` works with images and requires no model.

### Backend Selection

Backend selection is specific to each demo. `pictureRecognition` requests
automatic selection; `expressDemo` accepts a backend argument, and
`multithread_imgrecog` explicitly uses CPU. Build support and device availability
determine which backends can run.

### Performance

For better performance, enable caching:
```cpp
net->setCacheFile(".cachefile");
net->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);
```
The cache file is created on first run and reused afterward.
