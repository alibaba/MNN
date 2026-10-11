# MNN Python Examples

This directory contains Python examples demonstrating how to use MNN's Python API (pymnn) for various deep learning tasks.

Run the commands below from the repository root unless a section explicitly
changes directory. Replace model and data paths with your own files.

## Directory Structure

```
pymnn/examples/
├── MNNEngineDemo/     # Low-level MNN Interpreter API examples
├── MNNExpr/           # High-level Express API examples
├── MNNLlm/            # Large Language Model examples
└── MNNTrain/          # Training examples
```

---

## MNNEngineDemo - Low-Level API

Examples using the low-level `MNN.Interpreter` API for maximum control over model execution.

### mobilenet_demo.py

Basic image classification using MobilenetV1.

**Usage:**
```bash
python pymnn/examples/MNNEngineDemo/mobilenet_demo.py model.mnn image.jpg
```

**Features:**
- Load and run MNN models
- Image preprocessing (resize, normalize)
- Tensor manipulation
- Session configuration

### mobilenet_demo_2.py

MobilenetV1 example with runtime management, session information, and caching.

```bash
python pymnn/examples/MNNEngineDemo/mobilenet_demo_2.py model.mnn image.jpg
```

### gpu_session_demo.py

Demonstrates GPU acceleration using OpenCL/Metal backend.

**Usage:**
```bash
python pymnn/examples/MNNEngineDemo/gpu_session_demo.py model.mnn image.jpg
```

**Features:**
- GPU backend configuration
- Backend information and cache tuning

---

## MNNExpr - High-Level Express API

Examples using MNN's Express API - a NumPy-like interface for easier model building and inference.

### mobilenet_demo.py

Image classification using Express API (simpler than Engine API).

**Usage:**
```bash
python pymnn/examples/MNNExpr/mobilenet_demo.py model.mnn image.jpg
```

**Features:**
- High-level `MNN.expr` API
- Simplified tensor operations
- Automatic shape inference

### mnn_numpy_cv_demo.py

Image classification using MNN's NumPy-compatible API and `MNN.cv`.

```bash
python pymnn/examples/MNNExpr/mnn_numpy_cv_demo.py model.mnn image.jpg
```

**Features:**
- Image loading, resizing, and normalization
- Tensor layout conversion
- NumPy-style array operations

### gpu_express_demo.py

GPU acceleration with Express API.

```bash
python pymnn/examples/MNNExpr/gpu_express_demo.py model.mnn image.jpg
```

The selected backend must be enabled in your MNN build.

### gpu_io_interface.py

GPU memory management and I/O operations.

```bash
python pymnn/examples/MNNExpr/gpu_io_interface.py model.mnn input_0.txt output.txt
```

This example uses PyTorch CUDA tensors and requires a compatible CUDA-enabled
PyTorch installation and MNN build. `output.txt` contains reference values.

**Features:**
- GPU tensor allocation
- Host-device data transfer
- Memory optimization techniques

---

## MNNLlm - Large Language Models

Examples for running LLMs and VLMs with MNN.

### llm_example.py

Run text generation models (GPT-style, LLaMA, etc.).

**Usage:**
```bash
python pymnn/examples/MNNLlm/llm_example.py path/to/model/config.json
```

**Features:**
- Autoregressive text generation
- Streaming text to stdout
- Token-ID generation

The prompt is set in the script. Sampling options come from the model's
configuration; this script does not parse `--model` or `--prompt` options.

### vlm_example.py

Vision-Language Models (e.g., LLaVA, Qwen-VL).

**Usage:**
```bash
cd pymnn/examples/MNNLlm
python vlm_example.py /absolute/path/to/model/config.json
```

**Features:**
- Multi-modal input (image + text)
- Visual question answering
- Image captioning

The image path and prompt are set in the script. Run from the indicated
directory so its relative `resource/images/cat.jpg` path resolves correctly;
the script does not parse `--image` or `--prompt` options.

---

## MNNTrain - Training Examples

Examples demonstrating on-device training with MNN.

### mnist/

Train a simple CNN on MNIST dataset.

**Files:**
- `train_mnist.py`: Training script
- `dataset.py`: MNIST data loader

**Usage:**
```bash
cd pymnn/examples/MNNTrain/mnist
python train_mnist.py
```

### mobilenet_finetune/

Transfer learning: fine-tune MobileNet on custom dataset.

**Files:**
- `mobilenet_transfer.py`: Fine-tuning script
- `finetune_dataset.py`: Custom dataset loader

**Usage:**
```bash
cd pymnn/examples/MNNTrain/mobilenet_finetune
python mobilenet_transfer.py \
  --model_file /path/to/mobilenet_v2.mnn \
  --train_image_folder /path/to/train/images \
  --train_txt /path/to/train.txt \
  --test_image_folder /path/to/test/images \
  --test_txt /path/to/test.txt \
  --num_classes 10
```

### simple/

Simple training examples for learning MNN's training API.

**Files:**
- `solve_equation.py`: Learn to solve linear equations
- `grad_loss.py`: Basic gradient computation
- `make_solve_equation_graph.py`: Build computation graph

### module_save/

Examples of saving and loading trained models.

**Files:**
- `test_save.py`: Save/load model weights
- `grad_test.py`: Gradient persistence

---

## Audio Processing

Use `MNN.audio` for WAV loading and speech feature extraction; see the audio
Quick Start below and [audio documentation](../../docs/pymnn/audio.md).

---

## Installation

Install MNN Python package:

```bash
# From PyPI
pip install MNN

# From source
cd pymnn/pip_package
python build_deps.py llm
python setup.py install
```

---

## Common Prerequisites

Most examples require:

```bash
pip install numpy opencv-python
```

For the MNIST example, also install:

```bash
pip install mnist
```

---

## Model Files

Examples expect MNN model files (`.mnn` format). Convert from other formats using MNN's model converter:

```bash
# Convert ONNX to MNN
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode biz
```

See [tools/converter/README.md](../../tools/converter/README.md) for details.

Pre-converted models for testing:
- Download from [MNN Model Zoo](https://www.yuque.com/mnn/en/model_zoo)
- Or use the quick test models in `resource/model/`

---

## Quick Start

### 1. Image Classification

This example uses the input/output names and preprocessing expected by the
MobileNetV1 example model. Adjust them for other exported models.

```python
import MNN
import cv2
import numpy as np

net = MNN.nn.load_module_from_file(
    "mobilenet.mnn", ["input"], ["MobilenetV1/Predictions/Reshape_1"]
)
image = cv2.imread("cat.jpg")
if image is None:
    raise ValueError("Could not read cat.jpg")
image = cv2.resize(image[..., ::-1], (224, 224))
image = ((image - (103.94, 116.78, 123.68)) * 0.017).astype(np.float32)
input_var = MNN.expr.const(image[None], [1, 224, 224, 3], MNN.expr.NHWC)
input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
output = net.forward([input_var])[0]
output = MNN.expr.convert(output, MNN.expr.NHWC)
print(f"Predicted class: {np.argmax(output.read())}")
```

### 2. GPU Acceleration

```python
import MNN

config = {
    'backend': 'OPENCL',
    'precision': 'low'
}

interpreter = MNN.Interpreter("model.mnn")
session = interpreter.createSession(config)
```

Use `'METAL'` on macOS when supported by your build. Backend availability
depends on how MNN was compiled and on the device.

### 3. Audio Feature Extraction

```python
import MNN.audio as audio

waveform, sr = audio.load("speech.wav", sr=16000)
features = audio.fbank(waveform, sample_rate=sr, n_mels=80)
```

---

## API Comparison

| Task | Low-Level (Engine) | High-Level (Express) |
|------|-------------------|---------------------|
| **Pros** | Maximum control, explicit memory management | Simple, NumPy-like, less boilerplate |
| **Cons** | Verbose, manual tensor management | Less control over execution |
| **Best for** | Production deployment, fine-tuned performance | Prototyping, research, quick experiments |

**Recommendation:** Start with Express API for learning, switch to Engine API for production optimization.

---

## Resources

- [MNN Python API Documentation](https://www.yuque.com/mnn/en/python)
- [MNN Official Documentation](https://www.yuque.com/mnn/en)
- [MNN GitHub Repository](https://github.com/alibaba/MNN)

---

## Contributing

Found a bug or want to add an example? Please open an issue or submit a PR!
