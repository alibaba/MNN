# MNN Python Examples

This directory contains Python examples demonstrating how to use MNN's Python API (pymnn) for various deep learning tasks.

## Directory Structure

```
pymnn/examples/
├── MNNEngineDemo/     # Low-level MNN Interpreter API examples
├── MNNExpr/           # High-level Express API examples
├── MNNLlm/            # Large Language Model examples
├── MNNTrain/          # Training examples
└── audio_demo.py      # Audio processing example
```

---

## MNNEngineDemo - Low-Level API

Examples using the low-level `MNN.Interpreter` API for maximum control over model execution.

### mobilenet_demo.py

Basic image classification using MobilenetV1.

**Usage:**
```bash
python mobilenet_demo.py <model.mnn> <image.jpg>
```

**Features:**
- Load and run MNN models
- Image preprocessing (resize, normalize)
- Tensor manipulation
- Session configuration

### mobilenet_demo_2.py

Alternative MobilenetV1 example with different preprocessing approach.

### gpu_session_demo.py

Demonstrates GPU acceleration using OpenCL/Metal backend.

**Usage:**
```bash
python gpu_session_demo.py <model.mnn> <image.jpg>
```

**Features:**
- GPU backend configuration
- Performance comparison CPU vs GPU

---

## MNNExpr - High-Level Express API

Examples using MNN's Express API - a NumPy-like interface for easier model building and inference.

### mobilenet_demo.py

Image classification using Express API (simpler than Engine API).

**Usage:**
```bash
python mobilenet_demo.py <model.mnn> <image.jpg>
```

**Features:**
- High-level `MNN.expr` API
- Simplified tensor operations
- Automatic shape inference

### mnn_numpy_cv_demo.py

Computer vision operations using MNN's NumPy-compatible API.

**Features:**
- Image operations (resize, crop, rotate)
- Color space conversions
- NumPy-style array operations

### gpu_express_demo.py

GPU acceleration with Express API.

### gpu_io_interface.py

GPU memory management and I/O operations.

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
python llm_example.py --model <model_dir> --prompt "Your prompt here"
```

**Features:**
- Autoregressive text generation
- Token-by-token decoding
- Temperature and sampling control

### vlm_example.py

Vision-Language Models (e.g., LLaVA, Qwen-VL).

**Usage:**
```bash
python vlm_example.py --model <model_dir> --image <image.jpg> --prompt "Describe this image"
```

**Features:**
- Multi-modal input (image + text)
- Visual question answering
- Image captioning

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
cd mnist
python train_mnist.py
```

### mobilenet_finetune/

Transfer learning: fine-tune MobileNet on custom dataset.

**Files:**
- `mobilenet_transfer.py`: Fine-tuning script
- `finetune_dataset.py`: Custom dataset loader

**Usage:**
```bash
cd mobilenet_finetune
python mobilenet_transfer.py --dataset <your_data_dir>
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

### audio_demo.py

Audio feature extraction and processing.

**Usage:**
```bash
python audio_demo.py
```

**Features:**
- Audio I/O (load/save)
- Spectrogram computation
- Mel-spectrogram extraction
- Fbank features for speech recognition
- Whisper-compatible preprocessing

See [docs/pymnn/audio.md](../../docs/pymnn/audio.md) for detailed API reference.

---

## Installation

Install MNN Python package:

```bash
# From PyPI
pip install MNN

# From source
cd pymnn/pip_package
python setup.py install
```

---

## Common Prerequisites

Most examples require:

```bash
pip install numpy opencv-python
```

For training examples, also install:

```bash
pip install torchvision  # for dataset utilities
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

```python
import MNN
import cv2
import numpy as np

# Load model
net = MNN.nn.load_module_from_file("mobilenet.mnn", [], [])

# Preprocess image
img = cv2.imread("cat.jpg")
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0
input_var = MNN.expr.const(img.transpose(2, 0, 1)[None], [1, 3, 224, 224])

# Run inference
output = net.forward(input_var)
print(f"Predicted class: {output.argmax()}")
```

### 2. GPU Acceleration

```python
import MNN

# Configure GPU backend
config = {
    'backend': MNN.Backend_OPENCL,  # or Backend_METAL on macOS
    'precision': 'low'               # low/normal/high
}

interpreter = MNN.Interpreter("model.mnn")
session = interpreter.createSession(config)
```

### 3. Audio Feature Extraction

```python
import MNN.audio as audio

# Load audio
waveform, sr = audio.load("speech.wav", sr=16000)

# Extract features for ASR
features = audio.fbank(waveform, sample_rate=16000, n_mels=80)
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
