# MNN JavaScript Bindings

JavaScript/Node.js bindings for MNN (Mobile Neural Network), enabling high-performance deep learning inference in Node.js applications.

## Features

- 🚀 High-performance neural network inference
- 💻 Cross-platform support (Linux, macOS, Windows)
- 📦 TypeScript type definitions included
- 🔧 Simple and intuitive API
- ⚡ Based on native C++ addon using N-API

## Installation

### Prerequisites

- Node.js >= 14.0.0
- CMake >= 3.15
- C++ compiler (GCC/Clang/MSVC)
- MNN library built

### Install

```bash
npm install @alibaba/mnn
```

### Build from source

```bash
# Clone MNN repository
git clone https://github.com/alibaba/MNN.git
cd MNN

# Build MNN core library
mkdir build && cd build
cmake .. -DMNN_BUILD_SHARED_LIBS=ON
make -j4
cd ..

# Build JavaScript bindings
cd js
npm install
npm run build
```

## Quick Start

```javascript
const mnn = require('@alibaba/mnn');

// Load model
const interpreter = mnn.Interpreter.createFromFile('mobilenet.mnn');

// Create inference session
const config = {
    type: mnn.ForwardType.CPU,
    numThread: 4
};
const session = interpreter.createSession(config);

// Get input tensor
const input = interpreter.getSessionInput(session, 'data');
console.log('Input shape:', input.getShape()); // [1, 3, 224, 224]

// Prepare input data (224x224x3 RGB image)
const inputData = new Float32Array(1 * 3 * 224 * 224);
// ... fill inputData with preprocessed image data
input.copyFrom(inputData);

// Run inference
const result = interpreter.runSession(session);
if (result === mnn.ErrorCode.NO_ERROR) {
    console.log('Inference successful');
}

// Get output
const output = interpreter.getSessionOutput(session, 'prob');
const predictions = output.getData();
console.log('Predictions:', predictions);

// Cleanup
interpreter.release();
```

## API Reference

### Interpreter

#### Static Methods

##### `Interpreter.createFromFile(path: string): Interpreter`
Create interpreter from model file.

##### `Interpreter.createFromBuffer(buffer: Buffer): Interpreter`
Create interpreter from buffer.

#### Instance Methods

##### `createSession(config?: ScheduleConfig): Session`
Create inference session with optional configuration.

##### `resizeSession(session: Session): void`
Resize session after changing input tensor dimensions.

##### `runSession(session: Session): ErrorCode`
Run inference on the session.

##### `getSessionInput(session: Session, name?: string): Tensor`
Get input tensor by name (or first input if name not provided).

##### `getSessionOutput(session: Session, name?: string): Tensor`
Get output tensor by name (or first output if name not provided).

##### `release(): void`
Release interpreter resources.

### Tensor

##### `getShape(): number[]`
Get tensor shape as array.

##### `getDataType(): DataType`
Get tensor data type.

##### `getData(): TypedArray`
Get tensor data as TypedArray (Float32Array, Int32Array, etc.).

##### `copyFrom(data: TypedArray): void`
Copy data into tensor.

### Types

#### ScheduleConfig
```typescript
interface ScheduleConfig {
    type?: ForwardType;      // Backend type (CPU, OPENCL, etc.)
    numThread?: number;      // Number of threads (for CPU backend)
    backupType?: ForwardType; // Fallback backend type
    mode?: number;           // Session mode
}
```

#### ForwardType
```typescript
enum ForwardType {
    CPU = 0,
    METAL = 1,
    OPENCL = 2,
    OPENGL = 3,
    VULKAN = 4,
    NN = 5,
    CUDA = 6,
}
```

#### ErrorCode
```typescript
enum ErrorCode {
    NO_ERROR = 0,
    OUT_OF_MEMORY = 1,
    NOT_SUPPORT = 2,
    COMPUTE_SIZE_ERROR = 3,
    NO_EXECUTION = 4,
}
```

#### DataType
```typescript
enum DataType {
    FLOAT = 0,
    INT32 = 1,
    INT64 = 2,
    UINT8 = 3,
}
```

## Examples

See the [examples](./examples) directory for complete examples:

- [Basic Inference](./examples/inference.js) - Simple inference example
- [MobileNet Classification](./examples/mobilenet.js) - Image classification with MobileNet

## Performance

MNN JavaScript bindings provide near-native performance by using N-API to directly call MNN's C++ API with minimal overhead.

## Contributing

Contributions are welcome! Please see the main [MNN repository](https://github.com/alibaba/MNN) for contribution guidelines.

## License

Apache License 2.0

## Acknowledgements

- MNN Team at Alibaba
- Node.js N-API team
