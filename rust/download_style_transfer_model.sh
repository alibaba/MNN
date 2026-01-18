#!/bin/bash
set -e

# Default settings
MODEL_URL="https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx"
MODEL_ONNX="candy-9.onnx"
MODEL_MNN="style_transfer.mnn"
IMAGE_URL="https://raw.githubusercontent.com/alibaba/MNN/master/resource/images/cat.jpg"
IMAGE_FILE="input.jpg"
OUTPUT_FILE="output.jpg"

# Check for MNNConvert
if ! command -v MNNConvert &> /dev/null; then
    echo "Warning: MNNConvert not found in PATH."
    echo "If you haven't converted the model yet, please ensure MNNConvert is available."
    # We proceed assuming the user might already have the .mnn file or will get it.
fi

# Download model if .mnn doesn't exist
if [ ! -f "$MODEL_MNN" ]; then
    if [ -f "$MODEL_ONNX" ]; then
        echo "Found $MODEL_ONNX"
    else
        echo "Downloading ONNX model..."
        wget -q "$MODEL_URL" -O "$MODEL_ONNX" || curl -L "$MODEL_URL" -o "$MODEL_ONNX"
    fi

    if command -v MNNConvert &> /dev/null; then
        echo "Converting ONNX model to MNN..."
        MNNConvert -f ONNX --modelFile "$MODEL_ONNX" --MNNModel "$MODEL_MNN" --bizCode biz
    else
        echo "Error: MNNConvert tool is required to convert the model but was not found."
        echo "Please install MNN tools or provide '$MODEL_MNN' manually."
        exit 1
    fi
else
    echo "Found $MODEL_MNN"
fi

# Download test image
if [ ! -f "$IMAGE_FILE" ]; then
    echo "Downloading test image..."
    wget -q "$IMAGE_URL" -O "$IMAGE_FILE" || curl -L "$IMAGE_URL" -o "$IMAGE_FILE"
fi

# Run the Rust example
echo "Running Style Transfer example..."
echo "Command: cargo run --example style_transfer -- \"$MODEL_MNN\" \"$IMAGE_FILE\" \"$OUTPUT_FILE\""

# Note: This requires the MNN libraries to be available in the library path
# You might need to set LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS)
# e.g., export LD_LIBRARY_PATH=path/to/mnn/lib:$LD_LIBRARY_PATH

cargo run --example style_transfer -- "$MODEL_MNN" "$IMAGE_FILE" "$OUTPUT_FILE"

if [ -f "$OUTPUT_FILE" ]; then
    echo "Success! Output saved to $OUTPUT_FILE"
else
    echo "Failed to generate output."
    exit 1
fi
