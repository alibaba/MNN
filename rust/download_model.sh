#!/bin/bash
# Download Qwen3-0.6B model for MNN Rust binding testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MNN Rust Binding - Model Download Script ===${NC}"
echo ""

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo -e "${RED}Error: git-lfs is not installed${NC}"
    echo "Please install git-lfs first:"
    echo "  macOS: brew install git-lfs"
    echo "  Ubuntu: sudo apt-get install git-lfs"
    echo "  Other: https://git-lfs.github.com/"
    exit 1
fi

# Initialize git-lfs
echo -e "${YELLOW}Initializing git-lfs...${NC}"
git lfs install

# Create models directory
MODELS_DIR="../models"
MODEL_NAME="qwen3-0.6b"
MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}"

mkdir -p "${MODELS_DIR}"

# Check if model already exists
if [ -d "${MODEL_PATH}" ]; then
    echo -e "${YELLOW}Model directory already exists: ${MODEL_PATH}${NC}"
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
    rm -rf "${MODEL_PATH}"
fi

# Clone model repository
echo -e "${YELLOW}Cloning Qwen3-0.6B model from ModelScope...${NC}"
git clone https://www.modelscope.cn/MNN/Qwen3-0.6B-MNN.git "${MODEL_PATH}"

# Pull LFS files
echo -e "${YELLOW}Pulling LFS files...${NC}"
cd "${MODEL_PATH}"
git lfs pull

echo ""
echo -e "${GREEN}=== Download Complete! ===${NC}"
echo ""
echo "Model location: ${MODEL_PATH}"
echo ""
echo "Model files:"
ls -lh "${MODEL_PATH}"
echo ""
echo -e "${GREEN}You can now run the example:${NC}"
echo "  cd rust"
echo "  cargo run --example llm_example -- ${MODEL_PATH}/config.json"
echo ""
