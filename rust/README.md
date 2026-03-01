# MNN Rust Binding

Rust bindings for MNN (Mobile Neural Network) LLM API.

## Prerequisites

Before building the Rust binding, you need to build MNN first:

```bash
cd /path/to/MNN/build
cmake ..
make -j8
```

## Building

```bash
cd rust
cargo build
```

**Important**: The Rust bindings link dynamically with the MNN library. You need to set the library path before running:

```bash
# macOS
export DYLD_LIBRARY_PATH=/path/to/MNN/build:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=/path/to/MNN/build:$LD_LIBRARY_PATH

# Windows
set PATH=C:\path\to\MNN\build;%PATH%
```

For the examples in this repository:
```bash
# macOS
export DYLD_LIBRARY_PATH=../build:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=../build:$LD_LIBRARY_PATH
```

## Downloading Models

### Quick Start: Using the Download Script

We provide a convenient script to download the Qwen3-0.6B model:

```bash
cd rust
./download_model.sh
```

This will:
1. Check if git-lfs is installed
2. Clone the Qwen3-0.6B model from ModelScope
3. Pull all LFS files
4. Place the model in `../models/qwen3-0.6b/`

### Manual Download

If you prefer to download manually:

```bash
# Install git-lfs (if not already installed)
# macOS:
brew install git-lfs

# Ubuntu/Debian:
# sudo apt-get install git-lfs

# Initialize git-lfs
git lfs install

# Clone the model
mkdir -p ../models
git clone https://www.modelscope.cn/MNN/Qwen3-0.6B-MNN.git ../models/qwen3-0.6b

# Pull LFS files
cd ../models/qwen3-0.6b
git lfs pull
```

### Model Directory Structure

After downloading, your model directory should contain:

```
models/qwen3-0.6b/
├── config.json           # Model configuration
├── llm.mnn              # Model structure
├── llm.mnn.weight       # Model weights
├── llm_config.json      # LLM configuration
└── tokenizer.txt        # Tokenizer
```

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
mnn = { path = "/path/to/MNN/rust" }
```

### Quick Start Example

After downloading the model, you can run the included examples:

```bash
cd rust

# Set library path (macOS)
export DYLD_LIBRARY_PATH=../build:$DYLD_LIBRARY_PATH

# Run the basic example
cargo run --example llm_example -- ../models/qwen3-0.6b/config.json

# Or run the comprehensive test
cargo run --example qwen_inference
```

### Basic Usage

```rust
use mnn::Llm;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create LLM from config path
    let mut llm = Llm::create("../models/qwen3-0.6b/config.json")?;

    // Load model
    llm.load()?;

    // Generate response
    let response = llm.response("你好")?;
    println!("{}", response);

    Ok(())
}
```

## API

### `Llm`

- `Llm::create(config_path: &str)` - Create a new LLM instance
- `llm.load()` - Load the model
- `llm.response(query: &str)` - Generate response for a query
- `llm.generate(input_ids: &[i32])` - Generate output tokens from input tokens
- `llm.reset()` - Reset conversation history
- `llm.set_config(config: &str)` - Set configuration
- `llm.dump_config()` - Get current configuration

## License

Same as MNN project.
