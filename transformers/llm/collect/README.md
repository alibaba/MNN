# MNN Model Analysis Tools

This repository contains Python scripts for analyzing MNN  models using different callback mechanisms to collect statistics during model inference.

## Scripts Overview

### 1. get_max_values.py
Collects maximum activation values from MNN model layers during inference.

### 2. get_thresholds.py  
Calculates sparsity thresholds for MNN model layers based on target sparsity levels.

## Requirements

```bash
pip install datasets torch tqdm
```

You'll also need to build pymnn

```bash
cd /path/to/MNN/pymnn/pip_package
python build_deps.py llm
python setup.py install
```

## Usage

### Get Max Values

```bash
cd /path/to/MNN/transformers/llm/collect
python get_max_values.py -m <mnn_model_path> [options]
```

**Arguments:**
- `-m, --mnn-path`: Path to MNN model config (required)
- `-d, --eval_dataset`: Dataset for evaluation (default: 'wikitext/wikitext-2-raw-v1')
- `-o, --output-path`: Output file path (default: 'max_values.json')
- `-l, --length`: Sample length for processing (default: 512)

**Example:**
```bash
python get_maxval.py --m /path/to/MNN/transformers/llm/export/model/config.json -o ./max_val_test.json
```

### Get Thresholds

```bash
cd /path/to/MNN/transformers/llm/collect
python get_thresholds.py -m <mnn_model_path> [options]
```

**Arguments:**
- `-m, --mnn-path`: Path to MNN model config(required)
- `-d, --eval_dataset`: Dataset for evaluation (default: 'wikitext/wikitext-2-raw-v1')
- `-o, --output-path`: Output file path (default: 'thresholds.json')
- `-t, --target-sparsity`: Target sparsity level (default: 0.5)
- `-l, --length`: Sample length for processing (default: 512)

**Example:**
```bash
python get_thredsholds.py -m /path/to/MNN/transformers/llm/export/model/config.json -l 1024 -t 0.5 -o ./thresholds_0.5.json
```

## How It Works

Both scripts:
1. Load an MNN model and configure it for analysis
2. Load a text dataset (default: WikiText-2)
3. Tokenize and process the dataset text
4. Run model inference to collect statistics via callbacks
5. Save results to JSON files

The key difference is in the callback configuration:
- **Max Values**: Uses `enable_max_value_callback` to collect maximum activation values
- **Thresholds**: Uses `enable_threshold_callback` with target sparsity to calculate pruning thresholds

## Output

Both scripts generate JSON files containing the collected statistics that can be used for model optimization, pruning, or quantization analysis.