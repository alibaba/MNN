# MNN LLM Module

The MNN LLM module provides efficient inference for Large Language Models (LLMs) on edge devices, supporting both text-only and multimodal (vision, audio) models with optimized performance for mobile and embedded platforms.

## Features

- **Text Generation**: Autoregressive token generation with streaming support
- **Multimodal Support**: Vision-Language Models (VLMs) and Audio-Language Models
- **LoRA Adapters**: Dynamic adapter loading and switching
- **Edge-Optimized**: Quantization (INT4/INT8), KV cache management, and memory efficiency
- **Flexible Backends**: CPU, GPU (Metal/OpenCL/Vulkan), NPU support

---

## Installation

```bash
pip install MNN
```

---

## Quick Start

```python
import MNN.llm as mllm

# Create and load model
model = mllm.create('./qwen-1.8b-int4/config.json')
model.load()

# Generate response
response = model.response('Hello, how are you?', 0)
print(response)

# Stream response
# Note: stream=1 outputs to stdout and returns empty string
model.response(prompt, 1)  # Outputs directly to stdout
```

---

## Core Concepts

### Model Creation and Loading

Models are loaded from a directory containing:
- `config.json`: Model configuration
- `*.mnn` files: Quantized model weights
- `tokenizer.txt`: Tokenizer vocabulary

```python
# Create LLM instance
llm = mllm.create('./model_dir/config.json')

# Load model weights into memory
llm.load()
```

### Generation Modes

**Simple Response** (recommended):
```python
# Blocking generation
response = llm.response('What is AI?', 0)

# Streaming generation
# Note: stream=1 outputs to stdout and returns empty string
llm.response(prompt, 1)  # Outputs directly to stdout
```

**Manual Token Generation** (advanced):
```python
# Encode prompt to token IDs
input_ids = llm.tokenizer_encode('Hello')

# Generate output token IDs
output_ids = llm.generate(input_ids)

# Decode tokens to text
text = llm.tokenizer_decode(output_ids)
```

---

## API Reference

### Module Functions

#### create()

Create an LLM instance from config file.

```python
llm = mllm.create(
    config_path,           # Path to config.json
    embedding_model=False  # True for embedding models
)
```

**Parameters:**
- `config_path` (str): Path to `config.json` or model directory
- `embedding_model` (bool): Whether this is an embedding model (default: False)

**Returns:** `Llm` instance

---

### Llm Class

#### load()

Load model weights into memory.

```python
llm.load()
```

Must be called before inference.

---

#### response()

Generate text response from prompt (high-level API).

```python
output = llm.response(
    prompt,        # Input text or dict
    stream         # int: 0 for buffered, 1 for stdout streaming
)
```

**Parameters:**
- `prompt` (str | dict): Input prompt
  - `str`: Plain text prompt
  - `dict`: Structured prompt (see Chat Templates)
- `stream` (int): Streaming mode (default: 0)
  - `0`: Buffered - returns complete response string
  - `1`: Streaming - outputs directly to stdout, returns empty string

**Returns:**
- If `stream=0`: Complete response string
- If `stream=1`: Empty string (output goes to stdout)

**Example:**
```python
# Buffered response
response = llm.response('What is machine learning?', 0)
print(response)

# Streaming to stdout
llm.response('Write a poem', 1)  # Outputs directly, returns ""
```

**Note:** The Python binding does NOT support token-by-token iteration. Use `stream=1` for real-time output to stdout.

---

#### apply_chat_template()

Apply model's chat template to format prompts.

```python
formatted_prompt = llm.apply_chat_template(prompt)
```

**Parameters:**
- `prompt` (str | dict): Raw prompt or message dict

**Returns:** Formatted prompt string

**Example:**
```python
# String input
prompt = llm.apply_chat_template('Hello')

# Dict input (multi-turn conversation)
prompt = llm.apply_chat_template({
    'role': 'user',
    'content': 'Explain AI'
})
```

---

#### tokenizer_encode()

Encode text to token IDs.

```python
token_ids = llm.tokenizer_encode(prompt)
```

**Parameters:**
- `prompt` (str | dict): Text to encode
  - `str`: Plain text
  - `dict`: Multimodal input (see Multimodal section)

**Returns:** `list[int]` - Token IDs

**Example:**
```python
# Text encoding
ids = llm.tokenizer_encode('Hello world')
# Output: [151644, 872, 198, ...]

# Multimodal encoding (VLM)
ids = llm.tokenizer_encode({
    'text': '<img>image_0</img>Describe this',
    'images': [{
        'data': image_array,
        'height': 224,
        'width': 224
    }]
})
```

---

#### tokenizer_decode()

Decode token IDs to text.

```python
text = llm.tokenizer_decode(token_ids)
```

**Parameters:**
- `token_ids` (list[int]): Token IDs to decode

**Returns:** `str` - Decoded text

---

#### forward()

Low-level forward pass (single step).

```python
logits = llm.forward(input_ids)
```

**Parameters:**
- `input_ids` (list[int]): Input token IDs

**Returns:** Logits for next token prediction

**Use Case:** Custom sampling, beam search, advanced generation control

---

#### generate()

Generate complete sequence from input token IDs.

```python
output_ids = llm.generate(input_ids)
```

**Parameters:**
- `input_ids` (list[int]): Input token IDs

**Returns:** `list[int]` - Generated token IDs

**Note:** Lower-level than `response()`, returns token IDs instead of text.

---

#### txt_embedding()

Extract text embeddings (for embedding models only).

```python
embedding = llm.txt_embedding(prompt)
```

**Parameters:**
- `prompt` (str): Input text

**Returns:** `MNN.expr.Var` - Embedding vector

**Use Case:** Semantic search, text similarity, retrieval systems

---

#### set_config()

Configure model runtime settings.

```python
llm.set_config({
    'backend_type': 'metal',  # cpu, metal, opencl, vulkan
    'thread_num': 4,
    'precision': 'low',       # low, normal, high
    'memory_mode': 'low'      # low, normal
})
```

**Common Options:**
- `backend_type`: Inference backend
- `thread_num`: CPU threads
- `precision`: Computation precision
- `memory_mode`: Memory optimization level

---

#### reset()

Reset model state (clear KV cache and history).

```python
llm.reset()
```

**Use Case:** Start a new conversation or generation session.

---

#### generate_init()

Initialize generation with custom parameters.

```python
llm.generate_init(
    stream=None,    # Output stream
    end_with=None   # Stop token
)
```

**Parameters:**
- `stream`: Custom output stream (optional)
- `end_with` (str): Stop generation when this token appears

**Example:**
```python
# Stop at newline
llm.generate_init(None, "\n")
output_ids = llm.generate(input_ids)
```

---

#### stoped()

Check if generation has stopped.

```python
is_done = llm.stoped()
```

**Returns:** `bool` - True if generation completed

---

#### get_log()

Retrieve internal logs (requires compile-time flag).

```python
log = llm.get_log()
```

**Returns:** Accumulated log string

**Note:** Only works if MNN was compiled with `LLM_LOG_TO_STRING=ON`

---

### LoRA Adapters

#### create_lora()

Create a split LoRA instance sharing the base model.

```python
adapter_llm = base_llm.create_lora('./lora_weights.mnn')
```

**Parameters:**
- `lora_path` (str): Path to split LoRA weights

**Returns:** New `Llm` instance with adapter

**Use Case:** Multiple adapters without reloading base model

**Example:**
```python
# Load base model once
base = mllm.create('./qwen-int4/config.json')
base.load()

# Create multiple adapters
math_adapter = base.create_lora('./math_lora.mnn')
math_adapter.load()

code_adapter = base.create_lora('./code_lora.mnn')
code_adapter.load()

# Use adapters independently
math_response = math_adapter.response('Solve x^2 = 4', 0)
code_response = code_adapter.response('Write Python code', 0)
```

**Note:** `apply_lora()`, `select_module()`, and `release_module()` are NOT available in the Python binding.

---

---

#### create_lora()

Create a split LoRA instance sharing the base model.

```python
adapter_llm = base_llm.create_lora('./lora_weights.mnn')
```

**Parameters:**
- `lora_path` (str): Path to split LoRA weights

**Returns:** New `Llm` instance with adapter

**Use Case:** Multiple adapters without reloading base model

**Example:**
```python
# Load base model once
base = mllm.create('./qwen-int4/config.json')
base.load()

# Create multiple adapters
math_adapter = base.create_lora('./math_lora.mnn')
code_adapter = base.create_lora('./code_lora.mnn')

# Use adapters independently
math_response = math_adapter.response('Solve x^2 = 4', 0)
code_response = code_adapter.response('Write Python code', 0)
```

---


---


---

### Context Object

Access generation state and performance metrics.

```python
ctx = llm.get_context()
```

#### Properties

**Sequence Lengths:**
```python
ctx['prompt_len']       # Input prompt length
ctx['gen_seq_len']      # Generated sequence length
ctx['all_seq_len']      # Total sequence length
```

**Tokens:**
```python
ctx['current_token']    # Current token being generated
ctx['history_tokens']   # Previous tokens in conversation
ctx['output_tokens']    # Generated output tokens
```

**Generation Control:**
```python
ctx['end_with']         # Stop token
ctx['generate_str']     # Generated string so far
ctx['status']           # LlmStatus enum
```

**Performance Metrics:**
```python
ctx['load_us']          # Model load time (microseconds)
ctx['prefill_us']       # Prefill phase time
ctx['decode_us']        # Decode phase time
ctx['sample_us']        # Sampling time
ctx['vision_us']        # Vision encoding time (VLM)
ctx['audio_us']         # Audio encoding time
```

**Multimodal Metrics:**
```python
ctx['pixels_mp']        # Megapixels processed (VLM)
ctx['audio_input_s']    # Audio seconds processed
```

**Example:**
```python
response = llm.response('Hello', 0)

ctx = llm.get_context()
print(f'Prompt: {ctx['prompt_len']} tokens')
print(f'Generated: {ctx['gen_seq_len']} tokens')
print(f'Prefill: {ctx['prefill_us'] / 1000:.2f} ms')
print(f'Decode: {ctx['decode_us'] / 1000:.2f} ms')
```

---

### LlmStatus Enum

Generation status codes:

```python
LlmStatus.NOT_LOADED          # Model not loaded
LlmStatus.RUNNING             # Generation in progress
LlmStatus.NORMAL_FINISHED     # Completed normally
LlmStatus.MAX_TOKENS_FINISHED # Hit max token limit
LlmStatus.USER_CANCEL         # User cancelled
LlmStatus.INTERNAL_ERROR      # Error occurred
LlmStatus.TIMEOUT             # Generation timeout
```

**Example:**
```python
response = llm.response('Write a story', 0)
if llm.get_context().status == mllm.LlmStatus.MAX_TOKENS_FINISHED:
    print('Response truncated due to max tokens')
```

---

## Multimodal Models

### Vision-Language Models (VLMs)

```python
import MNN.llm as mllm
import MNN.cv as cv

# Load VLM (e.g., Qwen-VL, LLaVA)
vlm = mllm.create('./qwen-vl-int4/config.json')
vlm.load()

# Load image
image = cv.imread('photo.jpg')

# Multimodal prompt
prompt = {
    'text': '<img>image_0</img>What is in this image?',
    'images': [{
        'data': image,
        'height': image.shape[0],
        'width': image.shape[1]
    }]
}

# Generate description
response = vlm.response(prompt)
print(response)
```

### Audio-Language Models

```python
import MNN.llm as mllm
import MNN.audio as audio

# Load audio model
alm = mllm.create('./whisper-int4/config.json')
alm.load()

# Load audio
waveform, sr = audio.load('speech.wav', sr=16000)

# Audio prompt (model-specific format)
prompt = {
    'audio': waveform,
    'sample_rate': sr
}

response = alm.response(prompt)
```

---

## Examples

### Basic Text Generation

```python
import MNN.llm as mllm

# Load model
llm = mllm.create('./qwen-1.8b-int4/config.json')
llm.load()

# Single-turn generation
response = llm.response('Explain quantum computing in simple terms', 0)
print(response)

# Streaming generation
print('Response: ', end='')
# Note: stream=1 outputs to stdout and returns empty string
llm.response(prompt, 1)  # Outputs directly to stdout
print()
```

### Multi-Turn Conversation

```python
import MNN.llm as mllm

llm = mllm.create('./qwen-1.8b-int4/config.json')
llm.load()

# Turn 1
response1 = llm.response('My name is Alice', 0)
print(f'Bot: {response1}')

# Turn 2 (model remembers context via KV cache)
response2 = llm.response('What is my name?', 0)
print(f'Bot: {response2}')  # Should mention "Alice"

# Clear history for new conversation
llm.reset()
```

### Custom Sampling

```python
import MNN.llm as mllm

llm = mllm.create('./qwen-1.8b-int4/config.json')
llm.load()

# Encode prompt
prompt_text = llm.apply_chat_template('Tell me a joke')
input_ids = llm.tokenizer_encode(prompt_text)

# Generate with custom control
llm.generate_init(None, '\n')  # Stop at newline
output_ids = llm.generate(input_ids)

# Decode
response = llm.tokenizer_decode(output_ids)
print(response)
```

### LoRA Adapter Switching

```python
import MNN.llm as mllm

# Load base model
base = mllm.create('./llama-7b-int4/config.json')
base.load()

# Create multiple LoRA instances
math_llm = base.create_lora('./math_lora.mnn')
math_llm.load()

code_llm = base.create_lora('./code_lora.mnn')
code_llm.load()

# Use math adapter
math_response = math_llm.response('What is the integral of x^2?', 0)
print(f'Math: {math_response}')

# Use code adapter
code_response = code_llm.response('Write a quicksort in Python', 0)
print(f'Code: {code_response}')
```

### Performance Monitoring

```python
import MNN.llm as mllm

llm = mllm.create('./qwen-1.8b-int4/config.json')
llm.load()

# Generate
response = llm.response('Hello world', 0)

# Get metrics
ctx = llm.get_context()
print(f'Prompt tokens: {ctx['prompt_len']}')
print(f'Generated tokens: {ctx['gen_seq_len']}')
print(f'Prefill time: {ctx['prefill_us'] / 1000:.2f} ms')
print(f'Decode time: {ctx['decode_us'] / 1000:.2f} ms')
print(f'Speed: {ctx['gen_seq_len'] / (ctx['decode_us'] / 1e6):.1f} tokens/sec')
```

### Embedding Extraction

```python
import MNN.llm as mllm

# Load embedding model (second parameter is embedding_model flag: 0 or 1)
model = mllm.create('./bge-small/config.json', 1)
model.load()

# Extract embeddings
emb1 = model.txt_embedding('Machine learning is amazing')
emb2 = model.txt_embedding('AI is fascinating')

# Compute similarity (use numpy, not MNN.numpy)
import numpy as np
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f'Similarity: {similarity}')
```

---

## Model Preparation

### Converting Models

Use `mnnllm-convert` to convert Hugging Face models:

```bash
# Install conversion tools
pip install transformers torch

# Convert model
mnnllm-convert \
  --model Qwen/Qwen-1.8B-Chat \
  --quant 4 \
  --output ./qwen-1.8b-int4
```

### Model Structure

```
model_dir/
├── config.json          # Model configuration
├── embeddings.mnn       # Token embeddings
├── lm.mnn               # Language model layers
├── tokenizer.txt        # Tokenizer vocabulary
└── visual.mnn           # Vision encoder (VLMs only)
```

---

## Performance Tips

1. **Use Quantization**: INT4 models are 4x smaller with minimal accuracy loss
2. **Enable GPU**: Configure `backend_type` for Metal/OpenCL/Vulkan
3. **Optimize Threads**: Set `thread_num` to match CPU cores
4. **Batch Context**: Reuse KV cache for multi-turn conversations (don't call `reset()`)
5. **Stream Output**: Use `stream=1` for better perceived latency (outputs to stdout)
6. **Profile**: Monitor `get_context()` metrics to identify bottlenecks

---

## Supported Models

- **Qwen** (Qwen-1.8B, Qwen-7B, Qwen-VL)
- **LLaMA** (LLaMA-2, LLaMA-3)
- **ChatGLM** (ChatGLM-6B, ChatGLM2)
- **Baichuan** (Baichuan-7B, Baichuan-13B)
- **Phi** (Phi-2, Phi-3)
- **Embedding Models** (BGE, M3E)

See [MNN Model Zoo](https://www.yuque.com/mnn/en/model_zoo) for pre-converted models.

---

## Resources

- [MNN LLM Documentation](https://www.yuque.com/mnn/en/llm)
- [Model Conversion Guide](https://www.yuque.com/mnn/en/llm_convert)
- [Example Code](../../pymnn/examples/MNNLlm/)
- [GitHub Repository](https://github.com/alibaba/MNN)
