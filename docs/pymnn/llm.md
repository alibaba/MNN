# MNN LLM Module

`MNN.llm` provides model loading, text generation, tokenization, and access to
generation metrics. Supported multimodal models can also accept image and
audio inputs in their model-specific prompt format.

## Installation and Model Preparation

```bash
pip install MNN
```

An exported model needs `config.json` and the model, tokenizer, and embedding
files referenced by that configuration. Download a compatible MNN model or
follow the [LLM export instructions](../transformers/llm.md).
For a local Hugging Face model, the repository's export entry point is:

```bash
python transformers/llm/export/llmexport.py \
  --path /path/to/huggingface/model \
  --export mnn --quant_bit 4 --dst_path ./exported-model
```

Install the export dependencies described in the export guide before running
this command. The filenames and model paths in this page are examples and must
be replaced with your actual exported model files.

## Quick Start

```python
import MNN.llm as mllm

model = mllm.create("./exported-model/config.json")
model.set_config({"max_new_tokens": 64})
model.load()

response = model.response("Hello, how are you?")
print(response)

model.reset()
model.response("Write a short poem", stream=True)
```

The public Python wrappers accept keyword arguments. They convert arguments
before calling the underlying C extension. Use these wrappers when importing
`MNN.llm`; the C extension's positional-only methods are implementation details.

Run the text-model example from the repository root:

```bash
python pymnn/examples/MNNLlm/llm_demo.py ./exported-model/config.json
```

## Model Creation and Configuration

### create and load

```python
model = mllm.create(config_path, embedding_model=False)
model.load()
```

`create` returns a Python `Llm` object. Set `embedding_model=True` for an
embedding model. Load the model before tokenization or inference. For text
models, inspect `model.context.status` if model loading reports an error;
`LlmStatus.NOT_LOADED` means loading has not completed successfully.

### set_config

```python
model.set_config({
    "backend_type": "cpu",
    "thread_num": 4,
    "precision": "low",
    "max_new_tokens": 64,
    "reuse_kv": False,
})
```

The public API accepts a dictionary and serializes it to JSON for the C++
configuration parser. Set backend and resource options before `load()`. GPU
backends require a compatible build and device; use the exported model's
configuration and the [LLM documentation](https://www.yuque.com/mnn/en/llm)
for backend-specific requirements.

## Text Generation

### response

```python
response = model.response("Explain machine learning", stream=False)
model.response("Write a poem", stream=True)
```

`prompt` is a string or a supported multimodal dictionary. `stream` defaults
to false. False buffers the generated text and returns a string. True writes
generated text to native stdout and returns an empty string. It does not return
an iterator of tokens. Boolean values or positional `0` / `1` are accepted.

`response` applies the model's configured chat template to text prompts when
`use_template` is true. For already formatted history, disable `use_template`
as in the conversation example below, or use tokenization and `generate`.

### reset

```python
model.reset()
```

Clears generation state and KV history. Use it between independent requests.

### apply_chat_template

```python
prompt = model.apply_chat_template("Hello")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain AI"},
]
formatted_prompt = model.apply_chat_template(messages)
```

The argument is a string or a list of dictionaries containing string `role`
and `content` fields. A single role/content dictionary is not a message list.

### tokenizer_encode and tokenizer_decode

```python
token_ids = model.tokenizer_encode("Hello world")
text = "".join(model.tokenizer_decode(token_id) for token_id in token_ids)
```

`tokenizer_encode` returns a list of token IDs. `tokenizer_decode` accepts one
integer token ID and returns its text; decode lists one token at a time.
Some tokenizers include special tokens, so encoded/decoded text need not be
identical to the input string.

### Manual generation: generate_init, forward, and stoped

Use a manual decoding loop for token-level control, following the existing
[VLM example](../../pymnn/examples/MNNLlm/vlm_example.py):

```python
import MNN.numpy as np

prompt = model.apply_chat_template("Tell me a joke")
input_ids = model.tokenizer_encode(prompt)
model.generate_init()
output_ids = []
max_new_tokens = 64
for _ in range(max_new_tokens):
    logits = model.forward(input_ids)
    token = int(np.argmax(logits))
    model.context.current_token = token
    if model.stoped():
        break
    output_ids.append(token)
    input_ids = [token]
response = "".join(model.tokenizer_decode(token_id) for token_id in output_ids)
print(response)
```

The first `forward` processes the full prompt; subsequent calls process only
one selected token. `forward` returns an MNN Var of logits. `MNN.numpy.argmax`
selects a token greedily; this loop does not use the configured native sampler.
Update `model.context.current_token` before calling `stoped()`, which checks
that token against the model's stop tokens. Check before decoding or feeding
it back, including the first token. The spelling `stoped` is the name currently
exported by the Python API.

The Python loop enforces its own token budget. `forward()` does not enforce
`max_new_tokens` from the model configuration. This manual path does not perform
all of the native generation loop's output bookkeeping; use `output_ids` and
`response` above for the generated result rather than expecting context output
fields or generation status to match a `response()` call.

`generate_init(stream=None, end_with=None)` initializes generation state.
The current binding does not forward a Python stream object to C++.
`end_with` is a string written to the native output stream at generation end;
it is not a stop-string matcher. Passing a newline does not stop generation
when the model emits a newline. The manual loop above handles its own output.

### generate binding limitation

The existing public `model.generate(input_ids)` wrapper accepts only
`input_ids`. With the current native binding, this call uses a token limit of
`0` (prefill only), so setting `max_new_tokens` in the configuration does not
make it generate new tokens. Use `response()` for ordinary text generation or
the manual loop above for token-level control.

## Multi-Turn Conversation

Supply message history explicitly when the model's configuration does not
reuse KV state. `reuse_kv` defaults to false; consecutive text `response` calls
alone do not guarantee conversation memory.

```python
import MNN.llm as mllm

model = mllm.create("./exported-model/config.json")
model.set_config({"reuse_kv": False, "use_template": False})
model.load()
messages = []

for query in ["My name is Alice. Reply with my name only.", "What is my name? Reply with my name only."]:
    messages.append({"role": "user", "content": query})
    prompt = model.apply_chat_template(messages)
    response = model.response(prompt)
    messages.append({"role": "assistant", "content": response})
    print(response)

model.reset()
model.set_config({"use_template": True})
```

This sends the full conversation on each turn. Incremental KV reuse requires
model-specific prompt handling and should be configured and validated separately.

## Context and Performance Metrics

```python
model.response("Hello")
ctx = model.context
print(f"Prompt: {ctx.prompt_len} tokens")
print(f"Generated: {ctx.gen_seq_len} tokens")
print(f"Prefill: {ctx.prefill_us / 1000:.2f} ms")
print(f"Decode: {ctx.decode_us / 1000:.2f} ms")
if ctx.decode_us > 0:
    print(f"Speed: {ctx.gen_seq_len / (ctx.decode_us / 1e6):.1f} tokens/sec")
if ctx.status == mllm.LlmStatus.MAX_TOKENS_FINISHED:
    print("Generation reached the token limit")
```

The public `context` property refreshes and returns a `Context` object.
Use its attributes, rather than calling the internal `get_context` method.
Read `model.context` again after a later generation to obtain refreshed metrics.

| Attributes | Meaning |
|---|---|
| `prompt_len`, `gen_seq_len`, `all_seq_len` | Prompt, generated, and total token counts |
| `current_token`, `history_tokens`, `output_tokens` | Current and stored token IDs |
| `generate_str`, `end_with` | Generated text and output suffix |
| `load_us`, `prefill_us`, `decode_us`, `sample_us` | Timings in microseconds |
| `vision_us`, `audio_us` | Multimodal encoder timings in microseconds |
| `pixels_mp`, `audio_input_s` | Processed image megapixels and audio duration |
| `status` | A `mllm.LlmStatus` value |

Statuses include `NOT_LOADED`, `RUNNING`, `NORMAL_FINISHED`,
`MAX_TOKENS_FINISHED`, `USER_CANCEL`, `INTERNAL_ERROR`, and `TIMEOUT`.

`model.get_log()` returns accumulated native logs when built with
`LLM_LOG_TO_STRING`; otherwise the log string may be empty.

## Split LoRA Adapters

```python
import MNN.llm as mllm

base = mllm.create("./exported-model/config.json")
base.load()
adapter = base.create_lora("./math_lora.mnn")
response = adapter.response("Solve x^2 = 4")
print(response)
```

`create_lora` creates and loads a split LoRA model sharing the base module.
It returns a separate `Llm` instance, so do not call `load()` again on that
instance. Keep the base and adapter alive during inference. The Python wrapper
retains the base object for the adapter's lifetime.

Although legacy Python wrappers define `apply_lora`, `select_module`, and
`release_module`, the current C extension does not expose the corresponding
methods. Use `create_lora` with compatible split LoRA exports.

## Embedding Models

```python
import MNN.llm as mllm
import numpy as np

model = mllm.create("./embedding-model/config.json", embedding_model=True)
model.load()
embedding1 = model.txt_embedding("Machine learning is amazing")
embedding2 = model.txt_embedding("AI is fascinating")
vector1 = np.asarray(embedding1.read()).reshape(-1)
vector2 = np.asarray(embedding2.read()).reshape(-1)
denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
if denominator > 0:
    similarity = np.dot(vector1, vector2) / denominator
    print(f"Similarity: {similarity}")
```

`txt_embedding` returns an MNN Var and requires an embedding model. Use `.read()`
before ordinary NumPy operations. `MNN.numpy` also exists and operates on MNN
tensors; it is distinct from ordinary NumPy.

## Multimodal Inputs

Compatible VLMs accept a dictionary with `text` and an `images` list. Image
`data` must be an MNN Var, as returned by `MNN.cv.imread`:

```python
import MNN.llm as mllm
import MNN.cv as cv

model = mllm.create("./vlm-model/config.json")
model.load()
image = cv.imread("photo.jpg")
prompt = {
    "text": "<img>image_0</img>Describe this image",
    "images": [{"data": image, "height": image.shape[0], "width": image.shape[1]}],
}
model.response(prompt, stream=True)
```

The parser assigns image keys `image_0`, `image_1`, etc. Image sizes and prompt
tags must match the model's preprocessing and template requirements.

For supported audio-language models, the binding parses an `audios` list,
whose entries contain `file_path` or a waveform MNN Var under `waveform`.
It assigns keys `audio_0`, `audio_1`, etc. A top-level `audio` field is not
parsed. Use the actual model's required text tags, sample rate, and export
configuration; an arbitrary Whisper export is not automatically an MNN LLM
audio-language model.

## Resources

- [LLM export guide](../transformers/llm.md)
- [Python examples](../../pymnn/examples/MNNLlm/)
- [MNN LLM documentation](https://www.yuque.com/mnn/en/llm)
