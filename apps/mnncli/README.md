# MNNCLI

> Note: This project is under active development and may contain bugs or unfinished features. Use with caution.

MNNCLI is a command-line interface tool for MNN (Mobile Neural Network) that provides various functionalities for working with LLM models.

## Features

- **Model Management**: List, download, and delete models
- **Model Serving**: Start a web server to serve models via HTTP API
- **Model Execution**: Run models with prompts or prompt files
- **Benchmarking**: Performance benchmarking for models
- **Model Search**: Search for models in the Hugging Face repository

## Building

To build MNNCLI, run the following commands from the mnncli directory:

```bash
sh build.sh
```

The executable will be located at `build_mnncli/mnncli`.

## Usage

### List Models
```bash
./build_mnncli/mnncli list
```

### Serve Model
```bash
./build_mnncli/mnncli serve <model_name>
```

Optional host/port:
```bash
./build_mnncli/mnncli serve <model_name> --host 127.0.0.1 --port 8000
```

Open the server address (by default `http://127.0.0.1:8000/`) for a basic text chat with streamed replies.
The conversation stays in the current tab. If the server uses `--api-key`, enter the key on the page.

### Run Model
```bash
./build_mnncli/mnncli run <model_name> [-c config_path] [-p prompt] [-f prompt_file]
```

### Benchmark Model
```bash
./build_mnncli/mnncli benchmark <model_name> [-c config_path]
```

### Download Model
```bash
./build_mnncli/mnncli download <model_name>
```

### Search Models
```bash
./build_mnncli/mnncli search <keyword>
```

### Delete Model
```bash
./build_mnncli/mnncli delete <model_name>
```

## HTTP API Compatibility

### OpenAI-compatible endpoints

Available endpoints:
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /chat/completions` (alias)
- `POST /v1/embeddings` (requires an embedding model)

Minimal `curl` examples:

```bash
curl http://127.0.0.1:8000/v1/models
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-0.8B-MNN",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```


### Anthropic-compatible (`/v1/messages`)

Available endpoint:
- `POST /v1/messages`

Minimal `curl` example:

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{
    "model": "Qwen3.5-0.8B-MNN",
    "max_tokens": 128,
    "messages": [{
      "role": "user",
      "content": [{"type": "text", "text": "Hello"}]
    }]
  }'
```

This route supports text and image requests in the Anthropic Messages format, including non-streaming and streaming responses.

The server loads one chat model and an optional embedding model at startup. Use the corresponding ID from `/v1/models` in each request.
`GET /api/status` reports the active models, chat model configuration and vision support.
When using `--config`, an optional positional model name sets the ID; otherwise the config directory name is used.
Each request supplies its complete conversation history. Requests are serialized and their KV state is reset between
requests. `max_tokens` is supported on both APIs, and OpenAI requests may instead use `max_completion_tokens`.
Token counts and finish reasons come from the inference engine; OpenAI streaming also supports
`stream_options: {"include_usage": true}`. OpenAI responses include prefill and decode timings in `mnn_timings`,
including in the final usage chunk when requested.

The OpenAI and Anthropic chat routes accept text and base64 PNG/JPEG images when the loaded model supports vision.
OpenAI uses `image_url` data URLs; Anthropic uses base64 `image` source blocks. Remote image URLs, audio and video are
not supported. The maximum request size is 16 MiB and each decoded image is limited to 16 million pixels.

Both chat APIs reject tool definitions and tool-call history with HTTP 400.

Requests may set `temperature` (0 selects greedy decoding), `top_p`, and `top_k`; these overrides apply only to that
request. Positive temperatures must be at least 0.00001; when `top_p` or `top_k` is supplied without `temperature`,
the temperature is 1. Omitting all three uses the model's configured sampler. Structured output, stop sequences and Ollama's native
API are not implemented. Recognized unsupported generation options are rejected.

To expose embeddings alongside chat, provide a separate exported MNN embedding model:

```bash
mnncli serve my-chat --embedding-config /path/to/embedding/config.json --embedding-model my-embedding
curl http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"my-embedding","input":["hello", "world"]}'
```

The endpoint accepts one string or up to 128 strings and returns L2-normalized float vectors with the model's native
dimension. `dimensions`, base64 output and token-array input are not supported. Chat and embedding inference share a
lock and are processed serially.

The default bind address is `127.0.0.1`. Authentication is disabled unless an API key is supplied:

```bash
mnncli serve my-model --api-key YOUR_KEY
curl http://127.0.0.1:8000/v1/models -H 'Authorization: Bearer YOUR_KEY'
```

The server also accepts the API key in the `x-api-key` header. Use a TLS-terminating proxy for remote HTTPS access.

### API tests

For a model-backed API check, run `test/test_anthropic_api_contract.sh --model MODEL_NAME`, optionally adding
`--config /path/to/config.json`. This checks text chat and authentication as well as the Anthropic event sequence.
With deterministic model defaults, set `MNNCLI_VERIFY_GREEDY=1` to verify that a sampling override expires after one request.
To test a running server, use `python3 test/test_api_features.py --url URL --model CHAT_ID`. Add `--embedding-model ID`
to test embeddings, or `--vision-url URL --vision-model ID --check-vision-colors` to test image inputs and verify the
model distinguishes red from blue.

## Dependencies

- OpenSSL (for HTTPS support)
- MNN core library
- LLM engine library

## Notes

- The tool requires macOS 13.0+ when building on Apple platforms
- On Linux, ensure `libssl-dev` (or equivalent) is installed
- Models are cached in the user's cache directory
- `mnncli serve` currently provides OpenAI-compatible and Anthropic-compatible API routes
