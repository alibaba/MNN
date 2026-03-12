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
./build_mnncli/mnncli download <model_name> <repo_name>
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

### `mnncli serve` (OpenAI-compatible)

Available endpoints:
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /chat/completions` (alias)

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

This route is designed for Anthropic-style clients (including Claude-compatible integrations) and supports both non-stream and stream requests.
## Dependencies

- OpenSSL (for HTTPS support)
- MNN core library
- LLM engine library

## Notes

- The tool requires macOS 13.0+ when building on Apple platforms
- On Linux, ensure `libssl-dev` (or equivalent) is installed
- Models are cached in the user's cache directory
- `mnncli serve` currently provides OpenAI-compatible and Anthropic-compatible API routes
