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

### `mnncli serve` (text chat APIs)

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

This route supports a subset of Anthropic-style text, image and tool requests, including non-streaming and streaming responses.

The server loads one model at startup. `/v1/models` reports that model's ID, and requests must use the same ID.
When using `--config`, an optional positional model name sets the ID; otherwise the config directory name is used.
Each request supplies its complete conversation history. Requests are serialized and their KV state is reset between
requests. `max_tokens` is supported on both APIs, and OpenAI requests may instead use `max_completion_tokens`.
Token counts and finish reasons come from the inference engine; OpenAI streaming also supports
`stream_options: {"include_usage": true}`.

The OpenAI and Anthropic chat routes accept text and base64 PNG/JPEG images when the loaded model supports vision.
OpenAI uses `image_url` data URLs; Anthropic uses base64 `image` source blocks. Remote image URLs, audio and video are
not supported. The maximum request size is 16 MiB and each decoded image is limited to 16 million pixels.

Tool calling is supported for models whose chat template uses the `<tool_call>` format, such as Qwen templates.
Both JSON calls and Qwen3.5-style XML function/parameter calls are supported.
OpenAI `tools` and Anthropic `tools`/`tool_use`/`tool_result` are supported with automatic tool selection. When tools
are present, streaming output is buffered until tool calls can be parsed. Tool execution remains the client's job.
Only `auto` and `none` tool choices are supported. A model can still return plain text or invalid calls; the server
does not guarantee schema-constrained arguments.

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

The Anthropic route also accepts `x-api-key`. The built-in web page accepts the key as the `token` URL parameter.
The server does not provide TLS; use a TLS-terminating proxy for remote access. Command-line keys and URL parameters
may be visible in process listings or browser history.

### MNN Studio

Open the server address in a browser to use the built-in chat interface. Conversations are stored in the browser's
local storage; generation settings apply to individual requests. Image attachments are available when the loaded model
supports vision. The Models and API pages describe the currently running service, including an optional embedding model.
The Appearance panel supports system, light and dark modes and four accent colors. Appearance preferences are stored
locally for the current browser or desktop app.
The chat's Tools switch enables local time and arithmetic. For general-purpose tools, open **Tools and permissions**,
select an absolute workspace directory, choose the permissions for the current session, and enable it. Enabling a
session also turns on the chat's Tools switch. The text beside the switch shows whether command and file tools are
available; without a session, only time and arithmetic are available. The model can request shell commands and list,
read, or write files; each requested action
shows its exact arguments for a one-time approval. File tools require relative paths inside the workspace and reject
symbolic links. Shell commands run with the user's full OS permissions, can access other directories and the network,
and have a configurable timeout (up to 60 seconds). This is an approval interface, not an OS sandbox. Stopping a
response or closing a session does not necessarily stop an already approved action.

To use MCP tools, enable MCP and select an absolute path to a JSON configuration such as:

```json
{
  "mcpServers": {
    "local": {"command": "/path/to/server", "args": [], "env": {}},
    "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer YOUR_TOKEN"}}
  }
}
```

Connect a server from the permissions panel and approve its startup or connection. Its discovered tools are listed
in the panel and are available to the model while connected; every call requires separate approval. Local MCP servers
run with the user's permissions. Header and environment values are hidden in the approval preview; protect the config
file and do not share it if it contains credentials. MCP supports stdio and Streamable HTTP with static headers;
remote HTTP endpoints require HTTPS, except for loopback addresses. OAuth, legacy HTTP+SSE, and background server
notifications are not implemented. Sessions and approvals are in memory, expire, and are not restored on page reload.
Local tool endpoints require a same-origin loopback request and any configured API key, but are not a security boundary
against other processes running as the same user.

Calls and results appear in the conversation, and the model decides whether to call them. Tool-enabled generation is
buffered until the server can identify tool calls. Message metrics report the engine's prefill time and decode throughput;
the OpenAI-compatible response exposes the underlying values in the `mnn_timings` extension.

On macOS 13 or later, build a local desktop app and DMG on the target architecture:

```bash
bash apps/mnncli/macos/package.sh
```

The image is written to `apps/mnncli/dist/` and includes the server, but no model weights. Install the app and select
an exported MNN model's `config.json`; an embedding model can be selected separately from the toolbar. The app runs its
own authenticated server on `127.0.0.1:18957` and stores logs under `~/Library/Application Support/MNN Studio/logs`.
This local build uses an ad-hoc signature and is not notarized for distribution.

For a model-backed API check, run `test/test_anthropic_api_contract.sh --model MODEL_NAME`, optionally adding
`--config /path/to/config.json`. This checks text chat and authentication as well as the Anthropic event sequence.
With servers already running, `test/test_api_features.py --model CHAT_ID` can also check sampling; pass `--tools`,
`--embedding-model ID`, and `--vision-url URL --vision-model ID` to exercise those optional capabilities. The tools
check requires a model that reliably calls the test function, while `--check-vision-colors` additionally checks that
the vision model distinguishes red and blue test images.
Run `python3 test/test_agent_security.py --config /path/to/test-model/config.json` to exercise tool authorization,
workspace boundaries, commands and local MCP fixtures. `node test/test_web_tools.js` tests browser tool dispatch and
approval handling without launching a browser. `node test/test_web_live_command.js URL MODEL` exercises the Studio
streaming loop against a running server; it automatically approves only a harmless printf command in a temporary workspace.
## Dependencies

- OpenSSL (for HTTPS support)
- MNN core library
- LLM engine library

## Notes

- The tool requires macOS 13.0+ when building on Apple platforms
- On Linux, ensure `libssl-dev` (or equivalent) is installed
- Models are cached in the user's cache directory
- `mnncli serve` currently provides OpenAI-compatible and Anthropic-compatible API routes
