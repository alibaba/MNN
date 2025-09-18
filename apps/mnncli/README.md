# MNNCLI

MNNCLI is a command-line interface tool for MNN (Mobile Neural Network) that provides various functionalities for working with LLM models.

## Features

- **Model Management**: List, download, and delete models
- **Model Serving**: Start a web server to serve models via HTTP API
- **Model Execution**: Run models with prompts or prompt files
- **Benchmarking**: Performance benchmarking for models
- **Model Search**: Search for models in the Hugging Face repository

## Building

To build MNNCLI, set the `BUILD_MNNCLI` option to `ON` when configuring CMake:

```bash
cmake -DBUILD_MNNCLI=ON ..
make
```

## Usage

### List Models
```bash
./mnncli list
```

### Serve Model
```bash
./mnncli serve <model_name>
```

### Run Model
```bash
./mnncli run <model_name> [-c config_path] [-p prompt] [-f prompt_file]
```

### Benchmark Model
```bash
./mnncli benchmark <model_name> [-c config_path]
```

### Download Model
```bash
./mnncli download <model_name> <repo_name>
```

### Search Models
```bash
./mnncli search <keyword>
```

### Delete Model
```bash
./mnncli delete <model_name>
```

## Dependencies

- OpenSSL (for HTTPS support)
- MNN core library
- LLM engine library

## Notes

- The tool requires macOS 13.0+ when building on Apple platforms
- Models are cached in the user's cache directory
- The web server provides an OpenAI-compatible API interface
