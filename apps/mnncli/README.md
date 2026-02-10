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

## Dependencies

- OpenSSL (for HTTPS support)
- MNN core library
- LLM engine library

## Notes

- The tool requires macOS 13.0+ when building on Apple platforms
- On Linux, ensure `libssl-dev` (or equivalent) is installed
- Models are cached in the user's cache directory
- The web server provides an OpenAI-compatible API interface
