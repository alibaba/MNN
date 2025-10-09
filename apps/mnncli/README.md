# MNNCLI

MNNCLI is a command-line interface tool for MNN (Mobile Neural Network) that provides various functionalities for working with LLM models.

## Features

- **Model Management**: List, download, and delete models
- **Model Serving**: Start a web server to serve models via HTTP API
- **Model Execution**: Run models with prompts or prompt files
- **Benchmarking**: Performance benchmarking for models
- **Model Search**: Search for models in the Hugging Face repository

## Building

To build MNNCLI, run the following commands from the project's root directory:

```bash
cmake -B build -DBUILD_MNNCLI=ON
cmake --build build
```

The executable will be located at `build/mnncli`.

## Usage

### List Models
```bash
./build/mnncli list
```

### Serve Model
```bash
./build/mnncli serve <model_name>
```

### Run Model
```bash
./build/mnncli run <model_name> [-c config_path] [-p prompt] [-f prompt_file]
```

### Benchmark Model
```bash
./build/mnncli benchmark <model_name> [-c config_path]
```

### Download Model
```bash
./build/mnncli download <model_name> <repo_name>
```

### Search Models
```bash
./build/mnncli search <keyword>
```

### Delete Model
```bash
./build/mnncli delete <model_name>
```

## Dependencies

- OpenSSL (for HTTPS support)
- MNN core library
- LLM engine library

## Notes

- The tool requires macOS 13.0+ when building on Apple platforms
- Models are cached in the user's cache directory
- The web server provides an OpenAI-compatible API interface
