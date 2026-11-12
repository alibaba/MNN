# LFM3 text and image inference

Supported checkpoints are `LiquidAI/LFM3-1.7B`, `LiquidAI/LFM3-1.7B-Base`
and `LiquidAI/LFM3-VL-2B`. The Base checkpoint is intended for text completion;
the other two support chat. This integration supports still images, including
multiple images and dynamic image sizes. LFM3 video and audio are not included.

Use the official model configuration, tokenizer and weights. Install the export
dependencies in `transformers/llm/export/requirements.txt` and a Transformers
version providing `lfm3` and `lfm3_vl` (validated with Transformers 5.22.2).
Check the checkpoint's model card for its license and usage conditions; MNN's
license does not replace the model license. Record the downloaded revision when
comparing results.

## Export

Build `MNNConvert` with `MNN_BUILD_CONVERTER=ON`. From the repository root:

```bash
python transformers/llm/export/llmexport.py \
    --path /path/to/LFM3-VL-2B --export mnn \
    --dst_path /path/to/LFM3-VL-2B-MNN \
    --mnnconvert /path/to/build/MNNConvert \
    --quant_bit 4 --visual_quant_bit 4
```

For the text checkpoints, change `--path` and omit `--visual_quant_bit`.
Use `--quant_bit 8` for Q8; for FP16 graph weights use `--quant_bit 16
--disable_transformer_c4` (and `--visual_quant_bit 16` for VL).
These are weight formats: the existing external embedding path uses BF16,
and normalization, constants and intermediate computations retain their own
types. Quantization can change generation; compare the same prompts against
the official checkpoint.

## Inference and verification

Build with `MNN_BUILD_LLM=ON`, `MNN_LOW_MEMORY=ON`, and
`MNN_SUPPORT_TRANSFORMER_FUSE=ON`. Images additionally require
`MNN_BUILD_OPENCV=ON` (MNN's CV library). Enable `MNN_CUDA=ON` for CUDA.
Set `backend_type` to `cpu` or `cuda` in a configuration file and use
`memory: "normal"` for the validated CUDA configuration.

```bash
printf 'What is the capital of France?\n' > prompt.txt
./build/llm_demo /path/to/model/config.json prompt.txt
printf '<img>/path/to/cat.jpg</img>Describe the image.\n' > image.txt
./build/llm_demo /path/to/LFM3-VL-2B-MNN/config.json image.txt
```

Use raw completion input for Base rather than imposing an instruction template.
For chat, the official Jinja template accepts `enable_thinking: false` in
`jinja.context` to request direct replies.

The small-image resize policy is specific to LFM3; other vision families retain
their existing constraints. A still image is repeated to fill the vision
encoder's temporal patch. This is part of image encoding, not video support.

Model-dependent image checks are a manual integration target:

```bash
cmake -S . -B build -DMNN_BUILD_LLM=ON -DMNN_BUILD_OPENCV=ON \
    -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
    -DMNN_BUILD_TEST=ON -DMNN_LLM_BUILD_TEST=ON
cmake --build build -j
./build/test_lfm3_image /path/to/LFM3-VL-2B-MNN/config.json
./build/run_test.out llm/jinja_dictionary_get
./build/run_test.out expr/runtime_lifetime
./build/run_test.out expr/module_output_lifetime
python -m unittest discover -s transformers/llm/export/tests -p 'test_head_gate.py'
python -m unittest discover -s transformers/llm/export/tests -p 'test_external_const.py'
python -m unittest discover -s transformers/llm/export/tests -p 'test_lfm3_media_markers.py'
```

The image target checks small and non-square images, finite embeddings,
red/blue image understanding, multiple images, and rejected unsupported input.
It is not automatically run by `test.sh`; provide local model weights.
Desktop CPU/CUDA validation does not establish mobile, Metal or Vulkan support.
