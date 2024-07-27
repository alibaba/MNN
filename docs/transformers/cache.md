## StateCacheManager

### Compilation

LLM and TRANSFORMER_FUSE shall be set ON: `-DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON`

```bash
cmake .. -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_PORTABLE_BUILD=ON -DTFMODEL_OPTIMIZE=ON -DMNN_LOW_MEMORY=ON -DMNN_AVX512=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_TEST=ON
make -j20
```

Transformer path: 
Download from https://github.com/wangzhaode/mnn-llm/releases
`/home/hzx/Desktop/ANL/Project/LLM/MNN-LLM/model/qwen1_5-4b-chat-mnn/`
`./llm_demo /home/hzx/Desktop/ANL/Project/LLM/MNN-LLM/model/qwen1_5-4b-chat-mnn/`

export
```bash
cd transformers/llm/export/
python llm_export.py \
        --path ../../../model/Qwen1_5-4B-Chat \
        --type Qwen1_5-4B-Chat \
        --export_split \
        --export_token \
        --export_mnn \
        --export_embed --embed_bin --embed_bf16 \
        --onnx_path ../../../model/qwen1_5-4b-chat-onnx2 \
        --mnn_path  ../../../model/qwen1_5-4b-chat-mnn2
```

Currently fuse attention haven't added to python MNN package converter. To use it, you need directly use MNNConvert.
```bash
for i in $(seq 0 39)
do
    ./build/MNNConvert -f ONNX --modelFile ./model/qwen1_5-4b-chat-onnx/block_${i}.onnx --MNNModel ./model/qwen1_5-4b-chat-mnn-ff/block_${i}.mnn --weightQuantBits 4 --weightQuantAsymmetric --transformerFuse
done
```
