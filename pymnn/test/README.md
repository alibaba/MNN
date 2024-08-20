# Pymnn Test Cases

# 1. Unit Test
```bash
python3 unit_test.py
```

# 2. Model Test
```bash
python3 model_test.py ~/AliNNModel
```

# 3. Train Test
```bash
./train_test.sh
```

# 4. Quant Test
```bash
python3 ../examples/MNNQuant/test_mnn_offline_quant.py \
        --mnn_model your_model.mnn \
        --quant_imgs <path_to_imgs> \
        --quant_model ./quant_model.mnn
```

# 5. Benchmark MNN.numpy
```bash
pip install prettytable
python3 benchmark.py
```

# 6. Playgroud Test (just internal usage)
```bash
# 拷贝AliNNModel所有模型和测试数据到MNN工作台工程下
python scripts/pullTestModel.py --alinnmodel_path ../../../AliNNModel --playground_path playground/playground
# 拷贝AliNNModel指定模型（mobilenet, Ranfa）和测试数据到MNN工作台工程下
python scripts/pullTestModel.py --alinnmodel_path ../../../AliNNModel --playground_path playground/playground --models mobilenet Ranfa
```