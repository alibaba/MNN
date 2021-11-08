拷贝AliNNModel所有模型和测试数据到MNN工作台工程下
python scripts/pullTestModel.py --alinnmodel_path ../../../AliNNModel --playground_path playground/playground
拷贝AliNNModel指定模型（mobilenet, Ranfa）和测试数据到MNN工作台工程下
python scripts/pullTestModel.py --alinnmodel_path ../../../AliNNModel --playground_path playground/playground --models mobilenet Ranfa
