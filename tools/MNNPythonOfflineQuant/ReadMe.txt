这是用MNN的python接口改造的离线量化工具，适用于如下情况：
    1. 你的模型无法使用MNN离线量化工具tools/quantization进行量化，例如多输入，数据预处理比较复杂
    2. 你的模型无法使用MNN进行训练量化，受限于MNN的训练能力

为了使用这个工具，你需要提供：
    0. 使用 MNNConvert工具加上 --forTraining 将你的模型转换成MNN模型 (这步主要是为了保留模型中的BatchNorm，因此你保存pb或者onnx时不要做BatchNorm融合)
    1. 一个 calibration_dataset.py 文件，里面包含了你的校准数据集的定义
    2. 一个 config.yaml 文件，里面包含了你模型的输入输出的相关信息

可以参考提供的 calibration_dataset.py 和 config.yaml 来实现

特别注意校准集中返回输入数据的顺序和config文件中输入的顺序应该是对应的

使用方法（batch size可以根据你的模型调整）：
    python mnn_offline_quant.py --mnn_model origin_float_model.mnn --quant_model quant_model.mnn --batch_size 32

使用建议：
    1. 如果你的模型中卷积的激活是prelu的话，使用relu/relu6代替prelu可能会取得更好的量化精度和推理速度，这可能需要重新训练模型
    2. 如果模型的输入无法固定，请将batch size设置为1，并且calibration_dataset的返回值也使用实际输入值的形状


############################################################################


This is a python version of MNN offline quant tool, use this tool when:
    1. you can not use MNN offline quant tool (tools/quantization) to quantize your model, cases like multi-input, complecated preprocessing
    2. you can not use MNN's quant-aware-training (QAT) tool to quantize your model, because of MNN's limited training features

in order to use this tool, you need to provide:
    0. use --forTraining flag of MNNConvert to convert your model to MNN (this is mainly for preserving BatchNorm, 
        so you should NOT fuse BatchNorm when you save pb or onnx model files)
    1. a calibration_dataset.py file, in which you define your calibration dataset
    2. a config.yaml file, in which you provide information of inputs and outputs of your model

you can refer to the example file to write your own.

please Note, the order of returned input data in your calibration dataset should be aligned with the order of input your provide in your config.yaml file.

usage of the tool (you can adjust batch size according to your own model):
    python mnn_offline_quant.py --mnn_model origin_float_model.mnn --quant_model quant_model.mnn --batch_size 32

usage tips:
    1. if the activation function of conv is prelu in your model, use relu/relu6 instead of prelu may improve precision and inference speed of quantized model. re-training may be required.
    2. if the input shape can not be fixed, your should set batch_size=1, and the shape of returned values of calibration_dataset should be actual input's shape.
