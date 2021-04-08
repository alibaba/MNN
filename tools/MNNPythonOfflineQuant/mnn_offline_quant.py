from __future__ import print_function
import time
import argparse
import numpy as np
import tqdm
import MNN
import yaml
from calibration_dataset import calibration_dataset
from test_dataset import ImagenetDataset

nn = MNN.nn
F = MNN.expr


def get_mnn_format(format_str):
    fmt = str.lower(format_str)
    if fmt == 'nchw':
        return F.NCHW
    elif fmt == 'nhwc':
        return F.NHWC
    elif fmt == 'nc4hw4':
        return F.NC4HW4
    else:
        raise ValueError("unknown format:", format_str)

def quant_func(net, dataloader, opt):
    net.train(True)
    dataloader.reset()

    t0 = time.time()
    for i in tqdm.trange(dataloader.iter_number):
        example = dataloader.next()
        input_data = example[0]
        predicts = net.forward(input_data)
        # fake update
        opt.step(F.const(0.0, []))
        for predict in predicts:
            predict.read()

    t1 = time.time()
    cost = t1 - t0
    print("Epoch cost: %.3f s." % cost)


def main():
    '''
    offline quantization using MNN python api.

    1. you need to convert your model to mnn model

    2. you need to provide a calibration dataset by modifying preprocessing steps in
    'calibration_dataset.py' to suit your case.

    3. you need to provide a config yaml file in which provide input and output information about your model.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--mnn_model", type=str, required=True,\
        help="original float MNN model file")
    parser.add_argument("--quant_model", type=str, required=True, \
        help="name of quantized model to save")
    parser.add_argument("--batch_size", type=int, required=False, default=32,\
                        help="calibration batch size")

    args = parser.parse_args()

    mnn_model = args.mnn_model
    quant_model = args.quant_model
    batch_size = args.batch_size

    dataloader = MNN.data.DataLoader(calibration_dataset, batch_size=batch_size, shuffle=True)

    m = F.load_as_dict(mnn_model)

    inputs_outputs = F.get_inputs_and_outputs(m)
    for key in inputs_outputs[0].keys():
        print('input names:\t', key)
    for key in inputs_outputs[1].keys():
        print('output names:\t', key)
    
    config_file = "config.yaml"
    f = open(config_file)
    config = yaml.load(f)

    # get inputs and outputs
    inputs = []
    for name in config['inputs']['names']:
        inputs.append(m[name])
    
    outputs = []
    for name in config['output_names']:
        outputs.append(m[name])
    
    input_placeholders = []
    for i in range(len(inputs)):
        shape = config['inputs']['shapes'][i]
        fmt = config['inputs']['formats'][i]
        nnn_format = get_mnn_format(fmt)
        input_placeholders.append(F.placeholder(shape, nnn_format))

    net = nn.load_module(inputs, outputs, True)

    # no use optimizer
    opt = MNN.optim.SGD(net, 0.01, 0.9, 0.0005)

    nn.compress.train_quant(net, quant_bits=8)

    quant_func(net, dataloader, opt)

    # save model
    net.train(False)
    predicts = net.forward(input_placeholders)
    print("quantized model save to " + quant_model)
    F.save(predicts, quant_model)


if __name__ == "__main__":
    main()
