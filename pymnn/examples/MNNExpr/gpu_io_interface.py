import numpy as np
import MNN
import sys
import torch
import time

def inference():
    """ inference mobilenet_v1 using a specific picture """

    config = {}
    config['precision'] = 'low'
    config['backend'] = 2
    config['numThread'] = 4

    rt = MNN.nn.create_runtime_manager((config,))
    rt.set_cache(".cachefile")

    net = MNN.nn.load_module_from_file(sys.argv[1], ["data"], ["prob"], runtime_manager=rt)

    input_var = MNN.expr.placeholder([1, 3, 224, 224], MNN.expr.NCHW)
    image = np.loadtxt(sys.argv[2])
    image = image.astype(np.float32)

    torch_tensor = torch.from_numpy(image)
    torch_tensor = torch_tensor.cuda().type(torch.float16)
    input_var.set_device_ptr(torch_tensor.data_ptr(), 2)

    #inference
    output_var = net.forward([input_var])
    output_var = output_var[0]
    output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)

    # output_numpy= output_var.read()
    out_torch = torch.empty([1, 1000], dtype=torch.float16).cuda()
    output_var.copy_to_device_ptr(out_torch.data_ptr(), 2)
    output_numpy = out_torch.type(torch.float32).cpu().numpy()

    ref_out = np.loadtxt(sys.argv[3]).astype(np.float32)
    close = np.allclose(output_numpy.flatten(), ref_out, atol=0.03)

    print("USE GPU IO data, verify equal:", close)

if __name__ == "__main__":
    # python gpu_interface.py **/mobilenet/temp.bin **/mobilenet/input_0.txt **/mobilenet/output.txt
    inference()
