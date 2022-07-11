import tvm
from tvm import relay, autotvm
from tvm import rpc, relay
from tvm.contrib.download import download_testdata
from tvm.relay.expr_functor import ExprMutator
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.quantize.quantize import prerequisite_optimize
from tvm.contrib import utils, xcode, graph_runtime, coreml_runtime
from tvm.contrib.target import coreml as _coreml
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import os
import re
import sys
import time
import numpy as np
from PIL import Image

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_IOS_RPC_PROXY_HOST"]
# Set your desination via env variable.
# Should in format "platform=iOS,id=<the test device uuid>"
destination = os.environ["TVM_IOS_RPC_DESTINATION"]

if not re.match(r"^platform=.*,id=.*$", destination):
    print("Bad format: {}".format(destination))
    print("Example of expected string: platform=iOS,id=1234567890abcabcabcabc1234567890abcabcab")
    sys.exit(1)

proxy_port = 9090
key = "iphone"

# Change target configuration, this is setting for iphone6s
# arch = "x86_64"
# sdk = "iphonesimulator"
arch = "arm64"
sdk = "iphoneos"
target_host = "llvm -mtriple=%s-apple-darwin" % arch
input_name = 'input'

# override metal compiler to compile to iphone
@tvm.register_func("tvm_callback_metal_compile")
def compile_metal(src):
    return xcode.compile_metal(src, sdk=sdk)


def prepare_input():
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_name = "cat.png"
    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    img_path = download_testdata(img_url, "cat.png", module="data")
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synset = eval(f.read())
        image = Image.open(img_path).resize((224, 224))

    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image.astype("float32"), synset


def get_model(model_name, data_shape):
    import onnx
    onnx_model = onnx.load(model_name)
    input_name1 = "unique_ids_raw_output___9:0"
    input_name2 = "segment_ids:0"
    input_name3 = "input_mask:0"
    input_name4 = "input_ids:0"
    input_shape1 = [1]
    input_shape2 = [1, 256]
    shape_dict = { input_name1 : input_shape1, input_name2 : input_shape2, input_name3 : input_shape2, input_name4 : input_shape2 }
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    # we want a probability so add a softmax operator
    func = mod["main"]
    '''
    func = relay.Function(
        func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs
    )
    '''

    return func, params


network = 'resnet18'
log_file = "%s.log" % (network)
host = '192.168.31.138'
port = 9090
device_key = 'iphone'

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 30,
    "early_stopping": 30,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder("default"),
        runner=autotvm.RPCRunner(
            device_key,
            host=host,
            port=port,
            number=5,
            timeout=10,
        ),
    ),
}

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # process tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def inference(model_name):
    temp = utils.tempdir()
    image, synset = prepare_input()
    model, params = get_model(model_name, image.shape)

    def run(mod, target):
        import time
        t1 = time.time()
        with relay.build_config(opt_level=3):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)
        t2 = time.time()
        print("Compile time: %.3f sec" % (t2 - t1))
        # path_dso = temp.relpath("deploy.dylib")
        path_dso = '/Users/wangzhaode/tvm/apps/ios_rpc/deploy.dylib'
        lib.export_library(path_dso, xcode.create_dylib, arch=arch, sdk=sdk)
        xcode.codesign(path_dso)

        # Start RPC test server that contains the compiled library.
        xcode.popen_test_rpc(proxy_host, proxy_port, key, destination=destination, libs=[path_dso])

        # connect to the proxy
        remote = rpc.connect(proxy_host, proxy_port, key=key)

        if target == "metal":
            ctx = remote.metal(0)
        else:
            ctx = remote.cpu(0)
        lib = remote.load_module("deploy.dylib")
        m = graph_runtime.GraphModule(lib["default"](ctx))

        input_name1 = "unique_ids_raw_output___9:0"
        input_name2 = "segment_ids:0"
        input_name3 = "input_mask:0"
        input_name4 = "input_ids:0"
        dtype = 'int64'
        input_shape1 = [1]
        input_shape2 = [1, 256]
        data_1 = tvm.nd.array((np.random.uniform(size=input_shape1)).astype(dtype))
        data_2 = tvm.nd.array((np.random.uniform(size=input_shape2)).astype(dtype))
        m.set_input(input_name1, data_1)
        m.set_input(input_name2, data_2)
        m.set_input(input_name3, data_2)
        m.set_input(input_name4, data_2)
        '''
        m.run()
        tvm_output = m.get_output(0)
        top1 = np.argmax(tvm_output.asnumpy()[0])
        print("TVM prediction top-1:", top1, synset[top1])
        '''

        # evaluate
        ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=20)
        prof_res = np.array(ftimer().results) * 1000
        print("%-19s (%s)" % ("%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

    def annotate(func, compiler):
        """
        An annotator for Core ML.
        """
        # Bind free variables to the constant values.
        bind_dict = {}
        for arg in func.params:
            name = arg.name_hint
            if name in params:
                bind_dict[arg] = relay.const(params[name])

        func = relay.bind(func, bind_dict)

        # Annotate the entire graph for Core ML
        mod = tvm.IRModule()
        mod["main"] = func

        seq = tvm.transform.Sequential(
            [
                transform.SimplifyInference(),
                transform.FoldConstant(),
                transform.FoldScaleAxis(),
                transform.AnnotateTarget(compiler),
                transform.MergeCompilerRegions(),
                transform.PartitionGraph(),
            ]
        )

        with relay.build_config(opt_level=3):
            mod = seq(mod)

        return mod

    # CPU
    run(model, target_host)

if __name__ == "__main__":
    inference(sys.argv[1])
