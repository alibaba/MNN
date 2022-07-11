import sys

from cv2 import repeat
import tvm
import onnx
import time
import numpy as np
from tvm import relay, autotvm
from tvm.contrib import ndk
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime

def print_progress(msg):
    """print progress message

    Parameters
    ----------
    msg: str
        The message to print
    """
    sys.stdout.write(msg + "\r")
    sys.stdout.flush()

# host
host = '30.206.32.132'
port = 9090
key = 'android'
# arch
arch = "arm64"
target = "llvm -mtriple=%s-linux-android" % arch
target_host = None
# evaluate
repeat = 3

if __name__ == '__main__':
    model = sys.argv[1]
    input_name1 = "unique_ids_raw_output___9:0"
    input_name2 = "segment_ids:0"
    input_name3 = "input_mask:0"
    input_name4 = "input_ids:0"
    dtype = "int64"
    input_shape1 = [1]
    input_shape2 = [1, 256]
    shape_dict = { input_name1 : input_shape1, input_name2 : input_shape2, input_name3 : input_shape2, input_name4 : input_shape2 }
    # load onnx model
    onnx_model = onnx.load(model)
    # relay load
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    # relay build
    if len(sys.argv) > 2:
        # log_file = "android.mobilenetv2-7-modify.log"
        log_file = sys.argv[2]
        with autotvm.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, target_host=target_host, params=params)
            # save file
            tmp = tempdir()
            filename = "net.so"
            lib.export_library(tmp.relpath(filename), ndk.create_shared)

            # upload
            print_progress("uploading...")
            tracker = tvm.rpc.connect_tracker(host, port)
            remote = tracker.request(key)
            ctx = remote.context(str(target), 0)
            remote.upload(tmp.relpath(filename))
            rlib = remote.load_module(filename)
            module = runtime.GraphModule(rlib["default"](ctx))
            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))

            # evaluate
            print_progress("evaluating...")
            module.set_input(input_name, data_tvm)
            ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=repeat)
            prof_res = np.array(ftimer().results) * 1000
            print(
                "avg time: %-19s (%s)" % ("%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
            )
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)
        # save file
        tmp = tempdir()
        filename = "net.so"
        #lib.export_library(tmp.relpath(filename), ndk.create_shared)
        lib.export_library(filename, ndk.create_shared)

        # upload
        tracker = tvm.rpc.connect_tracker(host, port)
        remote = tracker.request(key)
        ctx = remote.context(str(target), 0)
        print("uploading...")
        # remote.upload(tmp.relpath(filename))
        remote.upload(filename)
        print("upload done")
        rlib = remote.load_module(filename)
        module = runtime.GraphModule(rlib["default"](ctx))
        data_1 = tvm.nd.array((np.random.uniform(size=input_shape1)).astype(dtype))
        data_2 = tvm.nd.array((np.random.uniform(size=input_shape2)).astype(dtype))

        # evaluate
        print("evaluating...")
        module.set_input(input_name1, data_1)
        module.set_input(input_name2, data_2)
        module.set_input(input_name3, data_2)
        module.set_input(input_name4, data_2)
        t1 = time.time() 
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=repeat)
        t2 = time.time() 
        print('evaluator time : {} ms'.format(1000 * (t2 - t1))) 
        prof_res = np.array(ftimer().results) * 1000
        print(
            "avg time: %-19s (%s)" % ("%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
        )
