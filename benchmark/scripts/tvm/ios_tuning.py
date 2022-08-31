import os
import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime
from tvm.contrib import xcode

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    import onnx
    onnx_model = onnx.load('../../onnx/' + name + '.onnx')
    input_name = 'data'
    if 'mobilenet' in name or 'shufflenet' in name:
        input_name = 'input'
    shape_dict = { input_name : input_shape }
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params, input_shape, input_name

proxy_host = os.environ["TVM_IOS_RPC_PROXY_HOST"]
host_ip = proxy_host
destination = os.environ["TVM_IOS_RPC_DESTINATION"]

device_key = 'iphone'
proxy_port = 9090

arch = "arm64"
sdk = "iphoneos"
target = "llvm -mtriple=%s-apple-darwin" % arch
target_host = "llvm -mtriple=%s-apple-darwin" % arch

# change this name to test-model-name
network = 'resnet50-v1-7'
log_file = "%s.%s.log" % (device_key, network)
dtype = 'float32'

autotvm.measure.measure_methods.check_remote = lambda *args: True

def fcompile(*args):
    xcode.create_dylib(*args, arch=arch, sdk=sdk)
    path = args[0]
    xcode.codesign(path)
    xcode.popen_test_rpc(proxy_host, proxy_port, device_key, destination=destination, libs=[path])

fcompile.output_format = "dylib"


tuning_option = {
    'log_filename': log_file,
    'tuner': 'random',
    'early_stopping': None,
    'n_trial': 30,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            n_parallel=1,
            build_func=fcompile,
            timeout=60
        ),
        runner=autotvm.RPCRunner(
            device_key, host=host_ip, port=9190, priority=9999,
            number=1, repeat=1, timeout=60, min_repeat_ms=150, enable_cpu_cache_flush=True)
    ),
}

def tune_tasks(tasks,
               measure_option,
               tuner='random',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, input_name = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    import time
    # run tuning tasks
    print("Tuning...")
    t1 = time.time()
    tune_tasks(tasks, **tuning_opt)
    t2 = time.time()
    print("Tuning time: %.3f sec" % (t2 - t1))

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        t1 = time.time()
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)
        t2 = time.time()
        print("Compile time: %.3f sec" % (t2 - t1))
        # export library
        path_dso = "tuned_deploy.dylib"
        lib.export_library(path_dso, xcode.create_dylib, arch=arch, sdk=sdk)
        xcode.codesign(path_dso)

        # Evaluate inference cost on tuned lib
        xcode.popen_test_rpc(proxy_host, proxy_port, device_key, destination=destination, libs=[path_dso])

        remote = autotvm.measure.request_remote(device_key, host_ip, 9190,
                                                timeout=10000)

        # Upload not needed for ios because dylib is built into app
        # remote.upload(path_dso)

        rlib = remote.load_module(path_dso)

        ctx = remote.cpu(0)
        
        module = runtime.create(graph, rlib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input(input_name, data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=20)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

if __name__ == '__main__':
    if os.path.exists("rpc_config.txt"):
        os.remove("rpc_config.txt")
    tune_and_evaluate(tuning_option)
