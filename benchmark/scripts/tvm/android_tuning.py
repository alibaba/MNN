import os
import tvm
import time
import numpy as np
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    shape_dict = { input_name : input_shape }
    import onnx
    onnx_model = onnx.load('../onnx/' + name + '.onnx')
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params, input_shape, output_shape

target = "llvm -mtriple=arm64-linux-android"
device_key = "android"
use_android = True

network = "mobilenetv2-7-modify"
input_name = "input"
log_file = "%s.%s.log" % (device_key, network)
dtype = "float32"
host = "30.206.32.132"
port = 9090
repeat = 100

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 1500,
    "early_stopping": 800,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
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

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    mod, params, input_shape, _ = get_network(network, batch_size=1)
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(
        # mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense"))
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"), )
    )
    print("# END Extract tasks.")

    # run tuning tasks
    print("Tuning...")
    t1 = time.time()
    tune_tasks(tasks, **tuning_opt)
    t2 = time.time()
    print("### tuning time is : %f s " % (t2 - t1))

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        t1 = time.time()
        with tvm.transform.PassContext(opt_level=3):
            #lib = relay.build_module.build(mod, target=target, params=params)
            lib = relay.build(mod, target=target, target_host=None, params=params)
        t2 = time.time()
        print("### compile time is : %f s " %  (t2 - t1))
        print("Export...")
        # export library
        tmp = tempdir()
        if use_android:
            from tvm.contrib import ndk
            filename = "net.so"
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

        # upload
        print("uploading...")
        tracker = tvm.rpc.connect_tracker(host, port)
        remote = tracker.request(device_key)
        ctx = remote.context(str(target), 0)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)
        module = runtime.GraphModule(rlib["default"](ctx))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))

        # evaluate
        print("evaluating...")
        module.set_input(input_name, data_tvm)
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=repeat)
        prof_res = np.array(ftimer().results) * 1000
        print(
            "avg time: %-19s (%s)" % ("%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
        )

if __name__ == '__main__':
    tune_and_evaluate(tuning_option)
