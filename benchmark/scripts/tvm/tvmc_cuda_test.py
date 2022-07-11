import time
from tvm.driver import tvmc

model_name = "bertsquad-10" # resnet50-v1-7 mobilenetv2-7  resnet50-v1-7  shufflenet-v2-10 squeezenet1.1-7  bertsquad-10
tune_number = 500
device_name = "cuda"
log_file = model_name + "-autotuner_records_" + str(tune_number) + ".json"
compile_file = model_name + "-tvm_autotuned_" + str(tune_number) + ".tar"


# for mobilenetv2-7.onnx, please set shape_dict={'input' : [1, 3, 224, 224]} when tvmc.load
# for bertsquad-10.onnx, please set shape_dict={'segment_ids:0' : [1, 256], 'input_mask:0' : [1, 256], 'input_ids:0' : [1, 256], 'unique_ids_raw_output___9:0' : [1]} when tvmc.load

model = tvmc.load("/home/mnnteam/benchmark/models/onnx/" + model_name + ".onnx") #Step 1: Load


start_tune = time.time()
tvmc.tune(model, target=device_name, trials=tune_number, tuning_records=log_file)
end_tune = time.time()
print('tune time cost', end_tune-start_tune, 's')


start_compile = time.time()
tvmc.compile(model, target=device_name, tuning_records=log_file, package_path=compile_file) #Step 2: Compile
end_compile = time.time()
print('compile time cost', end_compile-start_compile, 's')

new_package = tvmc.TVMCPackage(package_path=compile_file)

start_run = time.time()
result = tvmc.run(new_package, device=device_name, repeat=100) #Step 3: Run
end_run = time.time()
print('run time cost', end_run-start_run, 's')
