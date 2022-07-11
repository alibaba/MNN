import argparse
parser = argparse.ArgumentParser(description='convert onnx -> mnn/pb/tflite/pt/ptl')
parser.add_argument('--modeldir', help='model dir')
parser.add_argument('--debug', default=False, action='store_true', help='preverse torch code cache')
args = parser.parse_args()

import os
import shutil
from os.path import join, exists
import onnx
ONNX_DIR = join(args.modeldir, 'onnx')
MNN_DIR = join(args.modeldir, 'mnn')
TF_DIR = join(args.modeldir, 'pb')
TFLITE_DIR = join(args.modeldir, 'tflite')
TFLITE_FP16_DIR = join(args.modeldir, 'tflite', 'fp16')
TORCH_DIR = join(args.modeldir, 'torch')
TORCH_MOBILE_DIR = join(args.modeldir, 'torch_lite')
TORCH_CACHE_DIR = join(args.modeldir, 'torch_code_cache')

for dirpath in [MNN_DIR, TF_DIR, TFLITE_DIR, TFLITE_FP16_DIR, TORCH_DIR, TORCH_MOBILE_DIR, TORCH_CACHE_DIR]:
  if exists(dirpath):
    shutil.rmtree(dirpath, ignore_errors=True)
  os.makedirs(dirpath)

def convert_mnn(onnx_name):
    from subprocess import Popen, PIPE, STDOUT
    cvt_exe = join(os.getcwd(), 'MNN', 'build_converter', 'MNNConvert')
    onnx_path = join(ONNX_DIR, onnx_name)
    mnn_path = join(MNN_DIR, onnx_name.replace('.onnx', '.mnn'))
    process = Popen([cvt_exe, '-f', 'ONNX', '--modelFile', onnx_path, '--MNNModel', mnn_path, '--bizCode', 'MNN'], stdout=PIPE, stderr=STDOUT, text=True)
    cmd_out, _ = process.communicate()
    if not exists(mnn_path):
        print(cmd_out)
        exit(1)

import torch
def convert_pt(onnx_name):
  import importlib
  from onnx_pytorch import code_gen
  code_path = join(TORCH_CACHE_DIR, onnx_name.replace('.onnx', '.pt.code').replace('.', '_').replace('-', '_'))
  os.makedirs(code_path)
  onnx_model = onnx.load(join(ONNX_DIR, onnx_name))
  onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
  print(f">>> {onnx_name} => {code_path} <<<")
  code_gen.gen(onnx_model, code_path, shape_infer=False, overwrite=True)
  model = importlib.import_module(code_path.replace(os.sep, '.') + '.model').Model()
  model.eval()
  # torch.quantization.fuse_modules need specify layer to fuse manually, which isn't scalable and convenient
  # torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)
  from torch.utils.mobile_optimizer import optimize_for_mobile
  script_module = torch.jit.script(model)
  print(f">>> optimize CPU mobile {onnx_name} <<<")
  opt_module = optimize_for_mobile(script_module)
  opt_module.save(join(TORCH_DIR, onnx_name.replace('.onnx', '.pt')))
  opt_module._save_for_lite_interpreter(join(TORCH_MOBILE_DIR, onnx_name.replace('.onnx', '.ptl')))
  print(f">>> optimize Metal mobile {onnx_name} <<<")
  opt_module = optimize_for_mobile(script_module, backend='metal')
  opt_module._save_for_lite_interpreter(join(TORCH_MOBILE_DIR, onnx_name.replace('.onnx', '_metal.ptl')))

def convert_tflite(onnx_name):
  import json
  import tensorflow as tf
  from onnx_tf.backend import prepare
  pb_path = join(TF_DIR, onnx_name.replace('.onnx', '.pb'))
  print(f">>> {onnx_name} => {pb_path} <<<")
  pb_model = prepare(onnx.load(join(ONNX_DIR, onnx_name)))
  
  # replace None with 1, avoid dynamic size bug on GPU
  for name in pb_model.inputs:
    spec = pb_model.signatures[name]
    if None not in spec.shape.as_list():
      continue
    shape = list(map(lambda s: s or 1, spec.shape.as_list()))
    pb_model.signatures[name] = tf.TensorSpec(shape, dtype=spec.dtype, name=spec.name)
  
  pb_model.export_graph(pb_path)
  converter = tf.lite.TFLiteConverter.from_saved_model(pb_path)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  
  # Cast / RealDiv in bert model is not builtin op of tflite, we add tf.lite.OpsSet.SELECT_TF_OPS then model convert success
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
  converter.target_spec.supported_types = [tf.float32]
  lite_model = converter.convert()

  tflite_path = join(TFLITE_DIR, onnx_name.replace('.onnx', '.tflite'))
  print(f">>> {onnx_name} => {tflite_path} <<<")
  with open(tflite_path, 'wb') as f:
    f.write(lite_model)

  # save pb/tflite metadata used by bench
  def save_meta(pb_model, lite_model):
    from tensorflow.lite.tools.visualize import CreateDictFromFlatbuffer, NameListToString
    pb_t_map = {"serving_default_" + pb_model.signatures[s].name + ":0" : s for s in pb_model.inputs}
    lite_meta, lite_data = [], CreateDictFromFlatbuffer(lite_model)
    lite_graph = lite_data['subgraphs'][0]
    lite_inputs = [NameListToString(lite_graph['tensors'][i]['name']) for i in lite_graph['inputs']]
    pb_inputs = [pb_t_map[s] for s in lite_inputs]
    lite_meta_path = join(TFLITE_DIR, 'config.json')
    if exists(lite_meta_path):
      with open(lite_meta_path, 'r') as f:
        lite_meta = json.load(f)
    lite_meta.append({
      "model": onnx_name.replace('.onnx', ''),
      "inputs": [pb_t_map[s] for s in lite_inputs],
      "inner_inputs": lite_inputs
    })
    with open(lite_meta_path, 'w') as f:
        json.dump(lite_meta, f, indent=4)
  save_meta(pb_model, lite_model)
  # fp16
  converter.target_spec.supported_types = [tf.float16]
  tflite_path = join(TFLITE_FP16_DIR, onnx_name.replace('.onnx', '.tflite'))
  print(f">>> {onnx_name} => {tflite_path} <<<")
  with open(tflite_path, 'wb') as f:
    f.write(converter.convert())

onnx_names = [p for p in os.listdir(ONNX_DIR) if p.endswith('.onnx')]
# MNN
for onnx_name in onnx_names:
  convert_mnn(onnx_name)
# torch / torch lite
for onnx_name in onnx_names:
  convert_pt(onnx_name)
#pb / tflite
for onnx_name in onnx_names:
  convert_tflite(onnx_name)

if not args.debug:
  shutil.rmtree(TORCH_CACHE_DIR, ignore_errors=True)
