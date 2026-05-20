#!/usr/bin/python

import sys
import os
import argparse
import subprocess
import json
import shutil
import time

def makeIO(args, model_name, inputjson, external_file = None):
    exe = os.path.join(os.getcwd(), args.mnn_path, "generateIO")
    model = os.path.join(os.getcwd(), args.model, model_name)
    cache = os.path.join(os.getcwd(), args.cache_path)
    output = os.path.join(cache, 'testdir')
    os.makedirs(output, exist_ok=True)
    print(os.popen(exe + " " + model + " " + inputjson + " " + output + " " + external_file).read())

def makeIOJson(args, seq_len, hidden_size, mask_type):
    config = {
        "configs": [
            {
                "inputs": [
                    {
                        "name": "input_ids",
                        "shape": [seq_len, 1, hidden_size]
                    },
                    {
                        "name": "attention_mask",
                        "shape": [1, 1, seq_len, seq_len],
                        "type": mask_type
                    },
                    {
                        "name": "position_ids",
                        "shape": [1, seq_len],
                        "type": "int"
                    },
                    {
                        "name": "logits_index",
                        "shape": [1],
                        "type": "int",
                        "value": 0
                    }
                ],
                "outputs": [
                    "logits"
                ]
            },
            {
                "inputs": [
                    {
                        "name": "input_ids",
                        "shape": [1, 1, hidden_size]
                    },
                    {
                        "name": "attention_mask",
                        "shape": [1, 1, 1, 1],
                        "type": mask_type
                    },
                    {
                        "name": "position_ids",
                        "shape": [1, 1],
                        "type": "int"
                    },
                    {
                        "name": "logits_index",
                        "shape": [1],
                        "type": "int",
                        "value": -1
                    }
                ],
                "outputs": [
                    "logits"
                ]
            }
        ]
    }
    if "Qwen3.5" in args.model:
        cfg = config["configs"]
        inputs = cfg[0]["inputs"]
        for inp in inputs:
            if inp["name"] == "attention_mask":
                inp["shape"] = [2, 1, seq_len, seq_len, 3]
            if inp["name"] == "position_ids":
                inp["shape"] = [3, seq_len]

        inputs = cfg[1]["inputs"]
        for inp in inputs:
            if inp["name"] == "attention_mask":
                inp["shape"] = [2, 1, 1, 1, 3]		
            if inp["name"] == "position_ids":
                inp["shape"] = [3, 1]
    if "Qwen" in args.model and "VL" in args.model:
        cfg = config["configs"]
        inputs = cfg[0]["inputs"]
        for inp in inputs:
            if inp["name"] == "position_ids":
                inp["shape"] = [3, seq_len]

        new_input = {
            "name": "deepstack_embeds",
            "shape": [3, 1, 1]
        }
        inputs.append(new_input)

        inputs = cfg[1]["inputs"]
        for inp in inputs:
            if inp["name"] == "position_ids":
                inp["shape"] = [3, 1]

        new_input = {
            "name": "deepstack_embeds",
            "shape": [3, 1, 1]
        }
        inputs.append(new_input)
    cache = os.path.join(os.getcwd(), args.cache_path)
    with open(os.path.join(cache, 'input.json'), 'w') as f:
        f.write(json.dumps(config, indent=4))

def makeVLIOJson(args, image_sizes):
    configs = []
    for w, h in image_sizes:
        if "Qwen2.5" in args.model and "VL" in args.model:
            align_size = 28
            grid_h = (round(h / align_size) * align_size) // 14
            grid_w = (round(w / align_size) * align_size) // 14
            seq_len = grid_h * grid_w
            config = {
                "inputs": [
                    {"name": "patches", "shape": [seq_len, 1176]},
                    {"name": "position_ids", "shape": [2, seq_len]},
                    {"name": "attention_mask", "shape": [2, 1, seq_len, seq_len]},
                    {"name": "window_index", "shape": [seq_len//4]}
                ],
                "outputs": ["image_embeds"]
            }
        elif "Qwen3" in args.model or "Qwen3.5" in args.model:
            align_size = 32
            grid_h = (round(h / align_size) * align_size) // 16
            grid_w = (round(w / align_size) * align_size) // 16
            seq_len = grid_h * grid_w
            config = {
                "inputs": [
                    {"name": "patches", "shape": [seq_len, 1536]},
                    {"name": "position_ids", "shape": [2, seq_len]},
                    {"name": "attention_mask", "shape": [1, seq_len, seq_len]},
                    {"name": "idx_tensor", "shape": [4, seq_len]},
                    {"name": "weight_tensor", "shape": [4, seq_len]}
                ],
                "outputs": ["image_embeds"]
            }
        elif "FastVLM" in args.model:
            config = {
                "inputs": [
                    {"name": "input_images", "shape": [1, 3, h, w]}
                ],
                "outputs": ["image_embeds"]
            }
        else:
            raise ValueError(f"Unsupported visual model: {args.model}")
        configs.append(config)

    full_config = {"configs": configs}
    cache = os.path.join(os.getcwd(), args.cache_path)
    with open(os.path.join(cache, 'input.json'), 'w') as f:
        json.dump(full_config, f, indent=4)

def convert_fastvlm(args, image_sizes):
    qnn_sdk = os.environ["QNN_SDK_ROOT"]
    exe = os.path.join(os.getcwd(), args.mnn_path, "MNN2QNNModel")
    model = os.path.join(os.getcwd(), args.model, 'visual.mnn')
    cache = os.path.join(os.getcwd(), args.cache_path)
    output = os.path.join(cache, 'qnn')
    os.makedirs(output, exist_ok=True)
    result = " ".join([f"1x3x{w}x{h}" for w, h in image_sizes])
    print(os.popen(exe + " " + qnn_sdk + " " + str(args.soc_id) + " " + str(args.dsp_arch).lstrip('v') + " " + model + " " + output + " " + str(len(image_sizes)) + " " + result).read())

    for item in os.listdir(output):
        s = os.path.join(output, item)
        d = os.path.join(args.model, item)
        if os.path.exists(d):
            if os.path.isfile(d): os.remove(d)
            else: shutil.rmtree(d)
        shutil.move(s, d)
    qnn_file = 'visual_' + str(args.soc_id) + "_" + str(args.dsp_arch).lstrip('v') + '.mnn'
    config_npu = {
        "llm_model": "llm.mnn",
        "llm_weight": "llm.mnn.weight",
        "backend_type": "cpu",
        "thread_num": 4,
        "precision": "low",
        "memory": "low",
        "sampler_type": "penalty",
        "penalty": 1.1,
        "visual_model": qnn_file,
         "mllm": {
            "backend_type": "cpu",
            "thread_num": 4,
            "precision": "normal",
            "memory": "low"
        }
    }
    with open(os.path.join(args.model, "config_qnn.json"), 'w') as f:
        f.write(json.dumps(config_npu, indent = 4))
    shutil.rmtree(args.cache_path)
    

def seperate(args, model_name, ids):
    exe = os.path.join(os.getcwd(), args.mnn_path, "compilefornpu")
    model = os.path.join(os.getcwd(), args.model, model_name)
    config = {
        "type":"QNN",
        "skips":[
        ],
        "testdir":[
        ],
        "KVCACHE_SIZE_LIMIT":args.max_history_token,
        "cache":"qnn"
    }
    for i in ids:
        config['testdir'].append(os.path.join("testdir", '%d' %i))
    cache = os.path.join(os.getcwd(), args.cache_path)
    with open(os.path.join(cache, 'qnn.json'), 'w') as f:
        f.write(json.dumps(config, indent=4))
    process = subprocess.Popen(exe + ' ' + model + ' qnn/' + model_name + ' qnn.json', bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd = cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')

    process.wait()
    return process.returncode

def compile_qnn(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "qnn", "npu_convert.py")
    cache = os.path.join(os.getcwd(), args.cache_path)
    process = subprocess.Popen("python3 " + exe + ' npu_postreat.json %d ' %args.soc_id + ' ' + args.dsp_arch, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd = cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    return process.returncode

def output_qnn(args):
    if os.path.exists(os.path.join(args.model, 'qnn')):
        shutil.rmtree(os.path.join(args.model, 'qnn'))
    shutil.move(os.path.join(args.cache_path, 'qnn'), os.path.join(args.model, 'qnn'))
    if args.need_config_json is True:
        config_path = os.path.join(args.model, 'config.json')
        config_npu = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_npu = json.load(f)
        is_visual = args.model_name == "visual.mnn"
        if not is_visual:
            config_npu["llm_model"] = "qnn/llm.mnn"
            config_npu["chunk_limits"] = [args.chunk_size, 1]
        else:
            config_npu["visual_model"] = "qnn/visual.mnn"
        with open(os.path.join(args.model, "config_qnn.json"), 'w') as f:
            f.write(json.dumps(config_npu, indent = 4))
    shutil.rmtree(args.cache_path)

def convert_qnn(args, model_name, inputjson, external_file, ids):
    sta = time.time()
    print("Step1: Make IO")
    makeIO(args, model_name, inputjson, external_file)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end
    print("Step2: Seperate Model")
    seperate(args, model_name, ids)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end
    print("Step3: Compile to QNN")
    compile_qnn(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    print("Step4: Move result file to ", args.model)
    output_qnn(args)

    print("End")

def convert_visual(args):
    cache = os.path.join(os.getcwd(), args.cache_path)
    os.makedirs(cache, exist_ok=True)

    external_file = os.path.join(os.getcwd(), args.model, 'visual.mnn.weight')
    image_sizes = []
    try:
        size_strs = [s.strip() for s in args.image_sizes.split(',')]
        for sz in size_strs:
            if 'x' not in sz:
                raise ValueError(f"Invalid size format: {sz}, expected 'WxH'")
            w_str, h_str = sz.split('x')
            w, h = int(w_str), int(h_str)
            if w <= 0 or h <= 0:
                raise ValueError(f"Width and height must be positive: {sz}")
            image_sizes.append((w, h))
    except Exception as e:
        print(f"Error parsing --image_sizes: {e}")
        sys.exit(1)

    if not image_sizes:
        print("No valid image sizes provided.")
        sys.exit(1)
    if "FastVLM" in args.model:
        convert_fastvlm(args, image_sizes)
    else:
        makeVLIOJson(args, image_sizes)
        inputjson = os.path.join(cache, 'input.json')
        ids = list(range(len(image_sizes)))
        convert_qnn(args, 'visual.mnn', inputjson, external_file, ids)

def convert_llm(args):
    cache = os.path.join(os.getcwd(), args.cache_path)
    os.makedirs(cache, exist_ok=True)
    hidden_size = 768
    mask_type = "int"
    config_file_path = os.path.join(os.getcwd(), args.model, 'llm_config.json')
    with open(config_file_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
        if "hidden_size" in config_data:
            hidden_size = config_data["hidden_size"]
        else:
            print(f"Error: 'hidden_size' key not found in {config_file_path}")
            return npu_convert
        if "attention_mask" in config_data:
            mask_type = config_data["attention_mask"]
    
    ids = [0, 1]
    external_file = os.path.join(os.getcwd(), args.model, 'llm.mnn.weight')
    makeIOJson(args, 128, hidden_size, mask_type)
    inputjson = os.path.join(cache, 'input.json')
    convert_qnn(args, 'llm.mnn', inputjson, external_file, ids)

def convert_input_json(args):
    cache = os.path.join(os.getcwd(), args.cache_path)
    os.makedirs(cache, exist_ok=True)
    input_shape_num = 1
    with open(args.input_json, 'r') as f:
        data = json.load(f)
        if 'configs' in data and isinstance(data['configs'], list):
            input_shape_num = len(data['configs'])
            print(input_shape_num)
    
    ids = list(range(input_shape_num))
    args.need_config_json = False
    external_file = os.path.join(os.getcwd(), args.model, args.external_file)
    convert_qnn(args, args.model_name, args.input_json, external_file, ids)

def convert(args):
    if args.input_json != "":
        convert_input_json(args)
    elif args.model_name == "llm.mnn":
        convert_llm(args)
    elif args.model_name == "visual.mnn":
        convert_visual(args)

def main():
    parser = argparse.ArgumentParser(description='generate_llm_qnn', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', type=str, required=True,
                        help='model(`str` or `os.PathLike`):\nCan be either:')
    parser.add_argument('--soc_id', type=int, required=True,
                        help='type(`int`, *optional*):'
                        '\n\tThe soc_id., for 8gen3 is 57'
                        )
    parser.add_argument('--dsp_arch', type=str, required=True,
                        help='type(`str`, *optional*):'
                        '\n\tThe dsp_arch, for 8gen3 is v75.'
                        )
    parser.add_argument('--mnn_path', type=str, default="../../../build/",
                        help='mnn build path(`str` or `os.PathLike`):\nCan be either:'
    )
    parser.add_argument('--cache_path', type=str, default="tmp",
                        help='cache path for work'
                        )
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='chunk_size for npu'
                        )
    parser.add_argument('--max_history_token', type=int, default=0,
                        help='max history token, default is 0, which mean no limit for history token number'
    )
    parser.add_argument('--image_sizes', type=str, default="512x512",
                        help='Image sizes for vision model, e.g., "512x512" or "224x224,384x384,512x512"'
                        )
    parser.add_argument('--input_json', type=str, default="",
                        help='input json contain all input shape'
                        )
    parser.add_argument('--external_file', type=str, default="",
                        help='external file stored weight'
                        )
    parser.add_argument('--model_name', type=str, default="llm.mnn",
                        help='the name of model, like llm.mnn or visual.mnn'
                        )
    parser.add_argument('--need_config_json', type=bool, default=True,
                        help='wheather generate config json'
                        )
    args = parser.parse_args()
    convert(args)


if __name__ == '__main__':
    main()