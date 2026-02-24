#!/usr/bin/python

import sys
import os
import argparse
import subprocess
import json
import shutil
import math
import time
import numpy as np


def load_llm_config(model_dir):
    """Load llm_config.json from the model directory."""
    config_path = os.path.join(model_dir, 'llm_config.json')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)


def is_visual_model(llm_config):
    """Check if the model is a visual (VL) model."""
    return llm_config.get('is_visual', False)


def has_deepstack(llm_config):
    """Check if the model has deepstack (Qwen3-VL specific)."""
    return llm_config.get('has_deepstack', False)


# ============ LLM I/O Generation ============

def makeIO(args):
    """Generate test I/O for LLM model (text-only, no deepstack)."""
    exe = os.path.join(os.getcwd(), args.mnn_path, "generateLlmIO")
    output = os.path.join(args.cache_path, 'testdir')
    print(os.popen(exe + " " + args.model + " " + output + ' %d' % args.chunk_size).read())


def makeIO_llm_deepstack(args, llm_config):
    """Generate test I/O for LLM model with deepstack_embeds input (Qwen3-VL).

    Since the C++ generateLlmIO tool only handles 4 standard inputs
    (input_ids, attention_mask, position_ids, logits_index), we generate
    the test I/O in Python for models that require the extra
    deepstack_embeds input.
    """
    try:
        import MNN
        import MNN.expr as expr
        import MNN.numpy as mp
    except ImportError:
        print("Error: MNN Python package is required for VL model IO generation.")
        print("Please install MNN Python package first.")
        sys.exit(1)

    hidden_size = llm_config.get('hidden_size', 2048)
    model_path = os.path.join(os.getcwd(), args.model, 'llm.mnn')
    weight_path = model_path + '.weight'
    output_dir = os.path.join(args.cache_path, 'testdir')
    os.makedirs(output_dir, exist_ok=True)

    input_names = ['input_ids', 'attention_mask', 'position_ids', 'logits_index', 'deepstack_embeds']
    output_names = ['logits']

    config = MNN.ScheduleConfig()
    rtmgr = MNN.RuntimeManager(config)
    rtmgr.set_external_file(weight_path)
    net = MNN.Module(input_names, output_names, model_path, rtmgr)

    for seq_len, is_decode in [(args.chunk_size, False), (1, True)]:
        # input_ids: [seq_len, 1, hidden_size]
        input_ids = expr.input([seq_len, 1, hidden_size], expr.NCHW, expr.float)
        input_ids_data = np.random.rand(seq_len, 1, hidden_size).astype(np.float32)
        input_ids.write(input_ids_data.tobytes())
        input_ids.set_name('input_ids')

        # attention_mask: [1, 1, seq_len, seq_len]
        attention_mask = expr.input([1, 1, seq_len, seq_len], expr.NCHW, expr.float)
        mask_data = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                mask_data[0, 0, i, j] = -3.4028235e+38  # float lowest
        attention_mask.write(mask_data.tobytes())
        attention_mask.set_name('attention_mask')

        # position_ids: [seq_len]
        position_ids = expr.input([seq_len], expr.NCHW, expr.int)
        pos_data = np.arange(seq_len, dtype=np.int32)
        position_ids.write(pos_data.tobytes())
        position_ids.set_name('position_ids')

        # logits_index: [1]
        logits_index_val = -1 if is_decode else 0
        logits_index = expr.const([logits_index_val], [1], expr.NHWC, expr.int)
        logits_index.set_name('logits_index')

        # deepstack_embeds: [3, seq_len, hidden_size] (zeros for test)
        deepstack_embeds = expr.input([3, seq_len, hidden_size], expr.NCHW, expr.float)
        ds_data = np.zeros((3, seq_len, hidden_size), dtype=np.float32)
        deepstack_embeds.write(ds_data.tobytes())
        deepstack_embeds.set_name('deepstack_embeds')

        inputs = [input_ids, attention_mask, position_ids, logits_index, deepstack_embeds]
        outputs = net.forward(inputs)
        outputs[0].set_name('logits')

        sub_dir = os.path.join(output_dir, str(seq_len))
        os.makedirs(sub_dir, exist_ok=True)

        # Fix input VARPs to constants for saving
        for inp in inputs:
            inp.fix(expr.CONSTANT)

        expr.save(inputs, os.path.join(sub_dir, 'input.mnn'))
        expr.save(outputs, os.path.join(sub_dir, 'output.mnn'))
        print("Generated LLM IO for seq_len=%d at %s" % (seq_len, sub_dir))


# ============ Visual Model I/O Generation ============

def makeIO_visual(args, llm_config):
    """Generate test I/O for visual.mnn (Qwen3-VL).

    Creates test inputs matching the Qwen3-VL vision encoder:
      - patches:       [seq_len, C * temporal_patch_size * patch_size^2]
      - position_ids:  [2, seq_len]
      - attention_mask: [1, seq_len, seq_len]
      - idx_tensor:    [4, merged_seq_len]  (bilinear interpolation indices)
      - weight_tensor: [4, merged_seq_len]  (bilinear interpolation weights)
    """
    try:
        import MNN
        import MNN.expr as expr
        import MNN.numpy as mp
    except ImportError:
        print("Error: MNN Python package is required for VL model IO generation.")
        print("Please install MNN Python package first.")
        sys.exit(1)

    model_path = os.path.join(os.getcwd(), args.model, 'visual.mnn')
    weight_path = model_path + '.weight'
    if not os.path.exists(model_path):
        print("Warning: visual.mnn not found at %s, skipping visual IO generation." % model_path)
        return

    # Qwen3-VL vision parameters
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    num_grid = llm_config.get('num_grid_per_side', 1)

    # Use a representative image size (e.g., 448x448)
    # It must be a multiple of align_size = patch_size * merge_size = 32
    image_size = llm_config.get('image_size', 448)
    align_size = patch_size * merge_size
    image_h = int(round(image_size / align_size) * align_size)
    image_w = image_h

    grid_t = 1  # single image: temporal = 2 frames / temporal_patch_size = 1
    grid_h = image_h // patch_size
    grid_w = image_w // patch_size
    seq_len = grid_t * grid_h * grid_w
    channel = 3
    num_patches = grid_h * grid_w

    output_dir = os.path.join(args.cache_path, 'visual_testdir')
    os.makedirs(output_dir, exist_ok=True)

    input_names = ['patches', 'position_ids', 'attention_mask', 'idx_tensor', 'weight_tensor']
    output_names = ['image_embeds', 'deepstack_feature']

    config = MNN.ScheduleConfig()
    rtmgr = MNN.RuntimeManager(config)
    if os.path.exists(weight_path):
        rtmgr.set_external_file(weight_path)
    net = MNN.Module(input_names, output_names, model_path, rtmgr)

    # patches: [seq_len, C * temporal_patch_size * patch_size * patch_size]
    patch_dim = channel * temporal_patch_size * patch_size * patch_size
    patches = expr.input([seq_len, patch_dim], expr.NCHW, expr.float)
    patches_data = np.random.rand(seq_len, patch_dim).astype(np.float32)
    patches.write(patches_data.tobytes())
    patches.set_name('patches')

    # position_ids: [2, seq_len] (height and width positions)
    position_ids = expr.input([2, seq_len], expr.NCHW, expr.int)
    pos_data = np.zeros((2, seq_len), dtype=np.int32)
    wblock_size = merge_size * merge_size
    hblock_size = wblock_size * grid_w // merge_size
    for i in range(grid_h):
        h_idx = i // merge_size
        h_off = i % merge_size
        for j in range(grid_w):
            w_idx = j // merge_size
            w_off = j % merge_size
            index = h_idx * hblock_size + w_idx * wblock_size + h_off * 2 + w_off
            pos_data[0, index] = i  # height position
            pos_data[1, index] = j  # width position
    position_ids.write(pos_data.tobytes())
    position_ids.set_name('position_ids')

    # attention_mask: [1, seq_len, seq_len] (all zeros = full attention)
    attention_mask = expr.input([1, seq_len, seq_len], expr.NCHW, expr.float)
    mask_data = np.zeros((1, seq_len, seq_len), dtype=np.float32)
    attention_mask.write(mask_data.tobytes())
    attention_mask.set_name('attention_mask')

    # idx_tensor: [4, num_patches] -> reshape/permute -> [4, merged_len]
    idx_data = np.zeros((4, num_patches), dtype=np.int32)
    weight_data = np.zeros((4, num_patches), dtype=np.float32)

    h_idxs = np.zeros(grid_h, dtype=np.float32)
    w_idxs = np.zeros(grid_w, dtype=np.float32)
    for i in range(grid_h):
        h_idxs[i] = float(i) * (num_grid - 1) / max(grid_h - 1, 1)
    for i in range(grid_w):
        w_idxs[i] = float(i) * (num_grid - 1) / max(grid_w - 1, 1)

    for i in range(grid_h):
        h_floor = int(h_idxs[i])
        h_ceil = min(h_floor + 1, num_grid - 1)
        dh = h_idxs[i] - h_floor
        for j in range(grid_w):
            w_floor = int(w_idxs[j])
            w_ceil = min(w_floor + 1, num_grid - 1)
            dw = w_idxs[j] - w_floor
            idx = i * grid_w + j
            idx_data[0, idx] = h_floor * num_grid + w_floor
            idx_data[1, idx] = h_floor * num_grid + w_ceil
            idx_data[2, idx] = h_ceil * num_grid + w_floor
            idx_data[3, idx] = h_ceil * num_grid + w_floor
            weight_data[0, idx] = (1.0 - dh) * (1.0 - dw)
            weight_data[1, idx] = (1.0 - dh) * dw
            weight_data[2, idx] = dh * (1.0 - dw)
            weight_data[3, idx] = dh * dw

    # Reshape and permute: [4, grid_t, grid_h/merge, merge, grid_w/merge, merge]
    # -> permute [0,1,2,4,3,5] -> reshape [4, -1]
    idx_np = idx_data.reshape(4, grid_t, grid_h // merge_size, merge_size, grid_w // merge_size, merge_size)
    idx_np = idx_np.transpose(0, 1, 2, 4, 3, 5).reshape(4, -1)
    weight_np = weight_data.reshape(4, grid_t, grid_h // merge_size, merge_size, grid_w // merge_size, merge_size)
    weight_np = weight_np.transpose(0, 1, 2, 4, 3, 5).reshape(4, -1)

    merged_len = idx_np.shape[1]
    idx_tensor = expr.input([4, merged_len], expr.NCHW, expr.int)
    idx_tensor.write(idx_np.astype(np.int32).tobytes())
    idx_tensor.set_name('idx_tensor')

    weight_tensor = expr.input([4, merged_len], expr.NCHW, expr.float)
    weight_tensor.write(weight_np.astype(np.float32).tobytes())
    weight_tensor.set_name('weight_tensor')

    inputs = [patches, position_ids, attention_mask, idx_tensor, weight_tensor]
    outputs = net.forward(inputs)
    if len(outputs) >= 1:
        outputs[0].set_name('image_embeds')
    if len(outputs) >= 2:
        outputs[1].set_name('deepstack_feature')

    sub_dir = os.path.join(output_dir, '1')
    os.makedirs(sub_dir, exist_ok=True)

    for inp in inputs:
        inp.fix(expr.CONSTANT)

    expr.save(inputs, os.path.join(sub_dir, 'input.mnn'))
    expr.save(list(outputs), os.path.join(sub_dir, 'output.mnn'))
    print("Generated Visual IO at %s" % sub_dir)


# ============ Model Separation (compilefornpu) ============

def seperate(args):
    """Run compilefornpu on the LLM model only (text-only path)."""
    exe = os.path.join(os.getcwd(), args.mnn_path, "compilefornpu")
    model = os.path.join(os.getcwd(), args.model, 'llm.mnn')
    print("model:", model)
    config = {
        "type": "QNN",
        "skips": [],
        "testdir": [],
        "cache": "qnn"
    }
    config['testdir'].append(os.path.join("testdir", '1'))
    config['testdir'].append(os.path.join("testdir", '%d' % args.chunk_size))
    cache = os.path.join(os.getcwd(), args.cache_path)
    with open(os.path.join(cache, 'qnn.json'), 'w') as f:
        f.write(json.dumps(config, indent=4))

    process = subprocess.Popen(exe + ' ' + model + ' qnn/llm.mnn qnn.json', bufsize=1,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               cwd=cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()


def seperate_visual(args):
    """Run compilefornpu on the visual model (Qwen3-VL)."""
    exe = os.path.join(os.getcwd(), args.mnn_path, "compilefornpu")
    model = os.path.join(os.getcwd(), args.model, 'visual.mnn')
    if not os.path.exists(model):
        print("Warning: visual.mnn not found, skipping visual model separation.")
        return
    print("visual model:", model)
    config = {
        "type": "QNN",
        "skips": [],
        "testdir": [],
        "cache": "qnn_visual"
    }
    config['testdir'].append(os.path.join("visual_testdir", '1'))
    cache = os.path.join(os.getcwd(), args.cache_path)
    with open(os.path.join(cache, 'qnn_visual.json'), 'w') as f:
        f.write(json.dumps(config, indent=4))

    process = subprocess.Popen(exe + ' ' + model + ' qnn/visual.mnn qnn_visual.json', bufsize=1,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               cwd=cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()


# ============ QNN Compilation ============

def compile_qnn(args):
    """Compile LLM subgraphs to QNN context binaries."""
    exe = os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "qnn", "npu_convert.py")
    cache = os.path.join(os.getcwd(), args.cache_path)
    process = subprocess.Popen("python3 " + exe + ' npu_postreat.json %d ' % args.soc_id + ' ' + args.dsp_arch,
                               bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               cwd=cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()


def compile_qnn_visual(args):
    """Compile visual model subgraphs to QNN context binaries."""
    exe = os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "qnn", "npu_convert.py")
    cache = os.path.join(os.getcwd(), args.cache_path)
    # The visual model's compilefornpu generates a separate npu_postreat.json
    # in the qnn_visual cache directory. We rename it for clarity.
    visual_postreat = os.path.join(cache, 'npu_postreat_visual.json')
    if not os.path.exists(visual_postreat):
        # compilefornpu outputs npu_postreat.json in cwd; if the second run
        # overwrites the first, look for it there
        postreat_path = os.path.join(cache, 'npu_postreat.json')
        if os.path.exists(postreat_path):
            # Already compiled LLM, so the postreat was overwritten by the visual run.
            # This path handles it if called sequentially.
            visual_postreat = postreat_path
        else:
            print("Warning: npu_postreat for visual model not found, skipping.")
            return

    process = subprocess.Popen("python3 " + exe + ' ' + os.path.basename(visual_postreat)
                               + ' %d ' % args.soc_id + ' ' + args.dsp_arch,
                               bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               cwd=cache, text=True, shell=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()


# ============ Output ============

def output_qnn(args):
    """Move QNN output files and generate config (text-only model)."""
    if os.path.exists(os.path.join(args.model, 'qnn')):
        shutil.rmtree(os.path.join(args.model, 'qnn'))
    shutil.move(os.path.join(args.cache_path, 'qnn'), os.path.join(args.model, 'qnn'))
    config_npu = {
        "llm_model": "qnn/llm.mnn",
        "backend_type": "cpu",
        "thread_num": 1,
        "precision": "low",
        "chunk_limits": [args.chunk_size, 1],
        "memory": "low",
        "sampler_type": "penalty",
        "penalty": 1.1
    }
    with open(os.path.join(args.model, "config_qnn.json"), 'w') as f:
        f.write(json.dumps(config_npu, indent=4))
    shutil.rmtree(args.cache_path)


def output_qnn_visual(args, llm_config):
    """Move QNN output files and generate config (VL model, e.g. Qwen3-VL)."""
    if os.path.exists(os.path.join(args.model, 'qnn')):
        shutil.rmtree(os.path.join(args.model, 'qnn'))
    shutil.move(os.path.join(args.cache_path, 'qnn'), os.path.join(args.model, 'qnn'))
    # Move visual QNN outputs into the same qnn directory
    qnn_visual_dir = os.path.join(args.cache_path, 'qnn_visual')
    if os.path.exists(qnn_visual_dir):
        qnn_dest = os.path.join(args.model, 'qnn')
        for item in os.listdir(qnn_visual_dir):
            src_item = os.path.join(qnn_visual_dir, item)
            dst_item = os.path.join(qnn_dest, item)
            if os.path.isdir(src_item):
                if os.path.exists(dst_item):
                    shutil.rmtree(dst_item)
                shutil.move(src_item, dst_item)
            else:
                shutil.move(src_item, dst_item)
    config_npu = {
        "llm_model": "qnn/llm.mnn",
        "visual_model": "qnn/visual.mnn",
        "backend_type": "cpu",
        "thread_num": 1,
        "precision": "low",
        "chunk_limits": [args.chunk_size, 1],
        "memory": "low",
        "sampler_type": "penalty",
        "penalty": 1.1,
        "mllm": {
            "backend_type": "cpu",
            "thread_num": 4,
            "precision": "normal",
            "memory": "low"
        }
    }
    # Propagate VL-specific config from llm_config
    for key in ['is_visual', 'has_deepstack', 'image_size', 'image_mean', 'image_norm',
                'image_pad', 'vision_start', 'vision_end', 'num_grid_per_side',
                'image_size_unit', 'image_max_size']:
        if key in llm_config:
            config_npu[key] = llm_config[key]

    with open(os.path.join(args.model, "config_qnn.json"), 'w') as f:
        f.write(json.dumps(config_npu, indent=4))
    shutil.rmtree(args.cache_path)


# ============ Conversion Pipelines ============

def convert(args):
    """Convert text-only LLM model to QNN format."""
    cache = os.path.join(os.getcwd(), args.cache_path)
    os.makedirs(cache, exist_ok=True)
    sta = time.time()
    print("Step1: Make IO")
    makeIO(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end
    print("Step2: Seperate Model")
    seperate(args)
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


def convert_visual(args, llm_config):
    """Convert VL model (e.g. Qwen3-VL) to QNN format.

    This handles both the LLM backbone (with deepstack_embeds) and
    the visual encoder (visual.mnn), generating QNN-optimized versions
    of both models.
    """
    cache = os.path.join(os.getcwd(), args.cache_path)
    os.makedirs(cache, exist_ok=True)

    sta = time.time()
    print("=" * 60)
    print("Converting VL model to QNN (Qwen3-VL mode)")
    print("=" * 60)

    # Step 1: Generate LLM I/O (with deepstack_embeds)
    print("\nStep1: Generate LLM IO (with deepstack)")
    makeIO_llm_deepstack(args, llm_config)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end

    # Step 2: Generate Visual Model I/O
    print("\nStep2: Generate Visual IO")
    makeIO_visual(args, llm_config)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end

    # Step 3: Separate LLM model for NPU
    print("\nStep3: Separate LLM Model")
    seperate(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end

    # Step 4: Compile LLM to QNN
    print("\nStep4: Compile LLM to QNN")
    compile_qnn(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end

    # Step 5: Back up the LLM npu_postreat.json before visual overwrite
    postreat_path = os.path.join(cache, 'npu_postreat.json')
    if os.path.exists(postreat_path):
        shutil.copy2(postreat_path, os.path.join(cache, 'npu_postreat_llm.json'))

    # Step 6: Separate Visual model for NPU
    print("\nStep5: Separate Visual Model")
    seperate_visual(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    sta = end

    # Step 7: Compile Visual to QNN
    # After compilefornpu runs on visual.mnn, npu_postreat.json is overwritten
    print("\nStep6: Compile Visual to QNN")
    compile_qnn_visual(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')

    # Step 8: Move outputs and generate config
    print("\nStep7: Move result files to ", args.model)
    output_qnn_visual(args, llm_config)

    print("\nEnd")


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
    args = parser.parse_args()

    # Auto-detect model type from llm_config.json
    llm_config = load_llm_config(args.model)
    if is_visual_model(llm_config):
        print("Detected VL (Vision-Language) model")
        if has_deepstack(llm_config):
            print("Model has deepstack (Qwen3-VL)")
        convert_visual(args, llm_config)
    else:
        convert(args)


if __name__ == '__main__':
    main()
