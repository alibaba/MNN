#!/usr/bin/env python3
"""Correctness oracle for low-bit (2/3/4) MNN LLM exports.

Decodes quantized conv weights directly from the exported .mnn.weight file
(header + MSB-first bit unpack + fp16 alpha), injects them into the original
HuggingFace model, and greedy-generates. The Metal/CPU backend output must
match this token-for-token on coherent models (e.g. 3/4-bit); for degraded
low-bit models compare the common prefix and calibrate noise sensitivity
(see skills/metal-optimize/perf-playbook.md §1.1.8).

Independent of MNN runtime code, so it also works when the CPU low-bit
kernels are unavailable/broken on the host.

Usage:
    python3 mnn_quant_ref.py <mnn_model_dir> <hf_model_dir> [n_tokens]
"""
import os, sys, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def unpack(packed, total, bits):
    idx = np.zeros(total, dtype=np.uint8)
    mask = (1 << bits) - 1
    bitpos = 0
    for i in range(total):
        r = bitpos % 8
        shift = 8 - bits - r
        if shift < 0:
            b0 = packed[bitpos // 8]; b1 = packed[bitpos // 8 + 1]
            idx[i] = ((b0 << (-shift)) | (b1 >> (8 + shift))) & mask
        else:
            idx[i] = (packed[bitpos // 8] >> shift) & mask
        bitpos += bits
    return idx.astype(np.float32) - (1 << (bits - 1))


def decode_conv(data, ext, ic, oc, bits):
    off, wlen, alen = ext[0], ext[1], ext[2]
    hdr = data[off:off + wlen]
    p = 0
    dim_num = hdr[p]; p += 1
    dsz = 4 if (oc > 65535 or ic > 65535) else 2
    p += dim_num * dsz
    cnt = hdr[p]; p += 1
    cnt = 256 if cnt == 0 else cnt
    p += cnt
    packed = np.frombuffer(hdr[p:p + (oc * ic * bits + 7) // 8], np.uint8)
    alpha = np.frombuffer(data[off + wlen:off + wlen + alen], np.float16).astype(np.float32)
    idx = unpack(packed, oc * ic, bits).reshape(oc, ic)
    block_num = alpha.size // (oc * 2)
    block = ic // block_num
    a = alpha.reshape(oc, block_num, 2)
    w = np.zeros((oc, ic), np.float32)
    for b in range(block_num):
        w[:, b * block:(b + 1) * block] = idx[:, b * block:(b + 1) * block] * a[:, b, 1:2] + a[:, b, 0:1]
    return w


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    model_dir, hf_dir = sys.argv[1], sys.argv[2]
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    mj = json.load(open(f'{model_dir}/llm.mnn.json'))
    data = open(f'{model_dir}/llm.mnn.weight', 'rb').read()

    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    with torch.no_grad():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

    n_loaded = 0
    for op in mj['oplists']:
        if op['type'] != 'Convolution' or 'quanParameter' not in op.get('main', {}):
            continue
        name = op['name']
        ic = op['main']['common']['inputCount']; oc = op['main']['common']['outputCount']
        bits = op['main']['quanParameter']['aMaxOrBits']
        w = decode_conv(data, op['main']['external'], ic, oc, bits)
        parts = [p for p in name.split('/') if p]
        assert parts[-1] == 'Linear', name
        hf_name = '.'.join(parts[:-1])
        if 'lm_head' in parts:
            hf_name = 'lm_head'
        elif not hf_name.startswith('lm_head'):
            hf_name = 'model.' + hf_name
        mod = model
        for seg in hf_name.split('.'):
            mod = getattr(mod, seg)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(w))
            if 'lm_head' in parts and getattr(model.config, 'tie_word_embeddings', False):
                # tied models only: the input embedding lookup shares these weights.
                # Untied models keep their own embed_tokens - overwriting it here
                # would destroy the embedding table and collapse generation.
                model.model.embed_tokens.weight.copy_(torch.from_numpy(w))
        n_loaded += 1
    print(f"loaded {n_loaded} conv weights", file=sys.stderr)

    prompt = "Hello, my name is"
    txt = tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False)
    ids = tok.encode(txt, add_special_tokens=False)
    print(f"prompt tokens: {len(ids)}", file=sys.stderr)
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=n_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][len(ids):]
    print(tok.decode(gen, skip_special_tokens=False))


if __name__ == '__main__':
    main()
