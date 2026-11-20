"""Export MNN text assets for a Qwen3.5 model already compiled with TORQ."""

import errno
import filecmp
import json
import os
import tempfile
from pathlib import Path

from utils.tokenizer import LlmTokenizer


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def graph_size(root):
    manifest = root / 'torq.bm.json'
    ext = read_json(manifest)['extension_files']
    files = {manifest, root / 'torq.llm.json', root / ext['llm_token_embeddings']}
    for name in ('prefill', 'chunked_prefill', 'decode', 'linear_output'):
        graph = root / ext['graph'][name + '.bin']
        config = graph.parent / 'bm_configure.json'
        coeff = root / read_json(config)['coefficient_config_path']
        files.update((graph, config, coeff))
        files.update(coeff.parent / entry for entry in read_json(coeff)['coefficient_map'].values())
    size = 0
    for path in files:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing or empty TORQ model file: {path}')
        size += path.stat().st_size
    return size


def export_torq(args):
    if not args.torq_model_dir:
        raise ValueError('--export torq requires --torq_model_dir from the A210 compiler')
    if args.dst_file or args.lora_path or args.eagle_path or args.dflash_path:
        raise ValueError('TORQ export requires a directory and a compiled base text model')
    if args.tokenizer_path and Path(args.tokenizer_path).resolve() != Path(args.path).resolve():
        raise ValueError('TORQ uses the tokenizer from --path to match the token IDs in the compiled graph')
    graph = Path(args.torq_model_dir).resolve()
    dst = Path(args.dst_path).resolve()
    hf = read_json(Path(args.path) / 'config.json')
    text = hf.get('text_config', hf)
    manifest = read_json(graph / 'torq.llm.json')
    cfg, options = manifest['config'], manifest['forge_option']
    if (hf['model_type'] != 'qwen3_5' or cfg['architectures'] != 'qwen3_5' or
            options['base_dtype'] != 12 or options['max_seq_len'] <= 0 or
            any(text[k] != cfg[k] for k in ('hidden_size', 'num_hidden_layers', 'vocab_size'))):
        raise ValueError('Expected matching Qwen3.5 text and A210 BF16 graph configurations')
    size = graph_size(graph)
    tokenizer = LlmTokenizer.from_pretrained(args.path, model_type='qwen3_5')
    template = tokenizer.get_chat_template()
    if not template:
        raise ValueError('The Qwen3.5 tokenizer requires a chat template')
    jinja = {'chat_template': template, 'context': {'enable_thinking': False}}
    for name, token in [('bos', tokenizer.bos_token), ('eos', tokenizer.eos_token)]:
        if token:
            jinja[name] = token
    model = {'model_type': 'qwen3_5', 'hidden_size': cfg['hidden_size'],
             'layer_nums': cfg['num_hidden_layers'], 'jinja': jinja}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent) as staging:
        out = Path(staging) / 'model'
        tokenizer_file = Path(tokenizer.export(str(out)))
        config = {'delegate': 'TORQ', 'npu_model_dir': os.path.relpath(graph, dst),
                  'tokenizer_file': tokenizer_file.name, 'model_size_bytes': size,
                  'sampler_type': 'greedy', 'reuse_kv': False}
        for name, data in [('llm_config.json', model), ('config.json', config)]:
            (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        try:
            out.replace(dst)
        except OSError as error:
            if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            for file in out.iterdir():
                existing = dst / file.name
                if not existing.is_file() or not filecmp.cmp(file, existing, shallow=False):
                    raise FileExistsError(f'Choose a new or empty output directory: {dst}') from error
    print(f'TORQ config: {dst / "config.json"} (graph: {graph})')
    return str(dst / 'config.json')
