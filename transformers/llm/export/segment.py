import glob
import json
import os

from utils.config import LlmConfig
from utils.mnn_converter import MNNConverter
from utils.spinner import spinner_run
from utils.tokenizer import LlmTokenizer


def enabled(args):
    return getattr(args, 'segment', False)


@spinner_run(f'load segment export metadata ', True)
def load_metadata(exporter, model_path):
    model_path = os.path.abspath(os.path.expanduser(model_path))
    if exporter.args.tokenizer_path == exporter.args.path:
        exporter.args.tokenizer_path = model_path
    else:
        tokenizer_path = os.path.expanduser(exporter.args.tokenizer_path)
        if os.path.exists(tokenizer_path):
            tokenizer_path = os.path.abspath(tokenizer_path)
        exporter.args.tokenizer_path = tokenizer_path

    exporter.config = LlmConfig.from_pretrained(model_path)
    exporter.model_type = exporter.config.model_type
    exporter.tokenizer = LlmTokenizer.from_pretrained(
        exporter.args.tokenizer_path,
        model_type=exporter.model_type
    )
    exporter.model = None
    exporter.visual = None
    exporter.audio = None
    exporter.talker = None
    exporter.mtp = None
    exporter.scale_emb = None
    exporter.llm_config = {
        'model_type': exporter.config.model_type,
        'hidden_size': exporter.config.hidden_size,
        'layer_nums': exporter.config.num_hidden_layers,
        'attention_mask': 'float',
        'attention_type': exporter.config.attention_type,
        'is_mrope': False
    }
    if exporter.config.sliding_window > 0:
        exporter.llm_config['sliding_window'] = exporter.config.sliding_window
    if hasattr(exporter.tokenizer, 'get_chat_template'):
        chat_template = exporter.tokenizer.get_chat_template()
        if chat_template is not None:
            exporter.llm_config['jinja'] = {
                'chat_template': chat_template
            }
            if exporter.tokenizer.bos_token:
                exporter.llm_config['jinja']['bos'] = exporter.tokenizer.bos_token
            if exporter.tokenizer.eos_token:
                exporter.llm_config['jinja']['eos'] = exporter.tokenizer.eos_token
    if exporter.model_type == 'glm_ocr':
        exporter.llm_config['jinja'] = {
            'chat_template': "[gMASK]<sop>{% for message in messages %}{% if message.role == \"user\" %}<|user|>\n{{ message.content }}{% elif message.role == \"assistant\" %}<|assistant|>\n{{ message.content }}{% elif message.role == \"system\" %}<|system|>\n{{ message.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}",
            'eos': '<|endoftext|>'
        }
    source_llm_config = os.path.join(model_path, 'llm_config.json')
    if os.path.exists(source_llm_config):
        with open(source_llm_config, 'r', encoding='utf-8') as f:
            exporter.llm_config.update(json.load(f))
    return model_path


def _resource_dirs():
    export_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(export_dir, '../../..'))
    candidates = [
        os.path.join(repo_root, 'resource'),
        os.path.join(repo_root, 'transformers', 'llm', 'resource')
    ]
    return [path for path in candidates if os.path.isdir(path)]


def _workflow_score(exporter, workflow_path):
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
    except Exception:
        return None
    models = workflow.get('models', [])
    if not isinstance(models, list) or len(models) == 0:
        return None

    model_names = [model.get('name', '') for model in models if isinstance(model, dict)]
    lowered_names = [name.lower() for name in model_names]
    score = 0
    if any(name in ('hf_decoder', 'decoder', 'gpt2_decoder') for name in lowered_names):
        score += 20
    if any(name in ('logit', 'logit_mobile') for name in lowered_names):
        score += 20
    if any(name in ('encoder', 'encoder_mobile', 'audio_proj', 'ntp1', 'wpe') for name in lowered_names):
        score -= 30

    filename = os.path.basename(workflow_path).lower()
    model_type = str(getattr(exporter, 'model_type', '') or '').lower()
    if model_type and model_type in filename:
        score += 15
    if 'qwen' in model_type and 'qwen' in filename:
        score += 10
    if 'hf' in filename and any(name == 'hf_decoder' for name in lowered_names):
        score += 10

    blocks = []
    for model in models:
        if not isinstance(model, dict):
            continue
        for block in model.get('blocks', []):
            if isinstance(block, dict):
                blocks.append(block)

    cfg_pairs = {
        'hiddenSize': getattr(exporter.config, 'hidden_size', None),
        'number': getattr(exporter.config, 'num_hidden_layers', None),
        'headDim': getattr(exporter.config, 'head_dim', None),
        'numHead': getattr(exporter.config, 'num_attention_heads', None),
        'kvNumHead': getattr(exporter.config, 'num_key_value_heads', None),
        'max_position_embeddings': getattr(getattr(exporter.config, 'origin_config', None), 'max_position_embeddings', None)
    }
    weights = {
        'hiddenSize': 40,
        'number': 35,
        'headDim': 20,
        'numHead': 20,
        'kvNumHead': 20,
        'max_position_embeddings': 5
    }
    for key, cfg_value in cfg_pairs.items():
        if cfg_value is None or isinstance(cfg_value, list):
            continue
        for block in blocks:
            workflow_value = block.get(key)
            if workflow_value is None and key == 'max_position_embeddings':
                workflow_value = block.get('maxPositionEmbeddings')
            if workflow_value == cfg_value:
                score += weights[key]
                break
    return score


def _resolve_workflow(exporter):
    workflow = getattr(exporter.args, 'workflow', None)
    if workflow:
        workflow = os.path.abspath(os.path.expanduser(workflow))
        if not os.path.exists(workflow):
            raise FileNotFoundError(f'workflow json not found: {workflow}')
        return workflow

    candidates = []
    for resource_dir in _resource_dirs():
        for path in glob.glob(os.path.join(resource_dir, '**', '*.json'), recursive=True):
            score = _workflow_score(exporter, path)
            if score is not None and score > 0:
                candidates.append((score, os.path.abspath(path)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates:
        searched = ', '.join(_resource_dirs())
        raise RuntimeError(f'--workflow is not set and no suitable workflow json was found under: {searched}')

    best_score = candidates[0][0]
    best = [path for score, path in candidates if score == best_score]
    if len(best) > 1:
        lines = '\n'.join([f'  {path}' for path in best])
        raise RuntimeError(f'--workflow is not set and multiple suitable workflow json files were found:\n{lines}\nPlease pass --workflow explicitly.')

    workflow = candidates[0][1]
    print(f'--workflow is not set, use workflow json: {workflow}')
    return workflow


def _resolve_safetensors(model_path):
    model_path = os.path.abspath(os.path.expanduser(model_path))
    if os.path.isfile(model_path):
        if model_path.endswith('.safetensors'):
            return [model_path]
        raise RuntimeError(f'--segment expects --path to be a model directory or a .safetensors file, got: {model_path}')

    model_file = os.path.join(model_path, 'model.safetensors')
    if os.path.exists(model_file):
        return [model_file]

    index_files = sorted(glob.glob(os.path.join(model_path, '*.safetensors.index.json')))
    if index_files:
        with open(index_files[0], 'r', encoding='utf-8') as f:
            index = json.load(f)
        ordered = []
        for filename in index.get('weight_map', {}).values():
            if filename not in ordered:
                ordered.append(filename)
        paths = [os.path.join(model_path, filename) for filename in ordered]
    else:
        paths = sorted(glob.glob(os.path.join(model_path, '*.safetensors')))

    paths = [path for path in paths if os.path.exists(path)]
    if not paths:
        raise RuntimeError(f'no safetensors file found under: {model_path}')
    if len(paths) > 1:
        print(f'found {len(paths)} safetensors files, pass all of them to MNNConvert')
    return paths


def _quant_args(exporter):
    quant_bit = exporter.args.quant_bit
    if quant_bit == 32:
        return []
    if quant_bit == 16:
        return ['--fp16']
    return [
        '--weightQuantBits',
        str(quant_bit),
        '--weightQuantBlock',
        str(exporter.args.quant_block)
    ]


@spinner_run(f'convert safetensors model to ')
def _convert_safetensors(exporter, workflow_path, safetensors_paths):
    convert_args = [
        '',
        '-f',
        'ST',
        '-i',
        str(workflow_path)
    ]
    for safetensors_path in safetensors_paths:
        convert_args += ['-i', str(safetensors_path)]
    convert_args += [
        '-o',
        str(exporter.args.dst_path),
        '--allowCustomOp'
    ]
    if exporter.args.transformer_fuse:
        convert_args += ['--transformerFuse']
    if exporter.args.group_conv_native:
        convert_args += ['--groupConvNative']
    if exporter.args.sym:
        convert_args += ['--weightQuantAsymmetric=0']
    convert_args += ['--saveExternalData']
    if exporter.args.hqq:
        convert_args += ['--hqq']
    convert_args += _quant_args(exporter)
    MNNConverter(exporter).convert(convert_args)
    return exporter.args.dst_path


def _export_config(exporter, tokenizer_file):
    with open(f'{exporter.args.dst_path}/export_args.json', 'w', encoding='utf-8') as f:
        json.dump(exporter.args.__dict__, f, ensure_ascii=False, indent=4)
    config_json = f'{exporter.args.dst_path}/llm_config.json'
    with open(config_json, 'w', encoding='utf-8') as f:
        json.dump(exporter.llm_config, f, ensure_ascii=False, indent=4)

    stop_ids = getattr(exporter.tokenizer, 'stop_ids', [])
    eos_token = getattr(exporter.tokenizer, 'eos_token_id', None)
    if eos_token is None and len(stop_ids) > 0:
        eos_token = stop_ids[0]
    if isinstance(eos_token, list):
        eos_token = eos_token[0] if len(eos_token) > 0 else None
    config = {
        'forwardtype': 1,
        'precision': 2,
        'memory': 2,
        'speculative': 0,
        'draft_len': 1,
        'max_decode_tokens': exporter.max_new_tokens,
        'mnn_llm_version': 'segment',
        'tokenizer_file': os.path.basename(tokenizer_file)
    }
    if eos_token is not None:
        config['eos_token'] = int(eos_token)
    with open(f'{exporter.args.dst_path}/config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    return config_json


def export(exporter, export_type):
    if export_type != 'mnn':
        raise RuntimeError('--segment only supports --export mnn')
    workflow = _resolve_workflow(exporter)
    safetensors_paths = _resolve_safetensors(exporter.args.path)
    _convert_safetensors(exporter, workflow, safetensors_paths)
    tokenizer_file = exporter.export_tokenizer()
    _export_config(exporter, tokenizer_file)
