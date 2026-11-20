"""Create an opt-in CPU config for interleaved mnncli prompt checkpoints."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--slots', type=int, default=4)
parser.add_argument('--max-mb', type=int, default=256)
parser.add_argument('--refresh', action='store_true', help='Save advancing checkpoints; requires CPU prefix snapshot support')
parser.add_argument('--tail-tokens', type=int, help='Default: 4 with --refresh, otherwise 64')
args = parser.parse_args()
if args.tail_tokens is None:
    args.tail_tokens = 4 if args.refresh else 64
if not 1 <= args.slots <= 16 or not 1 <= args.max_mb <= 4096:
    parser.error('slots must be 1..16; max-mb must be 1..4096')
if not 1 <= args.tail_tokens <= 1024:
    parser.error('tail-tokens must be 1..1024')
source = args.config.resolve()
config = json.loads(source.read_text())
if config.get('backend_type', 'cpu') != 'cpu' or config.get('sampler_type') != 'greedy':
    parser.error('checkpoints require CPU and greedy sampling')
if config.get('quant_kv', 0) != 0 or config.get('speculative_type'):
    parser.error('quantized KV and speculative decoding are not supported')
base = Path(config.get('base_dir') or source.parent)
if not base.is_absolute():
    base = source.parent / base
for key in ['llm_model', 'llm_weight', 'tokenizer_file', 'llm_config', 'audio_model',
            'audio_weight', 'visual_model', 'visual_weight']:
    if config.get(key):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else (base / value).resolve())
output = args.out.resolve()
output.mkdir(parents=True, exist_ok=True)
if (output / 'config.runtime.json').exists():
    parser.error('output config exists; choose a fresh directory')
for name in ['kv-work', 'prefixes']:
    (output / name).mkdir(exist_ok=True)
config.update(base_dir='', reuse_kv=True, prompt_cache=True, kvcache_mmap=True,
              prompt_cache_slots=args.slots, prompt_cache_max_mb=args.max_mb,
              prompt_cache_tail_tokens=args.tail_tokens, prompt_cache_refresh=args.refresh,
              tmp_path=str(output / 'kv-work'),
              prefix_cache_path=str(output / 'prefixes'))
(output / 'config.runtime.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
print(output / 'config.runtime.json')
