import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def convert(onnx_path, mnn_path, extra):
    print('Onnx path: ', onnx_path)
    print('MNN path: ', mnn_path)
    print('Extra: ', extra)
    convert_path = '../../../build/MNNConvert'
    if not os.path.exists(convert_path):
        print(convert_path + " not exist, use pymnn instead")
        convert_path = 'mnnconvert'
    extra_args = shlex.split(extra) if extra else []
    models = ['text_encoder', 'unet', 'vae_decoder']
    for model in models:
        cmd = [
            convert_path,
            '-f', 'ONNX',
            '--modelFile', os.path.join(onnx_path, model, 'model.onnx'),
            '--MNNModel', os.path.join(mnn_path, model + '.mnn'),
            '--saveExternalData=1',
        ] + extra_args
        print(' '.join(shlex.quote(x) for x in cmd))
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)


def export_diffusion_mtok(tokenizer_src_root, model_root):
    this_dir = Path(__file__).resolve().parent
    llm_export_dir = (this_dir / '../../llm/export').resolve()
    if str(llm_export_dir) not in sys.path:
        sys.path.insert(0, str(llm_export_dir))

    from transformers import AutoTokenizer
    from utils.tokenizer import LlmTokenizer

    def read_model_type(dir_path):
        for name in ('config.json', 'tokenizer_config.json'):
            config_path = dir_path / name
            if not config_path.exists():
                continue
            try:
                with config_path.open('r', encoding='utf-8') as fp:
                    model_type = json.load(fp).get('model_type')
                if model_type:
                    return model_type
            except Exception:
                pass
        return None

    def load_tokenizer(dir_path, use_fast):
        return AutoTokenizer.from_pretrained(dir_path.as_posix(), trust_remote_code=True, use_fast=use_fast)

    def infer_model_type(dir_path):
        model_type = read_model_type(dir_path)
        if model_type:
            return model_type
        tokenizer = None
        for use_fast in (False, True):
            try:
                tokenizer = load_tokenizer(dir_path, use_fast=use_fast)
                break
            except Exception:
                pass
        if tokenizer is None:
            return 'clip'
        class_name = type(tokenizer).__name__.lower()
        if 'bert' in class_name:
            return 'bert'
        if 't5' in class_name:
            return 't5'
        return 'clip'

    src_root = Path(tokenizer_src_root)
    dst_root = Path(model_root)

    candidate_dirs = []
    if (src_root / 'tokenizer').is_dir():
        candidate_dirs.append((src_root / 'tokenizer', dst_root))
    elif src_root.is_dir():
        candidate_dirs.append((src_root, dst_root))

    for src_dir, dst_dir in candidate_dirs:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for item in src_dir.iterdir():
            if item.is_file():
                target = dst_dir / item.name
                if target.resolve() != item.resolve():
                    target.write_bytes(item.read_bytes())

        tokenizer_json = dst_dir / 'tokenizer.json'
        if not tokenizer_json.exists():
            hf_tok = None
            for use_fast in (False, True):
                try:
                    hf_tok = load_tokenizer(dst_dir, use_fast=use_fast)
                    break
                except Exception:
                    pass
            if hf_tok is None:
                raise RuntimeError(f'Failed to materialize tokenizer.json from {dst_dir}')
            hf_tok.save_pretrained(dst_dir.as_posix())

        model_type = infer_model_type(dst_dir)
        llm_tok = LlmTokenizer(dst_dir.as_posix(), model_type)
        out_path = llm_tok.export(dst_dir.as_posix(), model_path=dst_dir.as_posix(), model_type=model_type)
        if out_path.endswith('tokenizer.mtok'):
            print(f'Export tokenizer to {out_path}')
        else:
            print(f'Warning: tokenizer export fallback for {dst_dir}, got {out_path}')

if __name__ == '__main__':
    extra = ""
    extra = " ".join(sys.argv[3:])
    convert(sys.argv[1], sys.argv[2], extra)
    export_diffusion_mtok(sys.argv[1], sys.argv[2])
