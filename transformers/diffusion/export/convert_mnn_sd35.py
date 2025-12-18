import os
import shlex
import subprocess


def convert(onnx_path, mnn_path, extra):
    print('Onnx path: ', onnx_path)
    print('MNN path: ', mnn_path)
    print('Extra: ', extra)
    convert_path = '../../../build/MNNConvert'
    if not os.path.exists(convert_path):
        print(convert_path + " not exist, use pymnn instead")
        convert_path = 'mnnconvert'
    extra_args = shlex.split(extra) if extra else []
    models = ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'transformer', 'vae_decoder']
    for model in models:
        model_file = os.path.join(onnx_path, model, 'model.onnx')
        out_file = os.path.join(mnn_path, model + '.mnn')
        cmd = [
            convert_path,
            '-f', 'ONNX',
            '--modelFile', model_file,
            '--MNNModel', out_file,
            '--saveExternalData=1',
        ] + extra_args
        print(' '.join(shlex.quote(x) for x in cmd))
        try:
            result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(result.stdout)
        except Exception as e:
            print(f'Run convert failed for {model}: {e}')


def export_sd35_mtok(tokenizer_src_root, model_root):
    import json
    import shutil
    import sys
    from pathlib import Path

    this_dir = Path(__file__).resolve().parent
    llm_export_dir = (this_dir / '../../llm/export').resolve()
    if str(llm_export_dir) not in sys.path:
        sys.path.insert(0, str(llm_export_dir))

    from transformers import AutoTokenizer
    from utils.tokenizer import LlmTokenizer

    tokenizer_dirs = ['tokenizer', 'tokenizer_2', 'tokenizer_3']
    for name in tokenizer_dirs:
        src_tok_dir = os.path.join(tokenizer_src_root, name)
        dst_tok_dir = os.path.join(model_root, name)

        if not os.path.isdir(src_tok_dir) and not os.path.isdir(dst_tok_dir):
            print(f'Skip {name}: not found in source({tokenizer_src_root}) or output({model_root})')
            continue

        if os.path.isdir(src_tok_dir):
            os.makedirs(dst_tok_dir, exist_ok=True)
            for filename in os.listdir(src_tok_dir):
                src_file = os.path.join(src_tok_dir, filename)
                dst_file = os.path.join(dst_tok_dir, filename)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)

        tok_dir = dst_tok_dir

        tokenizer_json = os.path.join(tok_dir, 'tokenizer.json')
        if not os.path.exists(tokenizer_json):
            # Materialize tokenizer.json from vocab/merges or sentencepiece assets if possible.
            try:
                hf_tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True, use_fast=True)
                hf_tok.save_pretrained(tok_dir)
            except Exception as e:
                print(f'Skip {tok_dir}: cannot create tokenizer.json ({e})')
                continue

        model_type = 't5'
        config_path = os.path.join(tok_dir, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as fp:
                    model_type = json.load(fp).get('model_type', model_type)
            except Exception:
                pass

        llm_tok = LlmTokenizer(tok_dir, model_type)
        out_path = llm_tok.export(tok_dir, model_path=tok_dir, model_type=model_type)
        if out_path.endswith('tokenizer.mtok'):
            print(f'Generated mtok: {out_path}')
        else:
            print(f'Warning: tokenizer export fallback for {tok_dir}, got {out_path}')

if __name__ == '__main__':
    import sys
    from pathlib import Path

    if len(sys.argv) < 3:
        print('Usage: python convert_mnn_sd35.py <onnx_root> <mnn_root> [extra_convert_args] [--tokenizer_root=/path/to/tokenizers]')
        sys.exit(1)

    this_dir = Path(__file__).resolve().parent
    llm_export_dir = (this_dir / '../../llm/export').resolve()
    if str(llm_export_dir) not in sys.path:
        sys.path.insert(0, str(llm_export_dir))

    tokenizer_root = sys.argv[1]
    extra_args = []
    for arg in sys.argv[3:]:
        if arg.startswith('--tokenizer_root='):
            tokenizer_root = arg.split('=', 1)[1]
        else:
            extra_args.append(arg)
    extra = " ".join(extra_args)

    convert(sys.argv[1], sys.argv[2], extra)
    export_sd35_mtok(tokenizer_root, sys.argv[2])
