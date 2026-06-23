import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


WAN_MODELS = ("text_encoder", "transformer", "vae_decoder")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert exported Wan ONNX models to MNN.")
    parser.add_argument("onnx_path_pos", nargs="?", help="ONNX root. Kept for compatibility with older scripts.")
    parser.add_argument("mnn_root_pos", nargs="?", help="MNN output root. Kept for compatibility with older scripts.")
    parser.add_argument("--onnx_path", help="ONNX root produced by wan_onnx_export.py.")
    parser.add_argument("--mnn_root", "--output_path", dest="mnn_root", help="Directory for Wan MNN resources.")
    parser.add_argument(
        "--tokenizer_path",
        help="Tokenizer source directory. Defaults to <onnx_path>/tokenizer.",
    )
    parser.add_argument("--mnnconvert", help="Path to MNNConvert. Defaults to build/MNNConvert or mnnconvert in PATH.")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra MNNConvert arguments, for example: --extra --weightQuantBits=8",
    )
    args = parser.parse_args()
    args.onnx_path = args.onnx_path or args.onnx_path_pos
    args.mnn_root = args.mnn_root or args.mnn_root_pos
    if not args.onnx_path or not args.mnn_root:
        parser.error("both --onnx_path and --mnn_root are required")
    return args


def find_mnnconvert(explicit_path=None):
    if explicit_path:
        return explicit_path
    repo_root = Path(__file__).resolve().parents[4]
    local = repo_root / "build" / "MNNConvert"
    if local.exists():
        return local.as_posix()
    found = shutil.which("mnnconvert")
    if found:
        return found
    return "mnnconvert"


def normalize_extra(extra):
    args = []
    for item in extra or []:
        args.extend(shlex.split(item))
    return args


def read_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def find_arg_value(extra_args, name):
    prefix = name + "="
    for index, item in enumerate(extra_args):
        if item == name:
            if index + 1 < len(extra_args):
                return extra_args[index + 1]
            return True
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def build_flag_summary(extra_args):
    weight_quant_bits = find_arg_value(extra_args, "--weightQuantBits")
    weight_quant_block = find_arg_value(extra_args, "--weightQuantBlock")
    return {
        "fp16": bool(find_arg_value(extra_args, "--fp16")),
        "hqq": bool(find_arg_value(extra_args, "--hqq")),
        "transformer_fuse": bool(find_arg_value(extra_args, "--transformerFuse")),
        "weight_quant": weight_quant_bits is not None,
        "weight_quant_bits": int(weight_quant_bits) if weight_quant_bits not in (None, True) else weight_quant_bits,
        "weight_quant_block": int(weight_quant_block) if weight_quant_block not in (None, True) else weight_quant_block,
    }


def convert_one(convert_path, onnx_file, mnn_file, extra_args):
    if not onnx_file.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_file}")
    mnn_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        convert_path,
        "-f",
        "ONNX",
        "--modelFile",
        onnx_file.as_posix(),
        "--MNNModel",
        mnn_file.as_posix(),
        "--saveExternalData=1",
    ] + extra_args
    print(" ".join(shlex.quote(x) for x in cmd))
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"MNNConvert failed for {onnx_file} with exit code {result.returncode}")
    return {
        "command": cmd,
        "mnn_model": mnn_file.as_posix(),
        "onnx_model": onnx_file.as_posix(),
        "save_external_data": True,
    }


T5_DEFAULT_QUANT_ARGS = ["--weightQuantBits=8", "--weightQuantBlock=128"]


def convert_models(onnx_root, mnn_root, convert_path, extra_args):
    onnx_root = Path(onnx_root)
    mnn_root = Path(mnn_root)
    print("Onnx path:", onnx_root)
    print("MNN path:", mnn_root)
    print("Extra:", " ".join(shlex.quote(x) for x in extra_args))
    records = []
    for model in WAN_MODELS:
        model_extra = list(extra_args)
        if model == "text_encoder" and not find_arg_value(extra_args, "--weightQuantBits"):
            model_extra.extend(T5_DEFAULT_QUANT_ARGS)
        record = convert_one(
            convert_path,
            onnx_root / model / "model.onnx",
            mnn_root / f"{model}.mnn",
            model_extra,
        )
        record["name"] = model
        records.append(record)
    return records


def copy_tokenizer_source(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer source directory not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)


def read_model_type(tokenizer_dir):
    for name in ("config.json", "tokenizer_config.json"):
        path = tokenizer_dir / name
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fp:
                model_type = json.load(fp).get("model_type")
            if model_type:
                return model_type
        except Exception:
            pass
    return "t5"


def materialize_tokenizer_json(tokenizer_dir):
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if tokenizer_json.exists():
        return
    from transformers import AutoTokenizer

    errors = []
    for use_fast in (True, False):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir.as_posix(),
                trust_remote_code=True,
                use_fast=use_fast,
                local_files_only=True,
            )
            tokenizer.save_pretrained(tokenizer_dir.as_posix())
            if tokenizer_json.exists():
                return
        except Exception as e:
            errors.append(f"use_fast={use_fast}: {e}")
    raise RuntimeError(
        f"Failed to materialize tokenizer.json under {tokenizer_dir}. "
        "Provide a complete local tokenizer directory. Errors: "
        + " | ".join(errors)
    )


def export_wan_mtok(tokenizer_src_root, mnn_root):
    repo_root = Path(__file__).resolve().parents[4]
    llm_export_dir = repo_root / "transformers" / "llm" / "export"
    if str(llm_export_dir) not in sys.path:
        sys.path.insert(0, str(llm_export_dir))

    from utils.tokenizer import LlmTokenizer

    dst_tokenizer = Path(mnn_root) / "tokenizer"
    copy_tokenizer_source(tokenizer_src_root, dst_tokenizer)
    materialize_tokenizer_json(dst_tokenizer)
    model_type = read_model_type(dst_tokenizer)
    llm_tokenizer = LlmTokenizer(dst_tokenizer.as_posix(), model_type)
    out_path = llm_tokenizer.export(
        dst_tokenizer.as_posix(),
        model_path=dst_tokenizer.as_posix(),
        model_type=model_type,
    )
    if not out_path.endswith("tokenizer.mtok"):
        raise RuntimeError(f"Tokenizer export did not produce tokenizer.mtok: {out_path}")
    print(f"Generated mtok: {out_path}")
    return {
        "model_type": model_type,
        "mtok_path": out_path,
        "source_dir": Path(tokenizer_src_root).resolve().as_posix(),
        "tokenizer_dir": dst_tokenizer.as_posix(),
    }


def write_convert_report(onnx_root, mnn_root, convert_path, extra_args, module_records, tokenizer_record):
    report = {
        "extra_args": extra_args,
        "mnn_root": Path(mnn_root).resolve().as_posix(),
        "mnnconvert_path": convert_path,
        "modules": [
            {
                "module": record["name"],
                "onnx_path": record["onnx_model"],
                "mnn_path": record["mnn_model"],
            }
            for record in module_records
        ],
        "mtok_path": tokenizer_record["mtok_path"],
        "onnx_root": Path(onnx_root).resolve().as_posix(),
        "saveExternalData": True,
        "tokenizer": {
            "model_type": tokenizer_record["model_type"],
            "source_path": tokenizer_record["source_dir"],
            "target_path": tokenizer_record["tokenizer_dir"],
        },
        "tokenizer_source_path": tokenizer_record["source_dir"],
        "tokenizer_target_path": tokenizer_record["tokenizer_dir"],
        "tokenizer_mtok_path": tokenizer_record["mtok_path"],
        "export_report": read_json_if_exists(Path(onnx_root) / "export_report.json"),
        "flags": build_flag_summary(extra_args),
    }
    write_json(Path(mnn_root) / "convert_report.json", report)


def main():
    args = parse_args()
    onnx_root = Path(args.onnx_path)
    mnn_root = Path(args.mnn_root)
    tokenizer_src = Path(args.tokenizer_path) if args.tokenizer_path else onnx_root / "tokenizer"
    convert_path = find_mnnconvert(args.mnnconvert)
    extra_args = normalize_extra(args.extra)
    module_records = convert_models(onnx_root, mnn_root, convert_path, extra_args)
    tokenizer_record = export_wan_mtok(tokenizer_src, mnn_root)
    write_convert_report(onnx_root, mnn_root, convert_path, extra_args, module_records, tokenizer_record)


if __name__ == "__main__":
    main()
