#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

BASE_MODEL="${1:-${HOME}/workspace/models/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${2:-${REPO_ROOT}/build/multi_lora_sample}"
DEVICE="${3:-auto}"
PYTHON_BIN="${MNN_MULTI_LORA_PYTHON:-python3}"
MAX_STEPS="${MNN_MULTI_LORA_MAX_STEPS:-80}"
MNN_CONVERT="${MNN_MULTI_LORA_CONVERT:-${REPO_ROOT}/build/MNNConvert}"
export HF_HOME="${MNN_MULTI_LORA_HF_HOME:-${OUTPUT_DIR}/hf_cache}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

ADAPTER_ALPHA="${OUTPUT_DIR}/adapter_alpha"
ADAPTER_BETA="${OUTPUT_DIR}/adapter_beta"
MNN_ALPHA_EXPORT="${OUTPUT_DIR}/mnn_alpha_export"
MNN_BETA_EXPORT="${OUTPUT_DIR}/mnn_beta_export"
MNN_MODEL="${OUTPUT_DIR}/mnn_multi_lora"
DATA_FILE="${SCRIPT_DIR}/data.jsonl"
DATA_DIR="${OUTPUT_DIR}/data"

prepare_data() {
    local source_data="$1"
    local data_dir="$2"

    "${PYTHON_BIN}" - "${source_data}" "${data_dir}" <<'PY'
import json
import sys
from pathlib import Path


source = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
data_types = ("alpha_train", "alpha_eval", "beta_train", "beta_eval")
records = {data_type: [] for data_type in data_types}

if not source.is_file():
    raise SystemExit(f"Data file not found: {source}")

with source.open("r", encoding="utf-8") as input_file:
    for line_number, line in enumerate(input_file, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(record, dict):
            raise SystemExit(f"{source}:{line_number}: each row must be a JSON object")

        data_type = record.pop("type", None)
        if data_type not in records:
            allowed = ", ".join(data_types)
            raise SystemExit(
                f"{source}:{line_number}: invalid type {data_type!r}; expected one of: {allowed}"
            )
        if "messages" not in record:
            raise SystemExit(f"{source}:{line_number}: missing messages")
        records[data_type].append(record)

missing_types = [data_type for data_type, items in records.items() if not items]
if missing_types:
    raise SystemExit(f"{source}: no rows found for: {', '.join(missing_types)}")

output_dir.mkdir(parents=True, exist_ok=True)
for data_type, items in records.items():
    output_path = output_dir / f"{data_type}.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in items:
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output_file.write("\n")
    print(f"Prepared {len(items)} rows: {output_path}")
PY
}

train_adapter() {
    local train_data="$1"
    local eval_data="$2"
    local adapter_dir="$3"

    "${PYTHON_BIN}" "${REPO_ROOT}/transformers/llm/finetune/mnn_qlora.py" \
        --base_model "${BASE_MODEL}" \
        --train_data "${train_data}" \
        --validation_data "${eval_data}" \
        --output_dir "${adapter_dir}" \
        --quant_bit 4 \
        --quant_block 64 \
        --lm_quant_bit 4 \
        --lm_quant_block 64 \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --max_seq_len 96 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-3 \
        --warmup_steps 0 \
        --lr_scheduler_type constant \
        --max_steps "${MAX_STEPS}" \
        --logging_steps 10 \
        --save_steps 0 \
        --dtype auto \
        --device "${DEVICE}"
}

eval_adapter() {
    local eval_data="$1"
    local adapter_dir="$2"

    "${PYTHON_BIN}" "${REPO_ROOT}/transformers/llm/finetune/eval_lora_effect.py" \
        --base_model "${BASE_MODEL}" \
        --adapter_path "${adapter_dir}" \
        --eval_data "${eval_data}" \
        --fake_quant \
        --quant_bit 4 \
        --quant_block 64 \
        --lm_quant_bit 4 \
        --lm_quant_block 64 \
        --max_new_tokens 12 \
        --dtype auto \
        --device "${DEVICE}"
}

export_adapter() {
    local adapter_dir="$1"
    local export_dir="$2"

    (
        cd "${REPO_ROOT}/transformers/llm/export"
        "${PYTHON_BIN}" llmexport.py \
            --path "${BASE_MODEL}" \
            --lora_path "${adapter_dir}" \
            --lora_split \
            --export mnn \
            --quant_bit 4 \
            --quant_block 64 \
            --lm_quant_bit 4 \
            --lm_quant_block 64 \
            --mnnconvert "${MNN_CONVERT}" \
            --dst_path "${export_dir}"
    )
}

if [[ ! -f "${BASE_MODEL}/config.json" ]]; then
    echo "Base model not found: ${BASE_MODEL}" >&2
    exit 2
fi
if [[ ! -x "${MNN_CONVERT}" ]]; then
    echo "MNNConvert not found or not executable: ${MNN_CONVERT}" >&2
    echo "Build it first with MNN_BUILD_CONVERTER=ON." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
prepare_data "${DATA_FILE}" "${DATA_DIR}"

train_adapter "${DATA_DIR}/alpha_train.jsonl" "${DATA_DIR}/alpha_eval.jsonl" "${ADAPTER_ALPHA}"
train_adapter "${DATA_DIR}/beta_train.jsonl" "${DATA_DIR}/beta_eval.jsonl" "${ADAPTER_BETA}"

eval_adapter "${DATA_DIR}/alpha_eval.jsonl" "${ADAPTER_ALPHA}"
eval_adapter "${DATA_DIR}/beta_eval.jsonl" "${ADAPTER_BETA}"

export_adapter "${ADAPTER_ALPHA}" "${MNN_ALPHA_EXPORT}"
export_adapter "${ADAPTER_BETA}" "${MNN_BETA_EXPORT}"

rm -rf "${MNN_MODEL}"
mkdir -p "${MNN_MODEL}"
cp -R "${MNN_ALPHA_EXPORT}/." "${MNN_MODEL}/"
mv "${MNN_MODEL}/lora.mnn" "${MNN_MODEL}/lora_alpha.mnn"
cp "${MNN_BETA_EXPORT}/lora.mnn" "${MNN_MODEL}/lora_beta.mnn"

echo
echo "Multi-LoRA model assembled at: ${MNN_MODEL}"
echo "Run:"
echo "  ${REPO_ROOT}/build/multi_lora_demo \\"
echo "    ${MNN_MODEL}/config.json \\"
echo "    lora_alpha.mnn '<<ALPHA>>' \\"
echo "    lora_beta.mnn '[[BETA]]'"
