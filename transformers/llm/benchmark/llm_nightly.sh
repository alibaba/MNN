#!/bin/bash
set -e

# 1. Build MNN & PyMNN
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ numpy datasets modelscope lm_eval torch
pip install -r transformers/llm/export/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
echo ">>> Building PyMNN ..."
pushd pymnn/pip_package
rm -rf build/ dist/
python build_deps.py llm
python setup.py install --user
popd
pushd pymnn_build
make llm_bench
popd

# 2. Set Paths
CACHE_ROOT="/aoneci/runner/work/source/cache_dir"
THREAD_NUM=16
QWEN3_PATH="${CACHE_ROOT}/Qwen3-0.6B"
if [ ! -d "$QWEN3_PATH" ]; then
    modelscope download Qwen/Qwen3-0.6B --local_dir ${QWEN3_PATH}
fi

# 3. 准备评测缓存（包含 wikitext、arc_challenge、ceval）
echo ">>> Preparing Eval Cache..."
EVAL_CACHE_DIR="${CACHE_ROOT}/llm_eval_cache"
if [ ! -d "$EVAL_CACHE_DIR" ]; then
    modelscope download MNN/llm_eval_cache --local_dir ${EVAL_CACHE_DIR} --repo-type dataset
fi
# 解压缓存到 CACHE_ROOT 下的 HuggingFace datasets 目录
HF_CACHE_DIR="${CACHE_ROOT}/huggingface_datasets"
mkdir -p ${HF_CACHE_DIR}
tar -xzf ${EVAL_CACHE_DIR}/llm_eval_cache.tar.gz -C ${HF_CACHE_DIR}/
# 设置离线模式环境变量
export HF_DATASETS_CACHE=${HF_CACHE_DIR}
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers_cache"
export HF_DATASETS_OFFLINE=1
export HF_OFFLINE=1

# 4. Export Model
python transformers/llm/export/llmexport.py --path ${QWEN3_PATH} --export mnn --hqq
# change model/config.json thread num (Linux sed, 本地 macOS 测试请用 sed -i '')
sed -i "s/\"thread_num\": 4/\"thread_num\": ${THREAD_NUM}/" ./model/config.json || sed -i '' "s/\"thread_num\": 4/\"thread_num\": ${THREAD_NUM}/" ./model/config.json

# 5. Performance Test
echo ">>> Running Performance Benchmark for Qwen3-0.6B..."
./pymnn_build/llm_bench -m ./model/config.json -p 512 -n 128 -t ${THREAD_NUM} -j | tee llm_bench.log

# 6. PPL Test
echo ">>> Running PPL Test for Qwen3-0.6B on wikitext2..."
python transformers/llm/eval/evaluate_perplexity.py -m ./model/config.json | tee ppl_eval.log

# 7. Eval Test
echo ">>> Running Eval Test for Qwen3-0.6B..."
python transformers/llm/eval/llm_eval.py -m ./model/config.json -d arc_challenge,ceval-valid --limit 20

# 8. Report Summary to Aone CI
echo ">>> Nightly Test Report"

# 获取今天日期
TODAY=$(date +%Y-%m-%d)

# 获取系统信息
# ARM Linux cpuinfo 没有 model name，需要用 CPU part 代码映射
get_arm_cpu_name() {
    local part=$(grep -m1 "CPU part" /proc/cpuinfo 2>/dev/null | awk '{print $4}')
    case "$part" in
        0xd03) echo "Cortex-A53" ;;
        0xd04) echo "Cortex-A35" ;;
        0xd05) echo "Cortex-A55" ;;
        0xd07) echo "Cortex-A57" ;;
        0xd08) echo "Cortex-A72" ;;
        0xd09) echo "Cortex-A73" ;;
        0xd0a) echo "Cortex-A75" ;;
        0xd0b) echo "Cortex-A76" ;;
        0xd0c) echo "Neoverse-N1" ;;
        0xd0d) echo "Cortex-A77" ;;
        0xd40) echo "Neoverse-V1" ;;
        0xd41) echo "Cortex-A78" ;;
        0xd44) echo "Cortex-X1" ;;
        0xd46) echo "Cortex-A510" ;;
        0xd47) echo "Cortex-A710" ;;
        0xd48) echo "Cortex-X2" ;;
        0xd49) echo "Neoverse-N2" ;;
        0xd4a) echo "Neoverse-E1" ;;
        *) echo "ARM-$part" ;;
    esac
}
CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs)
if [ -z "$CPU_MODEL" ]; then
    CPU_MODEL=$(get_arm_cpu_name)
fi
CPU_CORES=$(nproc 2>/dev/null || echo 4)
MEMORY_GB=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 8)

# 提取 Prefill 速度和标准差
PREFILL_TPS=$(python3 -c "import json; data=json.load(open('llm_bench.json')); res=[x for x in data.get('results', []) if x.get('type') == 'prefill']; print(res[0].get('tps', 0) if res else 0)")
PREFILL_STD=$(python3 -c "import json; data=json.load(open('llm_bench.json')); res=[x for x in data.get('results', []) if x.get('type') == 'prefill']; print(res[0].get('std', 0) if res else 0)")
[ -z "$PREFILL_STD" ] && PREFILL_STD=0

# 提取 Decode 速度和标准差
DECODE_TPS=$(python3 -c "import json; data=json.load(open('llm_bench.json')); res=[x for x in data.get('results', []) if x.get('type') == 'decode']; print(res[0].get('tps', 0) if res else 0)")
DECODE_STD=$(python3 -c "import json; data=json.load(open('llm_bench.json')); res=[x for x in data.get('results', []) if x.get('type') == 'decode']; print(res[0].get('std', 0) if res else 0)")
[ -z "$DECODE_STD" ] && DECODE_STD=0

# 提取 PPL 数值
PPL_VALUE=$(grep "Perplexity" ppl_eval.log | awk '{print $2}')

# 提取 Eval 成绩 (从 results.json)
CEVAL_ACC=$(python3 -c "import json; res=json.load(open('results.json')); d=res['results'].get('ceval-valid', {}); print(d.get('acc,none') or d.get('acc', 0))")
ARC_ACC=$(python3 -c "import json; res=json.load(open('results.json')); d=res['results'].get('arc_challenge', {}); print(d.get('acc,none') or d.get('acc', 0))")

# 打印摘要
BENCH_SUMMARY="Prefill: ${PREFILL_TPS} ± ${PREFILL_STD} t/s, Decode: ${DECODE_TPS} ± ${DECODE_STD} t/s"
EVAL_SUMMARY="C-Eval: ${CEVAL_ACC}, ARC: ${ARC_ACC}"
echo "Performance: $BENCH_SUMMARY"
echo "Accuracy (PPL): $PPL_VALUE"
echo "Evaluation: $EVAL_SUMMARY"

# 9. Generate JSON Report
REPORT_FILE="${TODAY}.json"
cat > ${REPORT_FILE} << EOF
{
  "date": "${TODAY}",
  "suite": "nightly",
  "model": "Qwen3-0.6B",
  "environment": {
    "platform": "ARM Linux",
    "backend": "CPU",
    "thread_num": ${THREAD_NUM},
    "cpu_info": "${CPU_MODEL}",
    "cpu_cores": ${CPU_CORES},
    "memory_gb": ${MEMORY_GB}
  },
  "metrics": {
    "prefill": {
      "prompt_tokens": 512,
      "tokens_per_second": ${PREFILL_TPS},
      "std_dev": ${PREFILL_STD}
    },
    "decode": {
      "output_tokens": 128,
      "tokens_per_second": ${DECODE_TPS},
      "std_dev": ${DECODE_STD}
    },
    "perplexity": {
      "dataset": "wikitext2",
      "value": ${PPL_VALUE}
    },
    "evals": {
      "arc_challenge": { "acc": ${ARC_ACC} },
      "ceval-valid": { "acc": ${CEVAL_ACC} }
    }
  }
}
EOF

echo ">>> JSON report generated: ${REPORT_FILE}"
cat ${REPORT_FILE}

# 10. Sync to MNNBenchBoard Repo
echo ">>> Syncing results to MNNBenchBoard..."

git config --global user.email "mnn_ci@alibaba-inc.com"
git config --global user.name "MNN CI"

# 替换为你实际的仓库地址或本地路径
BENCHBOARD_REPO_URL="git@gitlab.alibaba-inc.com:AliNN/MNNBenchBoard.git"
BENCHBOARD_DIR="MNNBenchBoard_Sync"

# 如果目录不存在则克隆，存在则拉取最新代码
if [ ! -d "$BENCHBOARD_DIR" ]; then
    git clone "$BENCHBOARD_REPO_URL" "$BENCHBOARD_DIR"
fi

pushd "$BENCHBOARD_DIR"
git pull origin main

# 确保目录存在
mkdir -p static/data/nightly/

# 拷贝生成的 JSON 报告
cp "../${REPORT_FILE}" static/data/nightly/

# 提交并推送
git add static/data/nightly/"${REPORT_FILE}"
git commit -m "Update nightly benchmark for ${TODAY}"
git push origin main

popd
echo ">>> Successfully updated MNNBenchBoard with ${REPORT_FILE}"


# If in Aone CI environment, write to summary
if [ -n "$AONE_CI_SUMMARY" ]; then
    echo "TEST_CASE={\"name\":\"Qwen3-0.6B性能测试\", \"failed\":0, \"passed\":1, \"summary\":\"$BENCH_SUMMARY\"}" >> $AONE_CI_SUMMARY
    echo "TEST_CASE={\"name\":\"Qwen3-0.6B PPL测试\", \"failed\":0, \"passed\":1, \"summary\":\"$PPL_VALUE\"}" >> $AONE_CI_SUMMARY
    echo "TEST_CASE={\"name\":\"Qwen3-0.6B能力测评\", \"failed\":0, \"passed\":1, \"summary\":\"$EVAL_SUMMARY\"}" >> $AONE_CI_SUMMARY
fi