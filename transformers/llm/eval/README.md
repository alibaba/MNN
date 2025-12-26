# EVAL

用于评估和分析大语言模型（LLM）的性能。以下是各个脚本和目录的功能简介：

## 脚本说明

### evaluate_chat_ceval.py
  - **功能**
    用于评估聊天模型在中文教育评估（CEval）数据集上的表现。支持加载模型权重并对多个学科进行评估，生成详细的评估结果。
  - **参数**
    - `-m`：模型配置文件路径
    - `-d`：数据集名称
  - **示例**
    ```sh
    python evaluate_chat_ceval.py -m /path/to/model/config.json -d /path/to/ceval
    ```

### evaluate_perplexity.py
  - **功能**
    用于计算语言模型的困惑度（Perplexity），以衡量模型生成文本的质量。
  - **参数**
    - `-m`：模型配置文件路径
    - `-d`：数据集名称
  - **示例**
    ```sh
    python evaluate_perplexity.py -m /path/to/model/config.json -d "wikitext/wikitext-2-raw-v1"
    ```

### llm_eval.py
  - **功能**
    提供通用的语言模型评估功能，支持多种任务和数据集。
  - **参数**
    - `-m`：模型配置文件路径
    - `-d`：数据集名称
  - **示例**
    ```sh
    pip install lm_eval
    python llm_eval.py -m /path/to/model/config.json -d "arc_challenge"
    ```

### download_data.py
  - **功能**
    下载数据集，以便纯C++环境下的评测工具，如`ppl_eval`使用
  - **参数**
    - `-o`：目标目录
    - `-d`：数据集名称
  - **示例**
    ```sh
    python download_data.py -o wiki -d "wikitext/wikitext-2-raw-v1"
    ```

### `ppl_eval`
  - **功能**
    与`evaluate_perplexity.py`相似，计算ppl值，但支持纯C++环境使用
  - **参数**
    - config.json
    - 数据集目录（`download_data.py`的目标目录）
  - **示例**
    ```sh
    ./ppl_eval ../transformers/llm/export/model/config.json ../transformers/llm/eval/wiki
