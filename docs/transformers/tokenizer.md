# Tokenizer

MNN LLM 的 Tokenizer 负责文本编码（encode）和解码（decode），是 LLM 推理的核心组件。当前支持两种文件格式：

- **`.mtok` 二进制格式**（推荐）：Pipeline Tokenizer，高性能二进制格式
- **`.txt` 文本格式**（旧版）：兼容早期导出的模型

## .mtok 文件格式

`.mtok` 是 MNN 自定义的二进制 Tokenizer 格式，采用「Python 导出时预计算 + C++ 加载时零拷贝」的设计，具有加载快、编码效率高、无外部依赖等优点。

### 文件结构总览

```
┌─────────────────────────────────────┐
│           Text Header               │  ← 文本行，与旧格式兼容
│  Line 1: "430 4\n"                  │
│  Line 2: "special stop prefix\n"    │
│  Line 3: "id1 id2 ... \n"           │
├─────────────────────────────────────┤
│           Binary Body               │  ← 二进制数据，顺序读取
│  ┌─────────────────────────────┐    │
│  │  Normalizer                 │    │
│  │  (可含 NFKC/NFD 归一化表)     │    │
│  ├─────────────────────────────┤    │
│  │  PreTokenizer               │    │
│  ├─────────────────────────────┤    │
│  │  Model (BPE/WP/Unigram)     │    │
│  ├─────────────────────────────┤    │
│  │  Decoder                    │    │
│  ├─────────────────────────────┤    │
│  │  Added Tokens               │    │
│  ├─────────────────────────────┤    │
│  │  Chat Template  (optional)  │    │
│  ├─────────────────────────────┤    │
│  │  EOS Token      (optional)  │    │
│  ├─────────────────────────────┤    │
│  │  Flags          (optional)  │    │
│  ├─────────────────────────────┤    │
│  │  BOS Token      (optional)  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

### Text Header

前 3 行为文本格式，与旧版 `.txt` 格式的 Header 结构相同：

| 行 | 内容 | 说明 |
|----|------|------|
| 1 | `430 4` | magic number (430) + tokenizer type (4=PIPELINE) |
| 2 | `{special_num} {stop_num} {prefix_num}` | 各类特殊 token 的数量 |
| 3 | `{id1} {id2} ...` | 所有特殊、stop、prefix token 的 ID |

### 基础二进制类型

所有整数均为**小端序 (little-endian)**，字符串格式为 `[uint16 len][bytes]`：

```
u8     = 1 字节无符号整数
u16    = 2 字节无符号整数 (little-endian)
u32    = 4 字节无符号整数 (little-endian)
f64    = 8 字节双精度浮点 (IEEE 754)
str    = [u16 len][len bytes]              // 变长字符串
str_ref = [u16 len][len bytes]             // 同 str，C++ 端用 StringRef 零拷贝引用
```

### Normalizer

文本归一化器，对输入文本进行预处理（大小写、Unicode 归一化等）。

```
[u8 type]
  0 = None          // 无归一化
  1 = NFKC          // NFKC 归一化（无内嵌表，使用旧路径）
  2 = Prepend       // 前缀添加
       [str prepend_str]
  3 = Replace        // 字符串替换
       [str pattern] [str content]
  4 = Sequence       // 多个归一化器组合
       [u32 count] [Normalizer × count]
  5 = BertNormalizer // BERT 风格归一化
       [u8 clean] [u8 handle_chinese] [u8 strip_accents] [u8 lowercase]
  6 = NFKC+Table     // NFKC 归一化（内嵌预计算表）
       [NormTable]
  7 = BertNormalizer+NFD  // BERT + NFD 归一化
       [u8 clean] [u8 handle_chinese] [u8 strip_accents] [u8 lowercase]
       if strip_accents: [NormTable]   // NFD 归一化表
```

#### Unicode 归一化表 (NormTable)

`.mtok` 将 Unicode 归一化所需的映射表直接内嵌到文件中，使得 C++ 端无需依赖 ICU 等外部 Unicode 库。

**表结构：**

```
[u32 count]                     // 条目数量
For each entry:
  [u32 codepoint]               // 原始 Unicode 码点
  [u16 utf8_len]                // 归一化后的 UTF-8 字节长度
  [utf8_len bytes]              // 归一化后的 UTF-8 字节序列
```

**生成方式（Python 导出时预计算）：**

导出时遍历整个 Unicode 码点空间（U+0000 ~ U+10FFFF），对每个码点执行归一化，仅记录归一化结果与原始字符不同的条目：

```python
# NFKC 表：用于 SentencePiece 等模型的归一化
for cp in range(0x110000):
    ch = chr(cp)
    normalized = unicodedata.normalize('NFKC', ch)
    if normalized != ch:
        entries.append((cp, normalized.encode('utf-8')))

# NFD 表：用于 BERT 的 strip_accents（去除变音符号）
for cp in range(0x110000):
    ch = chr(cp)
    decomposed = unicodedata.normalize('NFD', ch)
    if decomposed != ch:
        entries.append((cp, decomposed.encode('utf-8')))
```

**C++ 端查找（二分查找）：**

表按 codepoint 升序排列，C++ 加载后通过二分查找完成归一化，单次查找 O(log n)：

```cpp
size_t lo = 0, hi = table_.size();
while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if ((uint32_t)cp > table_[mid].first) lo = mid + 1;
    else if ((uint32_t)cp < table_[mid].first) hi = mid;
    else { /* 命中，返回归一化结果 */ break; }
}
```

**典型场景：**
- **NFKC 表** (type=6)：将兼容字符映射到标准形式，如 `ﬁ` → `fi`，`①` → `1`。用于 SentencePiece (Llama/Gemma) 等模型
- **NFD 表** (type=7)：将字符分解为基字符 + 组合字符，如 `é` → `e` + `◌́`。用于 BERT 的 `strip_accents` 功能，分解后去除组合标记（category `Mn`）实现去变音符号

### PreTokenizer

预分词器，将输入文本拆分为子串，再分别送入 Model 进行编码。

```
[u8 type]
  0 = None
  1 = ByteLevel      // GPT-2 / Llama3 风格
       [u8 use_regex]  // 是否使用正则/scanner 拆分
  2 = Digits          // 数字拆分
       [u8 individual_digits]  // 是否逐位拆分
  3 = Metaspace       // SentencePiece 风格
       [str replacement] [u8 add_prefix_space]
  4 = Split           // 自定义正则拆分
       [str pattern] [u8 invert] [u8 behavior]
  5 = BertPreTokenizer  // BERT 风格（空格+标点拆分）
  6 = Sequence          // 多个预分词器组合
       [u32 count] [PreTokenizer × count]
```

### Model

核心分词模型，支持三种算法：

#### BPE (type=0)

```
[u8 type=0]
[u32 vocab_size]
[u8 byte_fallback]      // 是否支持 byte fallback
[u8 byte_level]         // 是否为 byte-level BPE
[u32 merge_size]        // merge 规则数量

// 词表（按 token 字符串字典序预排序，支持二分查找）
[str_ref token, u32 id] × vocab_size

// Merge 规则（按 key=(id1<<32)|id2 预排序，支持二分查找）
[u32 id1, u32 id2, u32 rank] × merge_size
```

#### WordPiece (type=1)

```
[u8 type=1]
[u32 vocab_size]
[str unk_token]                  // 未知 token，如 "[UNK]"
[str continuing_subword_prefix]  // 子词前缀，如 "##"
[u32 max_chars]                  // 单词最大字符数

// 词表（按 token 字符串字典序预排序）
[str_ref token, u32 id] × vocab_size
```

#### Unigram (type=2)

```
[u8 type=2]
[u32 vocab_size]
[u32 unk_id]          // 未知 token ID
[u8 byte_fallback]    // 是否支持 byte fallback

// 词表（按 token 字符串字典序预排序）
[str_ref token, u32 id, f64 score] × vocab_size
```

### Decoder

解码器，将 token ID 序列还原为文本。

```
[u8 type]
  0 = ByteLevel     // byte → unicode 反映射
  1 = ByteFallback  // byte token 还原
  2 = Metaspace     // '▁' → 空格
       [str replacement] [u8 add_prefix_space]
  3 = WordPiece     // 移除 "##" 前缀
       [str prefix] [u8 cleanup]
  4 = Fuse          // 拼接所有 token
  5 = Replace       // 字符串替换
       [str pattern] [str content]
  6 = Strip         // 去除首尾字符
       [str content] [u32 start] [u32 stop]
  7 = Sequence      // 多个解码器组合
       [u32 count] [Decoder × count]
```

### Added Tokens

模型额外添加的特殊 token（如 `<|im_start|>`、`<tool_call>` 等）。

```
[u32 count]
For each token:
  [u32 id]
  [u8 special]   // 是否为特殊 token
  [u8 lstrip]    // 匹配时是否去左空格
  [u8 rstrip]    // 匹配时是否去右空格
  [str content]  // token 文本
```

### Chat Template（可选）

内嵌 Jinja2 chat template，加载后无需额外读取 `tokenizer_config.json`。

```
[u32 tpl_len]
[tpl_len bytes]   // Jinja2 模板 UTF-8 文本

[u16 eos_len]
[eos_len bytes]   // EOS token 文本

[u8 flags]        // bit0: clean_up_tokenization_spaces

[u16 bos_len]
[bos_len bytes]   // BOS token 文本
```

## Chat Template 支持

MNN Tokenizer 内置了 Jinja2 模板引擎，用于将多轮对话格式化为模型所需的 prompt 格式。

### 模板来源

| 格式 | 模板来源 |
|------|---------|
| `.mtok` | 内嵌在文件尾部，从 `tokenizer_config.json` 中提取并写入 |
| `.txt` | 从 `llm_config.json` 的 `jinja.chat_template` 字段读取 |

### 使用方式

```cpp
// 简单对话
std::string prompt = tokenizer->apply_chat_template("你好");

// 多轮对话
ChatMessages messages = {
    {"system", "You are a helpful assistant."},
    {"user", "Hello!"},
    {"assistant", "Hi there!"},
    {"user", "What's the weather?"}
};
std::string prompt = tokenizer->apply_chat_template(messages);
```

### 模板语法

支持 Jinja2 核心语法子集：

```jinja
{# 变量 #}
{{ messages }}
{{ eos_token }}
{{ bos_token }}

{# 控制流 #}
{% for message in messages %}
  {% if message.role == 'user' %}
    <|im_start|>user\n{{ message.content }}<|im_end|>\n
  {% endif %}
{% endfor %}

{% if add_generation_prompt %}
  <|im_start|>assistant\n
{% endif %}

{# 过滤器 #}
{{ value | tojson }}
{{ value | trim }}
{{ value | length }}
{{ value | upper }}

{# 循环变量 #}
{{ loop.index }}     {# 1-based 序号 #}
{{ loop.first }}     {# 是否第一次迭代 #}
{{ loop.last }}      {# 是否最后一次迭代 #}
```

### 模板上下文变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `messages` | array | 消息数组，每项包含 `role` 和 `content` |
| `add_generation_prompt` | bool | 是否在末尾添加 assistant 提示 |
| `eos_token` | string | EOS 特殊 token 文本 |
| `bos_token` | string | BOS 特殊 token 文本 |
| `tools` | array | 工具定义（可选，用于 function calling） |

---

## 旧版 tokenizer.txt 文件格式

旧版采用纯文本格式，以 base64 编码存储 token。支持 4 种 tokenizer 类型。

### 文件结构

```
┌─────────────────────────────────────┐
│  Line 1: "430 {type}"               │  ← type: 0/1/2/3
│  Line 2: "{special} {stop} {pfx}"   │
│  Line 3: "id1 id2 ..."              │
├─────────────────────────────────────┤
│  Vocab Section (type-specific)      │
│  ...                                │
└─────────────────────────────────────┘
```

### Type 0: SENTENCEPIECE

适用于 Llama、Baichuan、ChatGLM 等使用 SentencePiece 的模型。

```
Line 4: "{vocab_size}"

Lines 5 ~ 4+vocab_size:
  "{base64_token} {score} {type}"

  score: float，Unigram 概率分数
  type:  1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED, 6=BYTE
```

示例：
```
430 0
5 2 1
0 1 2 3 4 100 101 151643
32000
PHM+ -1000 3
PA== -1000 2
PGJ5dGVfMHgwMD4= 0 6
...
```

### Type 1: TIKTOKEN

适用于 Qwen、GPT-4 等使用 Tiktoken 的模型。

```
Line 4: "{vocab_size}"

Lines 5 ~ 4+vocab_size:
  "{base64_token}"

  按行序排列，行号即 token ID
```

### Type 2: BERT

适用于 BERT、MiniLM 等模型。

```
Line 4: "{vocab_size}"

Lines 5 ~ 4+vocab_size:
  "{base64_token}"

  按行序排列，行号即 token ID
  使用 WordPiece 算法编码，"##" 前缀表示子词
```

### Type 3: HUGGINGFACE

适用于使用 HuggingFace Tokenizers 库的模型（BPE 算法）。

```
Line 4: "{vocab_size} {merge_size}"

Lines 5 ~ 4+vocab_size:
  "{token}"                          // 未经 base64 编码的原始 token

Lines 5+vocab_size ~ 4+vocab_size+merge_size:
  "{token1} {token2}"               // BPE merge 规则，按优先级排列
```

### 类型对比

| 特性 | `.mtok` (type=4) | `.txt` (type=0-3) |
|------|-------------------|-------------------|
| 文件格式 | 二进制 | 纯文本 |
| 词表存储 | 预排序数组 + 零拷贝 | `unordered_map<string, int>` |
| 词表查找 | 二分查找 O(log n) | 哈希表 O(1)~O(n) |
| Merge 规则 | 预计算 uint64 key + 二分 | wstring pair 哈希表 |
| 预分词 | 手写 Unicode scanner | `std::regex` |
| Unicode 归一化 | 预计算表内嵌文件，无外部依赖 | 运行时依赖 Unicode 库或不支持 |
| Chat Template | 内嵌文件 | 需从 config 读取 |
| 文件 I/O | 单次读取，零拷贝 | 逐行解析 |
| 字符串分配 | StringRef 指向 buffer | 每个 token 分配 string |

### 导出方式

```bash
# 导出模型（自动选择格式，优先 .mtok）
cd transformers/llm/export
python llmexport.py --path /path/to/model --export mnn

# 产物:
#   有 tokenizer.json → 导出 tokenizer.mtok
#   无 tokenizer.json → 导出 tokenizer.txt
```
