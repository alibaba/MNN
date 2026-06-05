# Built-in Python Design

This document describes the Python capability in `MnnLlmChat`.

Status: implemented for Agent mode through `python_exec`, Chaquopy, and the ActMe-derived `agent_python.py` sandbox.

Core technical source: `https://github.com/huangzhengxiang/ActMe.git`

## Purpose

LLMs are weak at deterministic computation and file transformations. A local Python runtime gives the agent a reliable path for:

- arithmetic and date calculations
- CSV/Excel-style table processing
- JSON cleanup
- HTML/text parsing
- generating files for the user
- checking Python syntax before execution

## Tool Set

The current MnnLlmChat port exposes one Agent tool:

- `python_exec`: run bounded Python code with declared input files and expected outputs.

Syntax validation is available inside `python_exec` through `compile_script(name)` after the agent stores code with `save_script(name, source)`.

Bundled packages:

- `openpyxl`
- `numpy`
- `pandas`

`numpy` and `pandas` are native/data Python packages. They are installed at build time through Chaquopy's Android-compatible package support, which increases APK size and may increase build time.

The sandbox import allowlist must stay aligned with installed packages. Current user-visible import roots include:

- `openpyxl`
- `numpy`
- `pandas`
- `dateutil`
- `et_xmlfile`
- `pytz`
- `tzdata`

## Execution Boundaries

The runtime should be local, temporary, and bounded:

- per-run timeout
- stdout/stderr size limit
- working directory scoped to the current session
- generated files returned through chat attachments
- no background daemon in the first version

## Agent Contract

Execute input:

```json
{
  "type": "python_exec",
  "code": "write_excel('sample.xlsx', {'Sheet1': [['Name'], ['Alice']]})",
  "input_files": [],
  "output_files": ["sample.xlsx"]
}
```

Observation:

```json
{
  "type": "python_exec",
  "status": "ok",
  "stdout": "{\"ok\": true}",
  "stderr": "",
  "output_files": []
}
```

## Excel and File Work

Excel support should be implemented as Python packages plus Android file handoff:

- Android registers Excel MIME types and imports the file into a chat session.
- The agent receives the local file path as an attachment observation.
- Python reads the workbook, summarizes sheets, performs analysis, and can generate a new workbook.
- Generated files are returned as chat attachments.

This keeps spreadsheet logic in Python instead of duplicating it in Kotlin.

Agents may use `pandas` for larger table operations, but should still prefer the built-in helpers `read_excel(path)` and `write_excel(filename, sheets)` for Android file handoff. For small tables, standard-library modules such as `csv`, `json`, `statistics`, and `collections` are often faster to start and easier to sandbox.

Generated files can be detected from workspace changes or declared by the model through `output_files`, `generated_files`, `expected_outputs`, or `files`. Path normalization is intentionally workspace-scoped: fake absolute paths or `file://` links are only returned when they can be matched back to files inside the agent workspace.
