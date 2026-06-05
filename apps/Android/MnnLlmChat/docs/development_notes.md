# Development Notes

This document records engineering rules and recurring pitfalls for the MnnLlmChat Android Agent work.

Core technical source for the Agent port: `https://github.com/huangzhengxiang/ActMe.git`

## Non-Negotiable Development Rules

- Keep normal chat and Agent chat as separate session modes. A conversation's mode is chosen when the session is created and is persisted as `Session.sessionMode`.
- Do not route normal chat through the Agent loop. Normal chat must keep the original single-pass path.
- Do not rebuild the prompt from database history on every local ChatActivity turn. The live native `LlmSession` owns active context and KV/prompt-cache state.
- Do not run Gradle builds from automation unless explicitly requested. Many dependency and native build steps are slow or environment-sensitive.
- Prefer small, isolated changes. Agent behavior touches prompt design, parser fallback, tool execution, UI status, file handoff, and native session behavior.

## Session and History

There are two different concepts of history:

- UI/database history: `ChatDataManager.getChatDataBySession(...)`, used for visible conversation history and history list restore.
- Native runtime context: `LlmSession.history_`, held by the active native MNN session and used for generation.

Important implications:

- For an active live session, each new user turn should submit only the new input. The native session already has previous context.
- When reopening a historical session after native release, visible database history must be loaded into `LlmSession.savedHistory`.
- Cold restore may pass a bounded, alternating user/assistant history list into native initialization so the model has conversational context again, but this is not the same as persisted KV cache.
- Prompt cache is currently in-memory. Persisting prompt/KV cache by `sessionId` requires explicit native save/load support and should be treated as a separate feature.

## Agent Prompt Contract

`AgenticPrompts.kt` is a runtime API contract, not only prompt text. Any change here must be checked against:

- `AgenticProtocol.kt`
- `AgenticOutputParser.kt`
- `AgenticToolExecutor.kt`
- existing logcat examples from real local models

Current tool calls should use:

```json
{
  "reply": "",
  "memory_updates": [],
  "skill_updates": [],
  "system_calls": [
    {
      "type": "python_exec",
      "code": "write_excel('sample.xlsx', {'Sheet1': [['Name'], ['Alice']]})",
      "input": "",
      "timeout_ms": 15000,
      "output_files": ["sample.xlsx"]
    }
  ]
}
```

Do not add decorative fields unless the executor uses them. The `reason` field was intentionally removed from the advertised protocol because local models tended to overfit to it and it did not improve execution.

## Parser Robustness

Local models often produce imperfect JSON. `AgenticOutputParser.kt` should keep accepting:

- strict JSON object
- fenced JSON blocks
- JSON embedded in normal text
- single tool object, such as `{"type":"python_exec","code":"..."}`
- `system_calls` as either an array or a single object
- string arrays for `memory_updates` or `skill_updates`
- truncated or malformed `python_exec` output when `type` and `code` can still be extracted safely

The parser should prefer recovering tool calls over failing the whole response because a secondary field is malformed.

## Python Sandbox

Python execution is implemented through Chaquopy and `app/src/main/python/agent_python.py`.

When adding or removing packages in `app/build.gradle`:

- update `ALLOWED_IMPORT_ROOTS` in `agent_python.py`
- update `AgenticPrompts.kt`
- update `docs/builtin_python_design.md`
- verify common transitive import roots if users are likely to import them directly

Current installed packages:

- `openpyxl`
- `numpy`
- `pandas`

Current user-visible allowlist should include those package roots and common runtime dependencies such as:

- `dateutil`
- `et_xmlfile`
- `pytz`
- `tzdata`

Security boundaries:

- Keep `os`, `sys`, `subprocess`, `socket`, `shutil`, `pathlib`, `multiprocessing`, and `ctypes` denied for user code.
- File access must stay inside `AgenticPythonEngine.workspaceDir(...)`.
- Relative file paths should resolve inside the agent workspace.
- Prefer `write_excel(filename, sheets)` for simple workbook creation because it handles Android file handoff cleanly and starts faster than pandas.

## Generated Files

Generated files are returned through `ChatFileAttachment`.

The attachment pipeline has three sources:

- Python runner snapshot diff: files created or modified inside the workspace.
- Declared output references in the tool call: `output_files`, `generated_files`, `expected_outputs`, or `files`.
- Final answer references: workspace file names or `file://...` links found in the final text.

Path normalization must never expose arbitrary filesystem paths. A file reference should only become an attachment if it resolves inside the agent workspace, or if a matching basename is found inside that workspace.

## Browser and Search

The current MnnLlmChat port uses lightweight Bing HTML search and HTTP page text extraction.

Development caveats:

- Browser/search behavior is network- and region-sensitive.
- Search result parsing must be debugged with the actual request URL and returned HTML shape.
- Dynamic pages, CAPTCHA, login walls, and heavy JavaScript pages may need a future GeckoView-backed implementation.
- Do not claim full browser parity with ActMe until rendered browsing and robust page text extraction are ported.

## UI and Visibility

Agent mode must remain visible, controllable, resumable where possible, and interruptible:

- show tool step progress in the assistant message/process area
- preserve user stop behavior
- avoid exposing raw JSON dictionaries as final answers
- return generated files as clickable attachments
- keep workspace browsing accessible from the history drawer

For orientation, main chat flows should remain portrait unless a feature explicitly needs another mode. Video/debug/scanner flows may keep their own behavior.

## Common Failure Patterns

- Model emits JSON in a fenced code block: parser should strip fences.
- Model emits a single tool object instead of full `system_calls`: parser should wrap it.
- Model emits memory/skill string arrays: parser should not reject the tool call.
- Model says it created a file but did not run code: prompt should tell Python code to directly perform the action.
- Model defines a Python function but does not call it: this is a model behavior issue; prompt and examples should avoid function-only snippets for simple tasks.
- Python package is installed but blocked by sandbox: update both Chaquopy dependencies and `ALLOWED_IMPORT_ROOTS`.
- File link uses a fake path like `file:///samples/a.xlsx`: normalize to workspace path and fallback to basename search.
- Reopened Agent session has no visible history: ensure DB history is passed into `LlmSession.savedHistory`.
- Reopened Agent session has no context: ensure cold native restore receives bounded alternating user/assistant history.

## Testing Without Full Builds

When avoiding Gradle builds, use static checks:

- `rg` for field names and protocol drift.
- inspect manifest changes directly.
- inspect Kotlin call sites after changing data classes.
- inspect Python sandbox allowlist after changing Chaquopy packages.
- use logcat examples to validate parser fallbacks conceptually.

Full validation still requires a device install and real local-model trials because local model formatting behavior is model-specific.
