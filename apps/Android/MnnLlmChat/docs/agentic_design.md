# Agentic Design for MnnLlmChat

This document describes the lightweight Agent mode added to `MnnLlmChat`.

核心技术来源 / Core technical source: `https://github.com/huangzhengxiang/ActMe.git`

The design is adapted from ActMe's mobile agent architecture, especially:

- JSON-based `system_calls`.
- A multi-pass agentic loop.
- Visible tool execution steps.
- Tool-call parsing fallbacks.
- Conservative mobile budgets and cancellation behavior.

MnnLlmChat remains an on-device local-model app. Agent mode does not change the model runtime into a cloud model. It lets the host app execute a small set of tools requested by the local model.

## Conversation Modes

Normal chat and Agent chat are session-level modes.

The mode is selected when a conversation is created:

- `normal`: use the original single-pass chat path.
- `agent`: use the agentic loop.

The selected mode is persisted in the `Session` table as `sessionMode`.

```text
normal session -> ChatPresenter.submitLlmRequest(...)
agent session  -> ChatPresenter.submitAgenticLlmRequest(...)
```

The mode is not a temporary UI toggle. Reopening a history item restores that session's own mode.

## Backward Compatibility

Database compatibility is handled by `ChatDatabaseHelper` and `ChatDataManager`.

Schema change:

```text
DB_VERSION: 8 -> 9
Session.sessionMode TEXT DEFAULT 'normal'
```

Upgrade behavior:

- Existing databases are migrated with `ALTER TABLE Session ADD COLUMN sessionMode TEXT DEFAULT 'normal'`.
- `ChatDataManager.ensureSessionModeColumn()` also checks and adds the column defensively.
- Old conversations without `sessionMode` are treated as `normal`.
- History queries fall back to older column sets when needed.

This supports upgrading from older app versions. Downgrading from a version that has opened DB v9 back to an older APK is not guaranteed, because the old code does not know DB v9.

## Current Tool Scope

The current MnnLlmChat port intentionally exposes only tools that are implemented in this app:

- `get_current_time`
- `web_search`
- `browser_url`
- `python_exec`

Python, Skill, and Memory are active in this MnnLlmChat port. ADB remains part of the fuller ActMe design and is intentionally not advertised here.

## Loop Shape

The Agent mode loop is:

```text
user message
-> local model planning pass
-> parse JSON/system_calls
-> execute tools visibly
-> append observations to continuation input
-> local model continuation pass
-> repeat until final reply, stop request, or budget exhaustion
```

Tool steps are shown in the assistant message while the run is active:

```text
[Agent] 规划中...
[Agent] 第 1 轮计划：1 个工具调用
[Agent] 联网搜索：中国银行 积存金 价格
[Agent] 联网搜索完成：搜索完成，...
[Agent] 已获得工具结果，继续推理...
```

When the run finishes, the visible message body is replaced by the final answer. The step log is kept in the thinking/process area.

## KV Cache and Prompt Cache Semantics

MnnLlmChat uses the native MNN `LlmSession` as the source of truth for live conversation context.

For the ChatActivity local-chat path:

- Each user turn submits only the newly arrived input to `LlmSession.generate(...)`.
- Database chat history is used for UI/history display, not for reconstructing the prompt on every turn.
- The native session keeps `keep_history=true`, maintains its in-memory `history_`, calls `llm_->response(history_, ...)`, and then calls `llm_->syncPromptCache(history_)`.
- As long as the same native `llm_` instance stays alive, MNN can reuse the in-memory prompt/KV cache for already processed context.

Reloading a session is different:

- Releasing or reloading creates a new native `llm_` instance.
- This app does not currently persist prompt cache or KV cache to disk.
- Reopening an old conversation restores visible database history and may pass a bounded alternating user/assistant history list into native initialization for useful conversational context.
- This cold restore is not the same as restoring the old native prompt cache.
- If persistent prompt-cache restore is required, it needs explicit native support for saving and loading cache state by `sessionId`.

The API server compatibility path is separate. OpenAI/Anthropic-style stateless calls may still use `submitFullHistory(...)` because those requests carry complete message history by protocol design.

## Prompt Contract

`AgenticPrompts.kt` tells the model:

- It is running inside MNN Chat.
- The current model is local/on-device.
- The local model itself cannot directly access the network.
- The host app can execute tools requested through `system_calls`.
- In Agent mode, search/browse requests should produce tool calls instead of refusal text.

Expected output shape:

```json
{
  "reply": "",
  "memory_updates": [],
  "skill_updates": [],
  "system_calls": [
    {
      "type": "web_search",
      "query": "query"
    },
    {
      "type": "python_exec",
      "code": "write_excel('sample.xlsx', {'Sheet1': [['Name'], ['Alice']]})",
      "timeout_ms": 15000,
      "output_files": ["sample.xlsx"]
    }
  ]
}
```

For final answers, `reply` should contain the user-facing text and `system_calls` should be empty or omitted.

## Parsing Fallbacks

`AgenticOutputParser.kt` accepts:

- strict JSON object
- fenced JSON block
- JSON object embedded in surrounding text
- single tool object, such as `{"type":"python_exec","code":"..."}`
- `system_calls` as either an array or a single object
- nested agent JSON inside the `reply` field
- string arrays for `memory_updates` and `skill_updates`
- loose `python_exec` extraction when malformed JSON still contains a recoverable `type` and `code`

The goal is to avoid showing raw dictionaries to users when a local model formats tool calls imperfectly.

## Tool Execution

`AgenticToolExecutor.kt` implements the current tool layer.

### get_current_time

Returns local datetime, weekday, timezone, and epoch milliseconds.

### web_search

Uses Bing HTML search:

```text
https://www.bing.com/search?q=...&form=QBRE&pq=...&qs=n&sp=-1&lq=0
```

The parser extracts title, URL, and snippet from `b_algo` blocks. Bing redirect URLs are decoded when possible.

### browser_url

Reads an HTTP/HTTPS page and extracts readable text from the HTML body. This is a lightweight browser-readable fallback, not the full GeckoView implementation used by ActMe.

### python_exec

Runs bounded Python 3.11 code through Chaquopy and the app's `agent_python.py` sandbox. The sandbox supports deterministic computation, JSON/text processing, reusable scripts, py_compile-style checks through `compile_script(name)`, Excel helpers through `read_excel(path)` and `write_excel(filename, sheets)`, and table/data packages including `numpy`, `pandas`, and `openpyxl`.

Generated file handoff is collected from:

- files created or modified inside the Python workspace
- declared `output_files`, `generated_files`, `expected_outputs`, or `files` in tool calls
- workspace-looking file references in the final answer

Only files resolved inside the agent workspace are returned as chat attachments.

### Memory and Skill

Agent JSON may include `memory_updates` and `skill_updates`. MnnLlmChat stores them in SQLite tables managed by `ChatDatabaseHelper` / `ChatDataManager`, injects them into future Agent prompts, and appends matching local skill hints when a user message contains a stored trigger keyword.

## Budgets and Stop Behavior

Agent mode uses conservative mobile budgets:

- max passes
- max total tool calls
- max browser calls
- max Python calls

Repeated search queries, repeated URLs, and repeated Python snippets are skipped. When the budget is exhausted, the model is asked to produce the best final answer from available observations and not request more tools.

Stop behavior:

- User stop sets `stopGenerating`.
- The loop checks stop state before model/tool continuation points.
- User stop returns a stopped result.
- Coroutine cancellation from lifecycle destruction is rethrown instead of being treated as a user stop.

## Implementation Points

Key files:

```text
app/src/main/java/com/alibaba/mnnllm/android/agent/AgenticPrompts.kt
app/src/main/java/com/alibaba/mnnllm/android/agent/AgenticProtocol.kt
app/src/main/java/com/alibaba/mnnllm/android/agent/AgenticOutputParser.kt
app/src/main/java/com/alibaba/mnnllm/android/agent/AgenticToolExecutor.kt
app/src/main/java/com/alibaba/mnnllm/android/agent/AgenticPythonEngine.kt
app/src/main/python/agent_python.py
app/src/main/java/com/alibaba/mnnllm/android/chat/ChatPresenter.kt
app/src/main/java/com/alibaba/mnnllm/android/chat/ChatActivity.kt
app/src/main/java/com/alibaba/mnnllm/android/chat/model/ChatDatabaseHelper.kt
app/src/main/java/com/alibaba/mnnllm/android/chat/model/ChatDataManager.kt
```

Important paths:

- `ChatActivity` chooses and restores the session mode.
- `ChatPresenter` applies the matching system prompt and chooses normal vs Agent execution.
- `ChatDatabaseHelper` and `ChatDataManager` persist `sessionMode`, Agent Memory, and Agent Skill.
- `AgenticToolExecutor` executes current tools.
- `AgenticPythonEngine` owns Chaquopy startup and bounded Python execution.

## Relationship to ActMe

ActMe is the upstream/reference project for the mobile agent design:

```text
https://github.com/huangzhengxiang/ActMe.git
```

ActMe contains the fuller implementation:

- multi-backend search
- GeckoView-rendered browsing
- Python sandbox and Excel processing
- ADB pairing/execution
- memory, schedule, and skill updates
- richer tool execution UI

MnnLlmChat currently ports the local-model agent loop, browser/time tools, Python execution, and lightweight Skill/Memory persistence. Future work can incrementally add the remaining ActMe capabilities while keeping normal chat stable.

## Known Limits

- Search and page reading are lightweight and may fail on CAPTCHA, login, heavy JavaScript, or anti-bot pages.
- Python execution is enabled through Chaquopy and the ActMe-derived `agent_python.py` sandbox.
- No ADB integration in this port.
- No cross-process task resume yet.
- Local models may still emit imperfect JSON; parser fallbacks reduce but do not eliminate this risk.
