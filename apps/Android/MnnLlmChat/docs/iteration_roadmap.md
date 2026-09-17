# Iteration Roadmap

This document describes how to continue evolving MnnLlmChat's Agent capabilities without destabilizing normal local chat.

Core technical source for the current Agent direction: `https://github.com/huangzhengxiang/ActMe.git`

## Product Direction

MnnLlmChat should remain a local on-device model application first.

Agent mode should add host-executed capabilities around the model:

- web search and page reading
- local Python computation and file generation
- lightweight memory and reusable skills
- visible multi-step execution
- safe file handoff back to the user

Normal chat should remain simple, fast, and compatible with older conversations.

## Iteration Principles

- Keep the normal path stable. Every Agent feature must be optional at the session level.
- Prefer host-side deterministic tools over asking the model to simulate exact work.
- Make every tool action visible to the user.
- Keep parser compatibility broad because local models do not reliably follow a single JSON shape.
- Treat prompt text, protocol classes, parser fallback, executor behavior, and docs as one contract.
- Add capability in vertical slices: prompt, protocol, parser, executor, UI, persistence, docs.

## Suggested Milestones

### 1. Stabilize Current Agent Loop

Goals:

- final answers should not show raw JSON
- generated files should always appear as attachments when present
- stop/cancel should not corrupt native history
- reopened Agent sessions should restore visible history and useful context

Key work:

- keep expanding parser fallbacks from real logcat failures
- add focused unit-style parser tests if the project test setup becomes practical
- keep tool budgets conservative on mobile

### 2. Improve Python and File Workflows

Goals:

- make Excel/CSV/JSON workflows reliable
- make generated files easy to inspect and open
- reduce cold-start friction for common data tasks

Key work:

- keep `ALLOWED_IMPORT_ROOTS` aligned with Chaquopy packages
- add examples that prefer `write_excel(...)` and `read_excel(...)`
- improve workspace file browser UX
- consider per-session workspace directories if global workspace clutter becomes a problem
- add explicit cleanup/export controls for workspace files

### 3. Upgrade Browser Capability

Goals:

- handle dynamic pages and pages that plain HTTP extraction cannot read
- make search/browse behavior easier to debug

Possible paths:

- port a GeckoView-backed browser layer from ActMe
- keep a debug-visible browser activity for inspecting actual rendered pages
- expose page title, final URL, selected text, and readable body text to the agent
- keep Bing as one search source, but design executor interfaces so other sources can be added later

### 4. Mature Skill and Memory

Goals:

- keep useful user preferences and workflows
- avoid saving noisy or hallucinated memory
- let users inspect and control stored information

Key work:

- add Memory/Skill management UI
- allow disabling or deleting skills
- require better structure for executable skills before automatic execution
- store source/session metadata for memory updates
- add confidence or review state for uncertain memory

### 5. Add Durable Task State

Goals:

- make long multi-step tasks resumable after app interruption
- make tool observations inspectable after completion

Key work:

- persist agent step logs
- persist tool observations
- persist generated file metadata
- design resume semantics separately from native KV cache
- avoid assuming a released native session can continue with the same hidden context

### 6. Consider Advanced Device Control

ADB/device control is intentionally not part of the current MnnLlmChat port.

If ported later, it should be treated as a high-risk capability:

- explicit user opt-in
- clear pairing UI
- visible command log
- stop button
- strict command allowlist or policy layer
- no hidden background control without user awareness

## Technical Debt to Watch

- Parser fallbacks can become hard to reason about. Keep examples in docs and add tests when possible.
- Prompt contract drift is easy. Update prompt, protocol, parser, executor, and docs together.
- Python dependencies increase APK size and build time.
- Native session reuse and history restore are subtle. Do not mix API stateless history submission with ChatActivity live-session semantics.
- Workspace files can accumulate. Add cleanup/export flows before the workspace becomes user-visible clutter.
- Browser scraping is brittle. Treat search/page failures as expected states, not exceptional app failures.

## Release Checklist for Agent Changes

- Normal chat still works without Agent mode.
- New conversations can choose normal or Agent mode.
- Reopened history restores the correct mode.
- Agent status steps are visible.
- Stop button interrupts the loop.
- Tool JSON is not shown as final answer.
- Python package allowlist matches installed packages.
- Generated files are attached and openable.
- Workspace file browser shows folders and files.
- Prompt cache behavior is documented if changed.
- README/docs are updated for any new tool, field, or user-visible workflow.

## Documentation Map

- `agentic_design.md`: current Agent architecture and loop.
- `builtin_browser_design.md`: web search and browser capability.
- `builtin_python_design.md`: Python runtime, sandbox, Excel/file handling.
- `skill_memory_design.md`: Memory and Skill persistence.
- `development_notes.md`: development rules and pitfalls.
- `iteration_roadmap.md`: future direction and staged work.
