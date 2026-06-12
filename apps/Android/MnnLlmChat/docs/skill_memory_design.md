# Skill and Memory Design

This document adapts ActMe's skill and memory concepts to `MnnLlmChat`.

Core technical source: `https://github.com/huangzhengxiang/ActMe.git`

Status: implemented as a lightweight MnnLlmChat SQLite port. Agent mode persists `memory_updates` and `skill_updates`, injects them into future Agent system prompts, and applies local skill hints when trigger keywords match.

## Memory

Memory records durable facts about the user or project:

- goals
- preferences
- recurring context
- recent active projects
- constraints the assistant should remember

Memory should be injected into the system prompt selectively. It should not blindly store every chat turn.

Minimal schema:

```json
{
  "category": "goal",
  "content": "User is preparing for an exam.",
  "source": "chat",
  "updated_at": 1780502400000
}
```

## Skill

Skill records reusable ways of doing work:

- study planning workflow
- web research workflow
- spreadsheet analysis workflow
- PDF/report generation workflow

Minimal schema:

```json
{
  "name": "spreadsheet_analysis",
  "description": "Inspect workbook sheets, summarize columns, run Python analysis, and return a result file.",
  "trigger_keywords": ["excel", "xlsx", "spreadsheet"],
  "action_template": "Inspect sheets first, then run Python, then explain and attach outputs.",
  "enabled": true
}
```

## Relation to Agentic Loop

Memory and skill are prompt context. The loop decides what to do next:

- memory tells the agent who the user is and what matters.
- skill tells the agent which workflow to prefer.
- tools execute the concrete browser or Python step.

## Safety

The user should be able to inspect, disable, or delete memory and skills. Sensitive or uncertain information should not be saved automatically.

In this MnnLlmChat port, Memory and Skill are persisted in the existing SQLite chat database. A future UI can expose inspection, disable, and deletion controls.
