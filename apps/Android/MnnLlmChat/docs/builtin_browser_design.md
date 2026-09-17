# Built-in Browser Design

This document describes the browser capability to port into `MnnLlmChat`.

Core technical source: `https://github.com/huangzhengxiang/ActMe.git`

Status: the current MnnLlmChat Agent mode includes a lightweight Bing/HTTP implementation. The full GeckoView-rendered browser flow remains an ActMe reference capability for future porting.

## Purpose

Search snippets are often unstable or incomplete. The agent should be able to open a concrete URL, wait for the page to load, read the visible content, and feed that observation back to the model.

## Capabilities

The browser layer should support:

- `web_search(query)`: returns result titles, snippets, and URLs.
- `browser_url(url, goal)`: opens or reads a specific page.
- page title, final URL, load status, and readable text extraction.
- visible step records in chat: started, loaded, failed, skipped.

## Recommended Runtime

For China-accessible web behavior, a real browser-backed path is more reliable than plain HTTP scraping for dynamic pages and anti-bot behavior. The preferred long-term options are:

- GeckoView-backed hidden or debug-visible browser.
- Android WebView fallback for simple pages.
- OkHttp fallback only for static text pages.

The tool should expose text observations, not screenshots only. If the browser can render a page but text extraction fails, the observation should say so explicitly.

## Agent Contract

`browser_url` input:

```json
{
  "type": "browser_url",
  "url": "https://example.com/page",
  "goal": "verify the current price table"
}
```

Observation:

```json
{
  "type": "browser_url",
  "status": "ok",
  "title": "Page title",
  "final_url": "https://example.com/page",
  "text": "readable page text",
  "error": null
}
```

## UI Requirements

Each browser action should be visible:

- why it was opened
- URL or host
- loading/failure status
- extracted title and short preview
- optional button to open the full browser view

The debug browser can be hidden by default, but there should be a developer-visible route for diagnosing search and page extraction problems.
