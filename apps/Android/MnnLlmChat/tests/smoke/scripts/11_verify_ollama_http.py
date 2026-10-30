#!/usr/bin/env python3
"""Exercise the Ollama text API against a running MNN Chat service.

Set MNN_BASE_URL and MNN_API_KEY. Authentication must be enabled and a model
must already be loaded. The script sends real inference requests.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request


BASE_URL = os.environ.get("MNN_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.environ.get("MNN_API_KEY", "")


def request(path, payload=None, key=None, method=None):
    headers = {}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        return urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as error:
        return error


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def status(path, payload, key, expected, method=None):
    with request(path, payload, key, method) as response:
        check(response.status == expected, f"{path}: expected {expected}, got {response.status}")
        response.read()


def preflight(origin, allowed, path="/api/chat", method="POST"):
    req = urllib.request.Request(
        BASE_URL + path, method="OPTIONS",
        headers={"Origin": origin, "Access-Control-Request-Method": method,
                 "Access-Control-Request-Headers": "authorization,content-type"},
    )
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        returned = response.headers.get("Access-Control-Allow-Origin")
        if allowed:
            check(response.status == 200 and returned in (origin, "*"),
                  f"preflight {origin}: status={response.status}, allow-origin={returned}")
            headers = response.headers.get("Access-Control-Allow-Headers", "").lower()
            check("authorization" in headers and "content-type" in headers,
                  f"preflight {origin}: required headers missing")
            if method == "DELETE":
                methods = response.headers.get("Access-Control-Allow-Methods", "").upper()
                check("DELETE" in methods, f"preflight {origin}: DELETE not allowed")
        else:
            check(returned not in (origin, "*"), f"preflight unexpectedly allowed {origin}")
    print(f"PASS CORS preflight {method} {origin} allowed={allowed}")


def generation(path, body, model, chat, stream):
    start = time.monotonic()
    with request(path, body, API_KEY) as response:
        if response.status != 200:
            raise AssertionError(f"{path}: HTTP {response.status}: {response.read().decode(errors='replace')}")
        expected_type = "application/x-ndjson" if stream else "application/json"
        check(response.headers.get_content_type() == expected_type, f"{path}: unexpected Content-Type")
        if stream:
            records = []
            first_seconds = None
            for raw in response:
                if not raw.strip():
                    continue
                if first_seconds is None:
                    first_seconds = time.monotonic() - start
                records.append(json.loads(raw))
        else:
            records = [json.load(response)]
            first_seconds = time.monotonic() - start
    check(records, f"{path}: empty response")
    check(all("error" not in record for record in records), f"{path}: generation error: {records}")
    check(all(record.get("model") == model for record in records), f"{path}: wrong model")
    check(all(record.get("done") is False for record in records[:-1]), f"{path}: premature done")
    check(records[-1].get("done") is True, f"{path}: missing final done")
    if chat:
        check(all(record.get("message", {}).get("role") == "assistant" for record in records), f"{path}: bad role")
        text = "".join(record["message"]["content"] for record in records)
    else:
        text = "".join(record["response"] for record in records)
    check(text.strip(), f"{path}: no generated text")
    check(records[-1].get("eval_count", 0) > 0, f"{path}: no generated tokens reported")
    if stream:
        check(len(records) > 1, f"{path}: expected incremental records and final record")
    print(f"PASS {path} stream={stream} records={len(records)} "
          f"tokens={records[-1]['eval_count']} first={first_seconds:.2f}s total={time.monotonic() - start:.2f}s "
          f"text={text[:160]!r}")


def main():
    check(API_KEY, "Set MNN_API_KEY to the configured MNN Chat key")
    for env_name, allowed in [("MNN_ALLOWED_ORIGIN", True), ("MNN_DENIED_ORIGIN", False),
                              ("MNN_SCHEME_DENIED_ORIGIN", False)]:
        origin = os.environ.get(env_name)
        if origin:
            preflight(origin, allowed)
            if os.environ.get("MNN_TEST_QUEUE_CLEAR") == "1":
                preflight(origin, allowed, path="/v1/queue/clear", method="DELETE")
    for path, body in [("/api/tags", None), ("/api/chat", {}), ("/api/generate", {})]:
        status(path, body, None, 401)
        status(path, body, API_KEY + "-incorrect", 401)
    status("/v1/queue/status", None, None, 401)
    status("/v1/queue/status", None, API_KEY, 200)
    status("/v1/models", None, None, 401)
    status("/v1/models", None, API_KEY + "-incorrect", 401)
    status("/v1/models", None, API_KEY, 200)
    if os.environ.get("MNN_TEST_QUEUE_CLEAR") == "1":
        status("/v1/queue/clear", None, None, 401, method="DELETE")
        status("/v1/queue/clear", None, API_KEY + "-incorrect", 401, method="DELETE")
        status("/v1/queue/clear", None, API_KEY, 200, method="DELETE")
    with request("/api/tags", key=API_KEY) as response:
        check(response.status == 200, f"/api/tags: HTTP {response.status}")
        models = json.load(response)["models"]
    check(len(models) == 1, "Expected exactly one loaded model from /api/tags")
    model = models[0]["name"]
    check(models[0]["model"] == model, "Model name and ID differ")
    print(f"PASS authentication and model discovery: {model}")
    status("/api/chat", {"model": model, "messages": [], "options": {}}, API_KEY, 400)
    status("/api/chat", {"model": model, "messages": [{"role": "user", "content": "hello"}],
                         "tools": [{"type": "function"}]}, API_KEY, 400)
    status("/api/generate", {"model": model, "prompt": "hello", "stream": "false"}, API_KEY, 400)
    status("/api/generate", {"model": model + "-missing", "prompt": "hello", "stream": False}, API_KEY, 404)
    print("PASS invalid request and missing model handling")
    chat_body = {"model": model, "messages": [{"role": "user", "content": "Say hello briefly. /no_think"}],
                 "tools": [], "options": {}}
    generate_body = {"model": model, "prompt": "Say hello briefly. /no_think"}
    generation("/api/chat", {**chat_body, "stream": False}, model, True, False)
    generation("/api/chat", chat_body, model, True, True)
    generation("/api/generate", {**generate_body, "stream": False}, model, False, False)
    generation("/api/generate", generate_body, model, False, True)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
