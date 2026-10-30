# Access MnnLlmChat API From PC

This document explains how to call the phone-side `MnnLlmChat` API from a computer.

## Important Rule

If you test from a computer **without** `adb forward`, do **not** use `127.0.0.1` or `localhost`.
Use the phone's **LAN IP** (for example `192.168.1.23`).

## Mode 1: LAN Direct (No `adb forward`)

Use this mode when your computer and phone are on the same LAN.

```bash
python3 test_api.py --host <PHONE_LAN_IP> --port 8080 --test all
```

Example:

```bash
python3 test_api.py --host 192.168.1.23 --port 8080 --test all
```

## Mode 2: ADB Forward

If you want to use `127.0.0.1`, you must enable `adb forward` first.

```bash
adb forward tcp:8080 tcp:8080
python3 test_api.py --adb-forward --host 127.0.0.1 --port 8080 --test all
```

## Claude Client Example (Anthropic-Compatible Route)

`MnnLlmChat` exposes an Anthropic-compatible endpoint at `/v1/messages`.

If your Claude client supports custom Anthropic base URL / API key settings:

```bash
export ANTHROPIC_BASE_URL=http://<PHONE_LAN_IP>:8080
export ANTHROPIC_API_KEY=<MNNLLMCHAT_API_KEY>
```

When not using `adb forward`, replace `<PHONE_LAN_IP>` with the real LAN IP and do not use `127.0.0.1`.

## Ollama-Compatible Text API

Start the API service with a model loaded. Set the client base URL to
`http://<PHONE_LAN_IP>:8080` (or `http://127.0.0.1:8080` on the same phone),
and use the API key shown in MNN Chat's API settings. This is an MNN Chat key,
not an Ollama cloud key. When authentication is enabled, the Ollama endpoints,
`/v1/models`, and the queue status and clear endpoints require it as a Bearer
token:

```bash
export MNN_BASE_URL=http://<PHONE_LAN_IP>:8080
export MNN_API_KEY='<MNN_CHAT_API_KEY>'
curl "$MNN_BASE_URL/api/tags" -H "Authorization: Bearer $MNN_API_KEY"
curl "$MNN_BASE_URL/api/chat" -H "Authorization: Bearer $MNN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"<CURRENT_MODEL_ID>","messages":[{"role":"user","content":"Hello"}],"stream":false}'
curl -N "$MNN_BASE_URL/api/generate" -H "Authorization: Bearer $MNN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"<CURRENT_MODEL_ID>","prompt":"Hello"}'
```

`/api/tags` lists only the active model's name and ID, without Ollama model
metadata; use its exact ID in requests. Chat
accepts text messages with `user`, `assistant`, and an optional initial `system`
role. Generate accepts `prompt` and optional `system`. Streaming is on by
default and returns newline-delimited JSON; `stream:false` returns one JSON
object. Images, tools, structured output, sampling options, model management,
and other Ollama fields and endpoints are not supported. Empty tools and options
are accepted; unsupported non-empty fields return HTTP 400. The response omits
`done_reason` because the API does not currently expose a reliable stop reason.
Browser clients also need CORS enabled in API settings.
When restricting origins, include the scheme and port (for example,
`http://127.0.0.1:23653`); `http` and `https` are treated as different origins.
For LAN clients, change the default loopback-only binding to a reachable
interface in API settings and use the phone's LAN IP. The built-in server uses
HTTP; an HTTPS origin in the CORS allowlist does not enable TLS. Authentication
failures use the existing Bearer challenge and may have an empty response body.
