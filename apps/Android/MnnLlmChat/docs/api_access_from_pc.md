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
