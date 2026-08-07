---
name: ios-install
description: Build and install the MNN Chat iOS app (apps/iOS/MNNLLMChat) on a real iPhone/iPad with a bundled local model. Use when asked to 安装/装 MNN Chat app 到 iPad/iPhone, deploy or reinstall the iOS chat app, bundle a new MNN model into the iOS app, or rebuild MNN.framework for the iOS app.
---

# iOS Install (MNN Chat)

## Overview

End-to-end pipeline: compile `MNN.framework` from an MNN source tree -> bundle a model into `LocalModel/` -> build `MNNLLMiOS` with personal-team signing -> install and launch on a connected device.

## Workflow

Run the bundled script (it prints progress and handles all known failure modes):

```bash
bash skills/ios-install/scripts/install_ios.sh --model <model-dir> [--device <udid>] [--src <mnn-source>]
```

Defaults: `--src ~/work/AliNNPrivate`, device auto-detected via `xcrun xctrace list devices`. Confirm the target device with the user if more than one is online. Use `--skip-framework` or keep an existing `apps/iOS/MNNLLMChat/MNN.framework` to reuse a previously built framework (the script skips the build when the framework already exists; delete it to force rebuild).

## Key facts (do not rediscover)

- **Environment pollution**: this machine's shell exports `SDKROOT`/`CPATH` pointing at the MacOSX SDK. Any iOS compile (cmake/make/xcodebuild) MUST run under `env -i PATH=/usr/bin:/bin ...` or it fails with `<cstddef> tried including <stddef.h>...` or `_c_standard_library_obsolete` modulemap errors. The script already does this.
- **Backend selection**: the app reads `backend_type` from the model's own `config.json` first (ModelConfigManager falls back to cpu only if absent). To use Metal, ensure the model config has `"backend_type": "metal"`; no app code change needed.
- **Signing**: personal team `3TLG5LZ643`, bundle id `com.jingbang.mnnchat.dev`, entitlements stripped of Extended Virtual Addressing / increased memory limit (personal teams cannot use them). Files: `assets/personal.entitlements`, applied via xcodebuild command-line overrides (which beat pbxproj values; an `-xcconfig` file does NOT).
- **Model bundling**: copy the model folder into `MNNLLMiOS/LocalModel/`; the app auto-discovers any subfolder containing `config.json` and shows it under Local Models with the folder name. Remove old/unused model folders there before building — each is ~1.2G and bloats the app.
- **Free-team app limit**: personal teams allow at most 3 apps per device. Install failure `ApplicationVerificationFailed` / `MIFreeProfileValidatedAppTracker` means the limit is hit — list installed apps, ask the user which to uninstall.
- **First-launch trust**: after a fresh team's first install, the user must trust the developer in Settings > General > VPN & Device Management before the app can launch.

## Verify

`xcrun devicectl device info apps --device <udid>` shows `MNN Chat` / the bundle id; the script launches the app at the end. If launch reports a signature/trust error, ask the user to trust the developer, then:

```bash
xcrun devicectl device process launch --device <udid> com.jingbang.mnnchat.dev
```
