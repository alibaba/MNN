# Smoke Workflow

## Preconditions

- Android device connected and authorized via adb
- Built AAB exists:
  - `apps/Android/MnnLlmChat/release_outputs/googleplay/app-googleplay-release.aab`
- `bundletool-all` jar exists (default `/tmp/bundletool-all-1.17.1.jar`)

## Workflow

1. Select target device (`adb devices -l`).
2. Build `.apks` from AAB with bundletool (`--mode=universal`).
3. Optionally uninstall previous package to avoid signature mismatch.
4. Install APK set with bundletool.
5. Verify package version with `adb shell dumpsys package`.
6. Launch app with `adb shell monkey`.
7. Capture current window/activity and UI dump.
8. Save artifacts and summary.

## Pass Criteria

- Install completes without error.
- Package exists on device.
- Version fields are readable.
- Launch command succeeds.
- Window/activity dump and screenshot artifacts are generated.

## Artifacts

- `artifacts/smoke_summary.txt`
- `artifacts/window_dump.txt`
- `artifacts/ui_dump.xml`
- `artifacts/main_screenshot.png`
