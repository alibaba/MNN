# Mobile Download Feature Testing Guide

This document outlines how to verify the model download functionality and status updates using the Mobile MCP tools.

## Prerequisites
*   Ensure the Android emulator or device is connected.
*   Ensure the `MnnLlmChat` app is installed and running.

## Testing Steps

### 1. Setup & Launch
First, list available devices and ensure your target device is ready.
```bash
mobile_list_available_devices
```

Launch the application:
```bash
mobile_launch_app(device="YOUR_DEVICE_ID", packageName="com.alibaba.mnnllm.android")
```

### 2. Locate a Model
Navigate to the Model Market. You can use filters to find specific models quickly (e.g., MobileLLM).

**Action: Open Filter Menu**
Find the coordinates for "全部筛选" (Filter All) and click it.
```bash
mobile_list_elements_on_screen(device="...")
mobile_click_on_screen_at_coordinates(device="...", x=..., y=...)
```

**Action: Apply Vendor Filter**
1.  List elements to find the "MobileLLM" (or other vendor) radio button.
2.  Click the vendor option.
3.  Click the "确认" (Confirm) button at the bottom.

### 3. Start Download
Find a model that is not yet downloaded (button says "下载" / "Download").

**Action: Click Download**
```bash
mobile_click_on_screen_at_coordinates(device="...", x=BUTTON_X, y=BUTTON_Y)
```

### 4. Monitor Progress
You can monitor the download progress via logs or by repeatedly checking the screen status.

**Log Monitoring:**
Check for download progress events:
```bash
adb -s YOUR_DEVICE_ID logcat -d | grep "onDownloadProgress"
```

**Screen Monitoring:**
List elements periodically to see the button text change (e.g., "Preparing..." -> "1.5%" -> "50.0%").

### 5. Verify Completion Status
Once the download finishes, verify the final UI state.

**Expected Result:**
*   **Button Text**: Should change to "对话" (Chat).
*   **Status Text**: Should show the file size (e.g., "626.72 MB").
*   **Negative Check**: The status text should **NOT** contain "(有更新可用)" or "(Update Available)".

**Action: Verify Elements**
```bash
mobile_list_elements_on_screen(device="...")
```
*Check the `text` field of the corresponding model's status TextView.*

## Troubleshooting
If the status incorrectly shows "Update Available":
1.  Check `adb logcat` for `onDownloadHasUpdate` events.
2.  Verify if the `hasUpdate` flag was set during the `onRepoInfo` callback in `ModelDownloadManager`.
3.  Ensure `ModelDownloadManager.onDownloadFileFinished` is being called and properly clearing the flag.
