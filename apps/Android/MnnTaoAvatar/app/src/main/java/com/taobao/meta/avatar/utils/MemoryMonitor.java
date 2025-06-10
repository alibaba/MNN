package com.taobao.meta.avatar.utils;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

public class MemoryMonitor {

    private static final String TAG = "MemoryMonitor";
    private static final long MEMORY_THRESHOLD = 4L * 1024 * 1024 * 1024 + 512 * 1024 * 1024; // 4.5GB in bytes
    private static final long MONITOR_INTERVAL = 500000; // 5 seconds
    private final ActivityManager mActivityManager;
    private final Handler mHandler;

    public MemoryMonitor(Context context) {
        mActivityManager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        HandlerThread handlerThread = new HandlerThread("MemoryMonitorThread");
        handlerThread.start();
        mHandler = new Handler(handlerThread.getLooper());
    }

    public void startMonitoring() {
        mHandler.post(memoryCheckRunnable);
    }

    public void stopMonitoring() {
        mHandler.removeCallbacks(memoryCheckRunnable);
    }

    private final Runnable memoryCheckRunnable = new Runnable() {
        @Override
        public void run() {
            long usedMemory = getUsedMemory();
            Log.d(TAG, "Current used memory: " + usedMemory + " bytes");

            if (usedMemory > MEMORY_THRESHOLD) {
                Log.d(TAG, "Memory threshold exceeded. Triggering GC.");
                System.gc();
            }

            // Schedule the next memory check
            mHandler.postDelayed(this, MONITOR_INTERVAL);
        }
    };

    private long getUsedMemory() {
        ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
        mActivityManager.getMemoryInfo(memoryInfo);
        return memoryInfo.totalMem - memoryInfo.availMem;
    }
}
