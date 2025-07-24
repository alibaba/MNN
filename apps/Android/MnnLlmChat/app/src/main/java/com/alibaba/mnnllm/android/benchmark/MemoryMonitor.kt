package com.alibaba.mnnllm.android.benchmark

import android.os.Debug
import android.util.Log
import java.util.Timer
import java.util.TimerTask
import java.util.concurrent.ConcurrentLinkedQueue

object MemoryMonitor {

    private var maxPssKb: Long = 0
    private val memoryHistory = ConcurrentLinkedQueue<Long>()
    private var timer: Timer? = null

    fun start(intervalSeconds: Long = 5) {
        stop() // Stop any existing timer
        timer = Timer()
        timer?.schedule(object : TimerTask() {
            override fun run() {
                val memoryInfo = Debug.MemoryInfo()
                Debug.getMemoryInfo(memoryInfo)
                
                // getTotalPss() 返回的是 KB 单位的值
                val currentPssKb = memoryInfo.totalPss.toLong()

                if (currentPssKb > maxPssKb) {
                    maxPssKb = currentPssKb
                }
                memoryHistory.add(currentPssKb)

                 Log.d("MemoryMonitor", "Current PSS: ${currentPssKb}KB, Max PSS: ${maxPssKb}KB")
            }
        }, 0, intervalSeconds * 1000)
    }

    fun stop() {
        timer?.cancel()
        timer = null
    }

    fun getMaxMemoryPssKb(): Long {
        return maxPssKb
    }

    fun getMemoryHistory(): List<Long> {
        return memoryHistory.toList()
    }

    fun reset() {
        stop()
        maxPssKb = 0
        memoryHistory.clear()
    }
}