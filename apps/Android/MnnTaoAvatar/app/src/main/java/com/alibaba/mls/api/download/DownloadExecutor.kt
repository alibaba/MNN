// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import java.util.concurrent.ExecutorService
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadFactory
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.Volatile

class DownloadExecutor private constructor() {
    private var downloadExecutor: ThreadPoolExecutor? = null

    fun getDownloadExecutor(): ThreadPoolExecutor? {
        if (downloadExecutor == null || downloadExecutor!!.isShutdown || downloadExecutor!!.isTerminated) {
            synchronized(this) {
                if (downloadExecutor == null || downloadExecutor!!.isShutdown || downloadExecutor!!.isTerminated) {
                    downloadExecutor = threadPoolExecutor
                }
            }
        }
        return downloadExecutor
    }

    companion object {
        @Volatile
        private var instance: DownloadExecutor? = null

        fun instance(): DownloadExecutor? {
            if (instance == null) {
                synchronized(DownloadExecutor::class.java) {
                    if (instance == null) {
                        instance = DownloadExecutor()
                    }
                }
            }
            return instance
        }

        fun provide(): ExecutorService? {
            return instance()!!.getDownloadExecutor()
        }

        @JvmStatic
        val executor: ExecutorService?
            get() = instance()!!.getDownloadExecutor()


        private val threadPoolExecutor: ThreadPoolExecutor
            get() {
                val namedThreadFactory: ThreadFactory =
                    object : ThreadFactory {
                        private val count =
                            AtomicInteger(1)

                        override fun newThread(r: Runnable): Thread {
                            return Thread(
                                r,
                                "AutoShutdownExecutor-Thread-" + count.getAndIncrement()
                            )
                        }
                    }

                val executor =
                    ThreadPoolExecutor(
                        10,  // Core pool size
                        20,  // Maximum pool size
                        3,  // Keep-alive time
                        TimeUnit.SECONDS,
                        LinkedBlockingQueue(),
                        namedThreadFactory
                    )
                executor.allowCoreThreadTimeOut(true)
                return executor
            }
    }
}
