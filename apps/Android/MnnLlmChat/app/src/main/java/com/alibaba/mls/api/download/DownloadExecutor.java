// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download;

import androidx.annotation.NonNull;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class DownloadExecutor {

    private static volatile DownloadExecutor instance = null;

    private ThreadPoolExecutor downloadExecutor;

    private DownloadExecutor() {
    }

    public static DownloadExecutor instance() {
        if (instance == null) {
            synchronized (DownloadExecutor.class) {
                if (instance == null) {
                    instance = new DownloadExecutor();
                }
            }
        }
        return instance;
    }

    public static ExecutorService provide() {
        return instance().getDownloadExecutor();
    }

    public static ExecutorService getExecutor() {
        return instance().getDownloadExecutor();
    }


    public ThreadPoolExecutor getDownloadExecutor() {
        if (downloadExecutor == null || downloadExecutor.isShutdown() || downloadExecutor.isTerminated()) {
            synchronized (this) {
                if (downloadExecutor == null || downloadExecutor.isShutdown() || downloadExecutor.isTerminated()) {
                    downloadExecutor = getThreadPoolExecutor();
                }
            }
        }
        return downloadExecutor;
    }

    @NonNull
    private static ThreadPoolExecutor getThreadPoolExecutor() {
        ThreadFactory namedThreadFactory = new ThreadFactory() {
            private final AtomicInteger count = new AtomicInteger(1);
            @Override
            public Thread newThread(Runnable r) {
                return new Thread(r, "AutoShutdownExecutor-Thread-" + count.getAndIncrement());
            }
        };

        ThreadPoolExecutor executor = new ThreadPoolExecutor(
                10, // Core pool size
                20, // Maximum pool size
                3, // Keep-alive time
                TimeUnit.SECONDS,
                new LinkedBlockingQueue<Runnable>(),
                namedThreadFactory
        );
        executor.allowCoreThreadTimeOut(true);
        return executor;
    }
}
