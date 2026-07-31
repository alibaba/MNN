package com.alibaba.mnntalk

import android.app.Application
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.download.ModelDownloadManager

class MNNTalkApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        ApplicationProvider.set(this)
        ModelDownloadManager.getInstance(this).setProgressCallbackIntervalMs(500L)
    }
}
