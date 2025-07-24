package com.alibaba.mnnllm.android.utils

import android.app.Activity
import android.app.Application
import android.os.Bundle
import java.lang.ref.WeakReference

/**
 * 全局工具类，用于跟踪当前活跃的 Activity
 */
object CurrentActivityTracker : Application.ActivityLifecycleCallbacks {
    private var currentActivityRef: WeakReference<Activity>? = null
    
    val currentActivity: Activity?
        get() = currentActivityRef?.get()
    
    fun initialize(application: Application) {
        application.registerActivityLifecycleCallbacks(this)
    }
    
    override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) {}
    
    override fun onActivityStarted(activity: Activity) {}
    
    override fun onActivityResumed(activity: Activity) {
        currentActivityRef = WeakReference(activity)
    }
    
    override fun onActivityPaused(activity: Activity) {}
    
    override fun onActivityStopped(activity: Activity) {}
    
    override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) {}
    
    override fun onActivityDestroyed(activity: Activity) {
        // 清理已销毁的Activity引用
        if (currentActivityRef?.get() == activity) {
            currentActivityRef = null
        }
    }
} 