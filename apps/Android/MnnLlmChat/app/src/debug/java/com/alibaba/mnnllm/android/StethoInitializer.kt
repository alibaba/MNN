package com.alibaba.mnnllm.android

import android.content.Context
import com.facebook.stetho.Stetho
import com.alibaba.mnnllm.android.debug.DownloadDumperPlugin
import com.alibaba.mnnllm.android.debug.ModelListDumperPlugin
import com.alibaba.mnnllm.android.debug.LoggerDumperPlugin

object StethoInitializer {
    fun initialize(context: Context) {
        val initializer = Stetho.newInitializerBuilder(context)
            .enableDumpapp {
                Stetho.DefaultDumperPluginsBuilder(context)
                    .provide(ModelListDumperPlugin())
                    .provide(LoggerDumperPlugin())
                    .provide(DownloadDumperPlugin())
                    .finish()
            }
            .enableWebKitInspector(Stetho.defaultInspectorModulesProvider(context))
            .build()
        Stetho.initialize(initializer)
    }
}
