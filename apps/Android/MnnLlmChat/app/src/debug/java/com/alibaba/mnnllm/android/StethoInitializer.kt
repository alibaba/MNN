package com.alibaba.mnnllm.android

import android.content.Context
import com.facebook.stetho.Stetho
import com.alibaba.mnnllm.android.debug.DownloadDumperPlugin
import com.alibaba.mnnllm.android.debug.ModelListDumperPlugin
import com.alibaba.mnnllm.android.debug.LoggerDumperPlugin
import com.alibaba.mnnllm.android.debug.MarketDumperPlugin
import com.alibaba.mnnllm.android.debug.BenchmarkDumperPlugin
import com.alibaba.mnnllm.android.debug.SanaDumperPlugin

object StethoInitializer {
    fun initialize(context: Context) {
        val initializer = Stetho.newInitializerBuilder(context)
            .enableDumpapp {
                Stetho.DefaultDumperPluginsBuilder(context)
                    .provide(ModelListDumperPlugin())
                    .provide(LoggerDumperPlugin())
                    .provide(MarketDumperPlugin())
                    .provide(DownloadDumperPlugin())
                    .provide(SanaDumperPlugin())
                    .provide(BenchmarkDumperPlugin())
                    .finish()
            }
            .enableWebKitInspector(Stetho.defaultInspectorModulesProvider(context))
            .build()
        Stetho.initialize(initializer)
    }
}
