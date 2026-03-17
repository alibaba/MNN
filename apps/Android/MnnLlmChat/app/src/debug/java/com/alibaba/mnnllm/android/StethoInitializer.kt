package com.alibaba.mnnllm.android

import android.content.Context
import com.facebook.stetho.Stetho
import com.alibaba.mnnllm.android.debug.ConfigDumperPlugin
import com.alibaba.mnnllm.android.debug.DownloadDumperPlugin
import com.alibaba.mnnllm.android.debug.HistoryDumperPlugin
import com.alibaba.mnnllm.android.debug.ModelListDumperPlugin
import com.alibaba.mnnllm.android.debug.LoggerDumperPlugin
import com.alibaba.mnnllm.android.debug.MarketDumperPlugin
import com.alibaba.mnnllm.android.debug.BenchmarkDumperPlugin
import com.alibaba.mnnllm.android.debug.DiffusionDumperPlugin
import com.alibaba.mnnllm.android.debug.SanaDumperPlugin
import com.alibaba.mnnllm.android.debug.StorageDumperPlugin
import com.alibaba.mnnllm.android.debug.VoiceDumperPlugin
import com.alibaba.mnnllm.api.openai.debug.LlmDumperPlugin
import com.alibaba.mnnllm.api.openai.debug.OpenApiDumperPlugin

object StethoInitializer {
    fun initialize(context: Context) {
        val initializer = Stetho.newInitializerBuilder(context)
            .enableDumpapp {
                Stetho.DefaultDumperPluginsBuilder(context)
                    .provide(ConfigDumperPlugin())
                    .provide(ModelListDumperPlugin())
                    .provide(LoggerDumperPlugin())
                    .provide(MarketDumperPlugin())
                    .provide(DownloadDumperPlugin())
                    .provide(SanaDumperPlugin())
                    .provide(DiffusionDumperPlugin())
                    .provide(BenchmarkDumperPlugin())
                    .provide(StorageDumperPlugin())
                    .provide(VoiceDumperPlugin())
                    .provide(LlmDumperPlugin())
                    .provide(OpenApiDumperPlugin())
                    .provide(HistoryDumperPlugin())
                    .finish()
            }
            .enableWebKitInspector(Stetho.defaultInspectorModulesProvider(context))
            .build()
        Stetho.initialize(initializer)
    }
}
