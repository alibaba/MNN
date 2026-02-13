package com.alibaba.mnnllm.android.debug

import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.PrintStream

class ModelListDumperPlugin : DumperPlugin {

    override fun getName(): String {
        return "models"
    }

    override fun dump(dumpContext: DumperContext) {
        val writer = dumpContext.stdout
        val args = dumpContext.argsAsList

        if (args.isEmpty()) {
            doUsage(writer)
            return
        }

        val command = args[0]

        when (command) {
            "dump" -> doDump(writer)
            "refresh" -> doRefresh(writer)
            else -> doUsage(writer)
        }
    }

    private fun doDump(writer: PrintStream) {
        val state = ModelListManager.modelListState.value
        writer.println("Current ModelListState: $state")
        if (state is ModelListManager.ModelListState.Success) {
            writer.println("Source: ${state.source}")
            writer.println("Models Count: ${state.models.size}")
            state.models.forEach {
                writer.println("  - ${it.modelItem.modelId} (Pinned: ${it.isPinned}, Local: ${it.isLocal})")
            }
        }
    }

    private fun doRefresh(writer: PrintStream) {
        writer.println("Triggering ModelList refresh...")
        // Launch in IO scope since notifyModelListMayChange is suspend
        CoroutineScope(Dispatchers.IO).launch {
            try {
                ModelListManager.notifyModelListMayChange(ModelListManager.ChangeReason.MANUAL_REFRESH)
                // Note: This output might not be visible immediately in the dump output stream if the process exits
                // But for a dumper plugin, usually the command finishes quickly. 
                // We depend on logs or subsequent 'dump' to verify.
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        writer.println("Refresh triggered.")
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp models <command>")
        writer.println("Commands:")
        writer.println("  dump     - Dump current model list state")
        writer.println("  refresh  - Trigger manual refresh")
    }
}
