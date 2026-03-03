package com.alibaba.mnnllm.android.debug

import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.modelist.ModelItemWrapper
import com.alibaba.mnnllm.android.modelmarket.TagMapper
import com.alibaba.mnnllm.android.utils.FileUtils
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.PrintStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

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
            "list" -> doList(writer, args.drop(1))
            "refresh" -> doRefresh(writer)
            "tags" -> doTags(writer, args.drop(1))
            "find" -> doFind(writer, args.drop(1))
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

    /**
     * List all models matching UI display exactly.
     * Usage: dumpapp models list [--verbose|-v]
     */
    private fun doList(writer: PrintStream, args: List<String>) {
        val verbose = args.any { it == "--verbose" || it == "-v" }
        val state = ModelListManager.modelListState.value

        if (state !is ModelListManager.ModelListState.Success) {
            writer.println("ModelList not ready. State: $state")
            return
        }

        val models = state.models
        writer.println("=== ModelListFragment Display (${models.size} models) ===")
        writer.println()

        models.forEachIndexed { index, wrapper ->
            printModelItem(writer, index + 1, wrapper, verbose)
        }

        writer.println()
        writer.println("Total: ${models.size} models")
    }

    /**
     * Print a single model item in a format matching the UI display.
     */
    private fun printModelItem(writer: PrintStream, index: Int, wrapper: ModelItemWrapper, verbose: Boolean) {
        val modelItem = wrapper.modelItem
        val modelId = modelItem.modelId ?: "unknown"
        val modelName = modelItem.modelName ?: ""

        // Get display tags exactly as shown in UI (using TagMapper, limited to 3)
        val rawTags = modelItem.getTags()
        val displayTags = TagMapper.getDisplayTagList(rawTags).take(3)

        // Format size like UI does
        val sizeStr = if (wrapper.downloadSize > 0) {
            FileUtils.formatFileSize(wrapper.downloadSize)
        } else {
            ""
        }

        // Format last chat time like UI does
        val lastChatTimeStr = formatLastChatTime(wrapper.lastChatTime)

        // Pinned indicator
        val pinnedIndicator = if (wrapper.isPinned) "[PINNED] " else ""

        writer.println("[$index] $pinnedIndicator$modelName")
        writer.println("    ID: $modelId")
        writer.println("    Tags: ${displayTags.joinToString(", ").ifEmpty { "(none)" }}")

        if (sizeStr.isNotEmpty()) {
            writer.println("    Size: $sizeStr")
        }

        if (lastChatTimeStr.isNotEmpty()) {
            writer.println("    Last Chat: $lastChatTimeStr")
        }

        if (verbose) {
            writer.println("    Local: ${wrapper.isLocal}")
            writer.println("    Raw Tags: ${rawTags.joinToString(", ").ifEmpty { "(none)" }}")
            writer.println("    Local Path: ${modelItem.localPath ?: "(none)"}")

            val downloadTimeStr = if (wrapper.downloadTime > 0) {
                SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(wrapper.downloadTime))
            } else {
                "(none)"
            }
            writer.println("    Download Time: $downloadTimeStr")

            // Extra tags (not shown to users in UI but available locally)
            val extraTags = ModelListManager.getExtraTags(modelId)
            if (extraTags.isNotEmpty()) {
                writer.println("    Extra Tags: ${extraTags.joinToString(", ")}")
            }

            // Model capabilities
            val capabilities = mutableListOf<String>()
            if (ModelListManager.isThinkingModel(modelId)) capabilities.add("Thinking")
            if (ModelListManager.isVisualModel(modelId)) capabilities.add("Visual")
            if (ModelListManager.isAudioModel(modelId)) capabilities.add("Audio")
            if (ModelListManager.isVideoModel(modelId)) capabilities.add("Video")
            if (capabilities.isNotEmpty()) {
                writer.println("    Capabilities: ${capabilities.joinToString(", ")}")
            }
        }

        writer.println()
    }

    /**
     * Format last chat time exactly as UI does:
     * - Same day: show time (e.g., "8:30")
     * - Different day: show date (e.g., "Jun 20" or "6月20日")
     */
    private fun formatLastChatTime(lastChatTime: Long): String {
        if (lastChatTime <= 0) {
            return ""
        }

        val now = System.currentTimeMillis()
        val chatDate = Date(lastChatTime)
        val today = Date(now)

        val isSameDay = isSameDay(chatDate, today)

        return if (isSameDay) {
            SimpleDateFormat("H:mm", Locale.getDefault()).format(chatDate)
        } else {
            val locale = Locale.getDefault()
            val dateFormat = if (locale.language == "zh") {
                SimpleDateFormat("M月d日", locale)
            } else {
                SimpleDateFormat("MMM d", locale)
            }
            dateFormat.format(chatDate)
        }
    }

    private fun isSameDay(date1: Date, date2: Date): Boolean {
        val cal1 = java.util.Calendar.getInstance()
        val cal2 = java.util.Calendar.getInstance()
        cal1.time = date1
        cal2.time = date2
        return cal1.get(java.util.Calendar.YEAR) == cal2.get(java.util.Calendar.YEAR) &&
                cal1.get(java.util.Calendar.DAY_OF_YEAR) == cal2.get(java.util.Calendar.DAY_OF_YEAR)
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

    private fun doTags(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp models tags <modelId>")
            return
        }
        val modelId = args[0]
        val model = ModelListManager.getModelIdModelMap()[modelId]
        if (model == null) {
            writer.println("Model not found: $modelId")
            return
        }

        val tags = ModelListManager.getModelTags(modelId)
        val extraTags = ModelListManager.getExtraTags(modelId)
        writer.println("Model: $modelId")
        writer.println("  modelName: ${model.modelName ?: ""}")
        writer.println("  tags: ${if (tags.isEmpty()) "[]" else tags.joinToString(prefix = "[", postfix = "]")}")
        writer.println("  extraTags: ${if (extraTags.isEmpty()) "[]" else extraTags.joinToString(prefix = "[", postfix = "]")}")
        writer.println("  isThinkingModel: ${ModelListManager.isThinkingModel(modelId)}")
        writer.println("  isVisualModel: ${ModelListManager.isVisualModel(modelId)}")
        writer.println("  isAudioModel: ${ModelListManager.isAudioModel(modelId)}")
        writer.println("  isVideoModel: ${ModelListManager.isVideoModel(modelId)}")
    }

    private fun doFind(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp models find <keyword>")
            return
        }
        val keyword = args.joinToString(" ").trim()
        if (keyword.isEmpty()) {
            writer.println("Usage: dumpapp models find <keyword>")
            return
        }
        val normalizedKeyword = keyword.lowercase()
        val matches = ModelListManager.getModelIdModelMap()
            .entries
            .filter { (modelId, model) ->
                modelId.lowercase().contains(normalizedKeyword) ||
                    (model.modelName?.lowercase()?.contains(normalizedKeyword) == true)
            }
            .sortedBy { it.key }

        if (matches.isEmpty()) {
            writer.println("No models matched keyword: $keyword")
            return
        }

        writer.println("Matched models (${matches.size}) for \"$keyword\":")
        matches.forEach { (modelId, model) ->
            val tags = ModelListManager.getModelTags(modelId)
            writer.println("  - $modelId")
            writer.println("    modelName: ${model.modelName ?: ""}")
            writer.println("    tags: ${if (tags.isEmpty()) "[]" else tags.joinToString(prefix = "[", postfix = "]")}")
        }
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp models <command>")
        writer.println("Commands:")
        writer.println("  list [-v|--verbose]   - List all models matching UI display")
        writer.println("  dump                  - Dump current model list state (raw)")
        writer.println("  refresh               - Trigger manual refresh")
        writer.println("  tags <modelId>        - Dump tags/capabilities for one model")
        writer.println("  find <keyword>        - Find models by id/name and print tags")
        writer.println()
        writer.println("Examples:")
        writer.println("  dumpapp models list           - List all models with display tags")
        writer.println("  dumpapp models list -v        - Verbose output with all details")
        writer.println("  dumpapp models tags ModelScope/MNN/Qwen3-0.6B-MNN")
    }
}
