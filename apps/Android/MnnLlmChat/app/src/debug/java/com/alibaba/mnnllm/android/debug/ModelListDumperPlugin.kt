package com.alibaba.mnnllm.android.debug

import android.content.Context
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.modelist.ModelItemWrapper
import com.alibaba.mnnllm.android.modelmarket.TagMapper
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mls.api.ModelItem
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.PrintStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.modelmarket.Tag
import com.alibaba.mnnllm.android.utils.DeviceUtils
import java.io.File
import java.nio.file.Files

internal data class StorageEntry(
    val name: String,
    val absolutePath: String,
    val isSymlink: Boolean,
    val symlinkTarget: String?,
    val targetExists: Boolean,
    val isDirectory: Boolean,
    val modelId: String?,
    val container: String
)

internal data class TagLocaleInfo(
    val deviceLocale: String,
    val isChinese: Boolean,
    val sampleTags: List<TagDisplaySample>
)

internal data class TagDisplaySample(
    val rawTag: String,
    val chineseDisplay: String,
    val englishDisplay: String,
    val currentDisplay: String
)

internal interface ModelListDebugController {
    fun getModelListState(): ModelListManager.ModelListState
    suspend fun notifyModelListMayChange(reason: ModelListManager.ChangeReason)
    fun getModelIdModelMap(): Map<String, ModelItem>
    fun getModelTags(modelId: String): List<String>
    fun getExtraTags(modelId: String): List<String>
    fun isThinkingModel(modelId: String): Boolean
    fun isVisualModel(modelId: String): Boolean
    fun isAudioModel(modelId: String): Boolean
    fun isVideoModel(modelId: String): Boolean
    fun scanStorageEntries(): List<StorageEntry> = emptyList()
    fun deleteSymlink(absolutePath: String): Boolean = false
    fun getTagLocaleInfo(): TagLocaleInfo
}

internal object DefaultModelListDebugController : ModelListDebugController {
    override fun getModelListState(): ModelListManager.ModelListState = ModelListManager.modelListState.value

    override suspend fun notifyModelListMayChange(reason: ModelListManager.ChangeReason) {
        ModelListManager.notifyModelListMayChange(reason)
    }

    override fun getModelIdModelMap(): Map<String, ModelItem> = ModelListManager.getModelIdModelMap()

    override fun getModelTags(modelId: String): List<String> = ModelListManager.getModelTags(modelId)

    override fun getExtraTags(modelId: String): List<String> = ModelListManager.getExtraTags(modelId)

    override fun isThinkingModel(modelId: String): Boolean = ModelListManager.isThinkingModel(modelId)

    override fun isVisualModel(modelId: String): Boolean = ModelListManager.isVisualModel(modelId)

    override fun isAudioModel(modelId: String): Boolean = ModelListManager.isAudioModel(modelId)

    override fun isVideoModel(modelId: String): Boolean = ModelListManager.isVideoModel(modelId)

    override fun scanStorageEntries(): List<StorageEntry> {
        val result = mutableListOf<StorageEntry>()
        val context = ApplicationProvider.get()
        val mnnModelsDir = File(context.filesDir, ".mnnmodels")
        if (!mnnModelsDir.exists()) return result

        val containerNames = setOf("modelscope", "modelers", "builtin")

        for (containerName in containerNames) {
            val containerDir = File(mnnModelsDir, containerName)
            if (!containerDir.exists() || !containerDir.isDirectory) continue
            containerDir.listFiles()?.forEach { file ->
                if (file.isDirectory || Files.isSymbolicLink(file.toPath())) {
                    result.add(buildStorageEntry(file, containerName))
                }
            }
        }

        // Root-level entries (HuggingFace models)
        mnnModelsDir.listFiles()?.forEach { file ->
            if (file.name !in containerNames &&
                (file.isDirectory || Files.isSymbolicLink(file.toPath()))
            ) {
                result.add(buildStorageEntry(file, "(root)"))
            }
        }

        return result.sortedWith(compareBy({ it.container }, { it.name }))
    }

    override fun deleteSymlink(absolutePath: String): Boolean {
        val context = ApplicationProvider.get()
        val mnnModelsDir = File(context.filesDir, ".mnnmodels")
        if (!absolutePath.startsWith(mnnModelsDir.absolutePath)) return false

        val file = File(absolutePath)
        if (!Files.isSymbolicLink(file.toPath())) return false

        return file.delete()
    }

    private fun buildStorageEntry(file: File, container: String): StorageEntry {
        val isSymlink = Files.isSymbolicLink(file.toPath())
        val symlinkTarget = if (isSymlink) {
            try {
                Files.readSymbolicLink(file.toPath()).toString()
            } catch (_: Exception) {
                null
            }
        } else null

        val modelId = when (container) {
            "modelscope" -> "ModelScope/MNN/${file.name}"
            "modelers" -> "Modelers/MNN/${file.name}"
            "builtin" -> "Builtin/MNN/${file.name}"
            "(root)" -> "HuggingFace/taobao-mnn/${file.name}"
            else -> null
        }

        return StorageEntry(
            name = file.name,
            absolutePath = file.absolutePath,
            isSymlink = isSymlink,
            symlinkTarget = symlinkTarget,
            targetExists = file.exists(),
            isDirectory = file.isDirectory,
            modelId = modelId,
            container = container
        )
    }

    override fun getTagLocaleInfo(): TagLocaleInfo {
        val context = ApplicationProvider.get()
        val config = context.resources.configuration
        val locale = if (android.os.Build.VERSION.SDK_INT >= 24) {
            if (config.locales.isEmpty()) Locale.getDefault() else config.locales.get(0)
        } else {
            @Suppress("DEPRECATION")
            config.locale ?: Locale.getDefault()
        }
        
        val isChineseWithContext = DeviceUtils.isChinese(context)
        val isChineseStatic = DeviceUtils.isChinese // 无参数版本
        
        // Sample tags to show the display difference
        val sampleTagKeys = listOf("Chat", "Multimodal", "Think", "Vision", "local", "builtin")
        val sampleTags = sampleTagKeys.map { tagKey ->
            val tag = TagMapper.getTag(tagKey)
            TagDisplaySample(
                rawTag = tagKey,
                chineseDisplay = tag.getDisplayText(useChinese = true),
                englishDisplay = tag.getDisplayText(useChinese = false),
                // Use context version to match UI (ModelItemHolder uses context)
                currentDisplay = tag.getDisplayText(isChineseWithContext)
            )
        }
        
        return TagLocaleInfo(
            deviceLocale = "${locale.language}_${locale.country}",
            isChinese = isChineseWithContext,
            sampleTags = sampleTags
        )
    }
}

class ModelListDumperPlugin internal constructor(
    private val controller: ModelListDebugController = DefaultModelListDebugController
) : DumperPlugin {

    override fun getName(): String {
        return "models"
    }

    override fun dump(dumpContext: DumperContext) {
        execute(dumpContext.argsAsList, dumpContext.stdout)
    }

    internal fun execute(args: List<String>, writer: PrintStream) {
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
            "tagslocale" -> doTagsLocale(writer)
            "find" -> doFind(writer, args.drop(1))
            "files" -> doFiles(writer)
            "unlink" -> doUnlink(writer, args.drop(1))
            else -> doUsage(writer)
        }
    }

    private fun doDump(writer: PrintStream) {
        val state = controller.getModelListState()
        writer.println("Current ModelListState: $state")
        if (state is ModelListManager.ModelListState.Success) {
            writer.println("Source: ${state.source}")
            writer.println("Models Count: ${state.models.size}")
            state.models.forEach {
                writer.println("  - ${it.modelItem.modelId} (Pinned: ${it.isPinned}, Downloaded: ${it.isLocal})")
            }
        }
    }

    /**
     * List all models matching UI display exactly.
     * Usage: dumpapp models list [--verbose|-v]
     */
    private fun doList(writer: PrintStream, args: List<String>) {
        val verbose = args.any { it == "--verbose" || it == "-v" }
        val state = controller.getModelListState()

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
     * Source label for My Models tag (ModelScope, HuggingFace, Modelers, Builtin).
     * Mirrors ModelItemHolder.getModelSource() for dumpapp debugging.
     */
    private fun getSourceLabel(modelId: String?, isLocal: Boolean): String {
        if (isLocal || modelId == null || modelId == "unknown") return "(local)"
        val ctx = runCatching { ApplicationProvider.get() as? Context }.getOrNull()
        val huggingfaceLabel = ctx?.getString(R.string.huggingface) ?: "HuggingFace"
        val modelscopeLabel = ctx?.getString(R.string.modelscope) ?: "ModelScope"
        val modelersLabel = ctx?.getString(R.string.modelers) ?: "Modelers"
        val builtinLabel = ctx?.getString(R.string.builtin) ?: "Builtin"
        return when {
            modelId.startsWith("HuggingFace/") || modelId.contains("taobao-mnn") -> huggingfaceLabel
            modelId.startsWith("ModelScope/") -> modelscopeLabel
            modelId.startsWith("Modelers/") -> modelersLabel
            modelId.startsWith("Builtin/") -> builtinLabel
            else -> "(none)"
        }
    }

    /**
     * Print a single model item in a format matching the UI display.
     */
    private fun printModelItem(writer: PrintStream, index: Int, wrapper: ModelItemWrapper, verbose: Boolean) {
        val modelItem = wrapper.modelItem
        val modelId = modelItem.modelId ?: "unknown"
        val modelName = modelItem.modelName ?: ""

        // Get display tags exactly as shown in UI (using TagMapper with context, limited to 3)
        // Must use context version to match ModelItemHolder behavior
        val context = ApplicationProvider.get()
        // Use modelItem.tags (same as ModelItemHolder) instead of modelItem.getTags()
        val rawTags = modelItem.tags
        val displayTags = TagMapper.getDisplayTagList(rawTags, context).take(3)

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
        writer.println("    Source: ${wrapper.sourceTag ?: getSourceLabel(modelItem.modelId, wrapper.isLocal)}")
        writer.println("    Tags: ${displayTags.joinToString(", ").ifEmpty { "(none)" }}")

        if (sizeStr.isNotEmpty()) {
            writer.println("    Size: $sizeStr")
        }

        if (lastChatTimeStr.isNotEmpty()) {
            writer.println("    Last Chat: $lastChatTimeStr")
        }

        if (verbose) {
            writer.println("    Downloaded: ${wrapper.isLocal}")
            writer.println("    Raw Tags (modelItem.tags): ${rawTags.joinToString(", ").ifEmpty { "(none)" }}")
            writer.println("    Raw Tags (getTags()): ${modelItem.getTags().joinToString(", ").ifEmpty { "(none)" }}")
            writer.println("    Display Tags: ${displayTags.joinToString(", ").ifEmpty { "(none)" }}")
            writer.println("    Local Path: ${modelItem.localPath ?: "(none)"}")

            val downloadTimeStr = if (wrapper.downloadTime > 0) {
                SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(wrapper.downloadTime))
            } else {
                "(none)"
            }
            writer.println("    Download Time: $downloadTimeStr")

            // Extra tags (not shown to users in UI but available locally)
            val extraTags = controller.getExtraTags(modelId)
            if (extraTags.isNotEmpty()) {
                writer.println("    Extra Tags: ${extraTags.joinToString(", ")}")
            }

            // Model capabilities
            val capabilities = mutableListOf<String>()
            if (controller.isThinkingModel(modelId)) capabilities.add("Thinking")
            if (controller.isVisualModel(modelId)) capabilities.add("Visual")
            if (controller.isAudioModel(modelId)) capabilities.add("Audio")
            if (controller.isVideoModel(modelId)) capabilities.add("Video")
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
                controller.notifyModelListMayChange(ModelListManager.ChangeReason.MANUAL_REFRESH)
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
            doTagsAll(writer)
            return
        }
        if (args[0] == "all" || args[0] == "--all") {
            doTagsAll(writer)
            return
        }
        val modelId = args[0]
        val model = controller.getModelIdModelMap()[modelId]
        if (model == null) {
            writer.println("Model not found: $modelId")
            return
        }

        val tags = controller.getModelTags(modelId)
        val extraTags = controller.getExtraTags(modelId)
        val isLocal = modelId.startsWith("local/")
        writer.println("Model: $modelId")
        writer.println("  modelName: ${model.modelName ?: ""}")
        writer.println("  source: ${getSourceLabel(modelId, isLocal)}")
        writer.println("  tags: ${if (tags.isEmpty()) "[]" else tags.joinToString(prefix = "[", postfix = "]")}")
        writer.println("  extraTags: ${if (extraTags.isEmpty()) "[]" else extraTags.joinToString(prefix = "[", postfix = "]")}")
        writer.println("  isThinkingModel: ${controller.isThinkingModel(modelId)}")
        writer.println("  isVisualModel: ${controller.isVisualModel(modelId)}")
        writer.println("  isAudioModel: ${controller.isAudioModel(modelId)}")
        writer.println("  isVideoModel: ${controller.isVideoModel(modelId)}")
    }

    private fun doTagsAll(writer: PrintStream) {
        val modelEntries = controller.getModelIdModelMap()
            .entries
            .sortedBy { it.key }
        if (modelEntries.isEmpty()) {
            writer.println("No models found.")
            return
        }

        writer.println("All model tags (${modelEntries.size})")
        modelEntries.forEach { (modelId, model) ->
            val tags = controller.getModelTags(modelId)
            val extraTags = controller.getExtraTags(modelId)
            val isLocal = modelId.startsWith("local/")
            writer.println("  - $modelId")
            writer.println("    modelName: ${model.modelName ?: ""}")
            writer.println("    source: ${getSourceLabel(modelId, isLocal)}")
            writer.println("    tags: ${if (tags.isEmpty()) "[]" else tags.joinToString(prefix = "[", postfix = "]")}")
            writer.println("    extraTags: ${if (extraTags.isEmpty()) "[]" else extraTags.joinToString(prefix = "[", postfix = "]")}")
        }
    }

    /**
     * Dump tag locale information for debugging tag display issues.
     * Shows device locale, isChinese result, and sample tag display in both languages.
     */
    private fun doTagsLocale(writer: PrintStream) {
        val info = controller.getTagLocaleInfo()
        
        writer.println("=== Tag Locale Debug Info ===")
        writer.println()
        writer.println("Device Locale: ${info.deviceLocale}")
        writer.println("DeviceUtils.isChinese: ${info.isChinese}")
        writer.println()
        
        // Check if TagMapper is initialized
        val allTags = TagMapper.getAllTags()
        writer.println("TagMapper initialized: ${allTags.isNotEmpty()}")
        writer.println("TagMapper tag count: ${allTags.size}")
        if (allTags.isEmpty()) {
            writer.println()
            writer.println("BUG: TagMapper.tagMap is EMPTY!")
            writer.println("This means TagMapper.initializeFromConfig() was never called.")
            writer.println("Tags will show raw English keys instead of translations.")
            writer.println()
            writer.println("TagMapper is only initialized in ModelMarketViewModel.loadModels(),")
            writer.println("which is only called when user enters Model Market page.")
            return
        }
        writer.println()
        
        writer.println("Expected behavior:")
        writer.println("  - zh_CN: should show Chinese tags")
        writer.println("  - zh_TW/zh_HK/zh_SG/zh: should show Chinese tags (currently broken)")
        writer.println("  - en_US/other: should show English tags")
        writer.println()
        writer.println("Sample Tags Display:")
        writer.println("  %-12s | %-10s | %-10s | %-10s".format("Raw Tag", "Chinese", "English", "Current"))
        writer.println("  ${"-".repeat(12)} | ${"-".repeat(10)} | ${"-".repeat(10)} | ${"-".repeat(10)}")
        
        info.sampleTags.forEach { sample ->
            writer.println("  %-12s | %-10s | %-10s | %-10s".format(
                sample.rawTag,
                sample.chineseDisplay,
                sample.englishDisplay,
                sample.currentDisplay
            ))
        }
        
        writer.println()
        if (!info.isChinese && info.deviceLocale.startsWith("zh")) {
            writer.println("BUG DETECTED: Device locale is Chinese (${info.deviceLocale}) but isChinese=false")
            writer.println("This causes tags to display in English instead of Chinese.")
        } else if (info.isChinese) {
            writer.println("OK: Tags should display in Chinese")
        } else {
            writer.println("OK: Tags should display in English (non-Chinese locale)")
        }
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
        val matches = controller.getModelIdModelMap()
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
            val tags = controller.getModelTags(modelId)
            writer.println("  - $modelId")
            writer.println("    modelName: ${model.modelName ?: ""}")
            writer.println("    tags: ${if (tags.isEmpty()) "[]" else tags.joinToString(prefix = "[", postfix = "]")}")
        }
    }

    private fun doFiles(writer: PrintStream) {
        val entries = controller.scanStorageEntries()
        if (entries.isEmpty()) {
            writer.println("No entries found in .mnnmodels/")
            return
        }

        writer.println("=== Storage Entries (${entries.size}) ===")
        writer.println()

        var currentContainer = ""
        for (entry in entries) {
            if (entry.container != currentContainer) {
                currentContainer = entry.container
                writer.println("[$currentContainer]")
            }
            val typeStr = if (entry.isSymlink) "SYMLINK" else if (entry.isDirectory) "DIR" else "FILE"
            writer.println("  ${entry.name} [$typeStr]")
            if (entry.modelId != null) {
                writer.println("    Model ID: ${entry.modelId}")
            }
            if (entry.isSymlink) {
                writer.println("    Target: ${entry.symlinkTarget ?: "unknown"}")
                writer.println("    Target Exists: ${entry.targetExists}")
            }
        }
    }

    private fun doUnlink(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp models unlink <modelId|name>")
            writer.println("Only removes the outermost symlink, preserving actual model files.")
            writer.println()
            writer.println("Examples:")
            writer.println("  dumpapp models unlink ModelScope/MNN/stable-diffusion-v1-5")
            writer.println("  dumpapp models unlink stable-diffusion-v1-5")
            return
        }

        val modelRef = args.joinToString(" ")
        val entries = controller.scanStorageEntries()

        val entry = entries.find { it.modelId == modelRef }
            ?: entries.find { it.name == modelRef }

        if (entry == null) {
            writer.println("No entry found for: $modelRef")
            writer.println("Use 'dumpapp models files' to list all entries.")
            return
        }

        if (!entry.isSymlink) {
            writer.println("Entry is not a symlink: ${entry.absolutePath}")
            writer.println("Only symlinks can be unlinked. This is a ${if (entry.isDirectory) "directory" else "file"}.")
            return
        }

        val success = controller.deleteSymlink(entry.absolutePath)
        if (success) {
            writer.println("Unlinked: ${entry.name}")
            writer.println("  Removed symlink: ${entry.absolutePath}")
            writer.println("  Target preserved: ${entry.symlinkTarget}")
        } else {
            writer.println("Failed to unlink: ${entry.absolutePath}")
        }
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp models <command>")
        writer.println("Commands:")
        writer.println("  list [-v|--verbose]   - List all models matching UI display")
        writer.println("  dump                  - Dump current model list state (raw)")
        writer.println("  refresh               - Trigger manual refresh")
        writer.println("  tags [modelId|--all]  - Dump tags/extraTags for one or all models")
        writer.println("  tagslocale            - Debug tag locale display (zh_CN/zh_TW issue)")
        writer.println("  find <keyword>        - Find models by id/name and print tags")
        writer.println("  files                 - List all entries in .mnnmodels/ with symlink info")
        writer.println("  unlink <modelId|name> - Remove outermost symlink only (preserves data)")
        writer.println()
        writer.println("Examples:")
        writer.println("  dumpapp models list           - List all models with display tags")
        writer.println("  dumpapp models list -v        - Verbose output with all details")
        writer.println("  dumpapp models tags           - Dump tags/extraTags for all models")
        writer.println("  dumpapp models tagslocale     - Check why tags show English on Chinese device")
        writer.println("  dumpapp models tags ModelScope/MNN/Qwen3-0.6B-MNN")
        writer.println("  dumpapp models files          - Show storage entries with symlink info")
        writer.println("  dumpapp models unlink ModelScope/MNN/stable-diffusion-v1-5")
    }
}
