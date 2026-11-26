// Created for direct ModelScope/HuggingFace catalog discovery.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.modelmarket

import android.util.Log
import com.alibaba.mls.api.source.ModelSources
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.ConcurrentHashMap

/**
 * Discovers the model-market catalog straight from the ModelScope and HuggingFace
 * repository APIs, mirroring the iOS approach in alibaba/MNN#5132.
 *
 * The two hubs decide *membership* (which models exist) and *freshness*
 * (file_size / params). The bundled CDN index (model_market.json) is used only as
 * optional "labels": curated tags, categories, vendor, size, description and the
 * Modelers source. When both APIs are unreachable the caller falls back to the CDN
 * index entirely, so the market still renders offline.
 *
 * Format support follows the packages the app can actually run:
 * - Sana image editing: llm/ + connector.mnn + vae_encoder.mnn + transformer.mnn
 * - Stable Diffusion:    text_encoder.mnn + unet.mnn + vae_decoder.mnn
 * - LLM / multimodal:    config.json + llm.mnn
 */
object RemoteModelCatalog {

    private const val TAG = "RemoteModelCatalog"

    // Deliberately generous; larger packages are excluded from the phone market.
    private const val MAX_FILE_SIZE = 16L * 1024 * 1024 * 1024

    private const val MS_LIST_URL = "https://modelscope.cn/openapi/v1/models"
    private const val MS_FILES_URL = "https://modelscope.cn/api/v1/models/%s/repo/files?Root=&Revision=master"

    private const val MS_OWNER = "MNN"
    private const val MS_PAGE_SIZE = 50
    private const val MS_MAX_PAGES = 40
    private const val MS_FILE_CONCURRENCY = 8

    private val gson = Gson()

    /**
     * In-memory verification snapshot: modelName -> (lastModified, verified file list).
     * Repos whose hub `last_modified` matches skip the per-repo file-list call entirely,
     * which turns a cold ~200-request crawl into a warm delta check.
     */
    data class SnapshotEntry(val lastModified: String?, val files: Set<String>)

    private val fileSnapshot = ConcurrentHashMap<String, SnapshotEntry>()

    /** Replace the snapshot from a persisted copy (called by ModelRepository on load). */
    fun restoreSnapshot(entries: Map<String, SnapshotEntry>) {
        fileSnapshot.clear()
        fileSnapshot.putAll(entries)
    }

    /** Current snapshot for persistence. */
    fun snapshot(): Map<String, SnapshotEntry> = HashMap(fileSnapshot)

    // ==================== Public entry point ====================

    /**
     * Enumerate both hubs, verify package format and merge into market items.
     * [labels] are the curated CDN entries keyed by model name; they may be null.
     * [onPartial] (optional) receives an intermediate catalog as soon as the hub
     * listings are in and the snapshot-verified subset is merged, so the UI can
     * render most of the list while the remaining repos are still being verified.
     * Returns null when neither hub could be reached so the caller can fall back
     * to the CDN index.
     */
    suspend fun fetchCatalog(
        client: OkHttpClient,
        labels: List<ModelMarketItem>?,
        onPartial: (suspend (List<ModelMarketItem>) -> Unit)? = null
    ): List<ModelMarketItem>? = withContext(Dispatchers.IO) {
        try {
            val t0 = System.currentTimeMillis()
            // Enumerate both hubs in parallel; either one alone is enough to proceed.
            val (ms, hf) = coroutineScope {
                val msDeferred = async { listModelScope(client) }
                val hfDeferred = async { listHuggingFace(client) }
                msDeferred.await() to hfDeferred.await()
            }
            Log.i(TAG, "fetchCatalog: listing done in ${System.currentTimeMillis() - t0}ms ms=${ms.size} hf=${hf.size}")
            if (ms.isEmpty() && hf.isEmpty()) {
                Log.w(TAG, "fetchCatalog: both hubs returned nothing")
                return@withContext null
            }
            val labelsByName = labels?.associateBy { it.modelName } ?: emptyMap()
            val msByName = ms.associateBy { it.name }
            val hfByName = hf.associateBy { it.name }

            // HuggingFace already ships the file list (siblings) in the list response,
            // so only ModelScope-exclusive candidates need an extra repo/files call.
            // The snapshot lets unchanged repos reuse the previously verified file list.
            val filesByName = HashMap<String, Set<String>>()
            val needMsFiles = ArrayList<String>()
            for (name in (msByName.keys + hfByName.keys).toSet()) {
                val hfFiles = hfByName[name]?.fileNames()
                if (!hfFiles.isNullOrEmpty()) {
                    filesByName[name] = hfFiles
                    rememberSnapshot(name, hfByName[name]?.lastModified, hfFiles)
                    continue
                }
                val msRepo = msByName[name] ?: continue
                val cached = fileSnapshot[name]
                if (cached != null && cached.files.isNotEmpty() &&
                    msRepo.lastModified != null && msRepo.lastModified == cached.lastModified
                ) {
                    filesByName[name] = cached.files
                } else {
                    needMsFiles.add(name)
                }
            }

            // Phase 1: emit everything already verified (HF siblings + snapshot hits)
            // so the market renders within one round-trip instead of waiting for the
            // per-repo verification crawl.
            if (onPartial != null && filesByName.isNotEmpty()) {
                val partial = merge(ms, hf, filesByName, labelsByName)
                Log.i(TAG, "fetchCatalog: partial emit ${partial.size} items, ${needMsFiles.size} repos still verifying")
                onPartial(partial)
            }

            // Phase 2: verify the remaining ModelScope-exclusive repos.
            if (needMsFiles.isNotEmpty()) {
                val semaphore = Semaphore(MS_FILE_CONCURRENCY)
                coroutineScope {
                    needMsFiles.map { name ->
                        async {
                            semaphore.withPermit {
                                val files = fetchModelScopeFiles(client, msByName[name]!!.id)
                                if (files != null) {
                                    rememberSnapshot(name, msByName[name]?.lastModified, files)
                                    name to files
                                } else null
                            }
                        }
                    }.awaitAll().filterNotNull().forEach { (name, files) -> filesByName[name] = files }
                }
            }
            val merged = merge(ms, hf, filesByName, labelsByName)
            Log.i(
                TAG,
                "fetchCatalog: total ${System.currentTimeMillis() - t0}ms files=${filesByName.size} merged=${merged.size}"
            )
            merged
        } catch (e: Exception) {
            Log.e(TAG, "fetchCatalog failed", e)
            null
        }
    }

    private fun rememberSnapshot(name: String, lastModified: String?, files: Set<String>) {
        if (files.isNotEmpty()) {
            fileSnapshot[name] = SnapshotEntry(lastModified, files)
        }
    }

    // ==================== Pure merge / filter logic (unit-testable) ====================

    /**
     * Build the catalog from both hubs. A model is kept when it is eligible and its
     * package layout is supported. CDN labels enrich tags/vendor/size when present.
     */
    internal fun merge(
        msRepos: List<MsRepo>,
        hfRepos: List<HfRepo>,
        filesByName: Map<String, Set<String>>,
        labelsByName: Map<String, ModelMarketItem>
    ): List<ModelMarketItem> {
        val msByName = msRepos.associateBy { it.name }
        val hfByName = hfRepos.associateBy { it.name }
        val names = (msByName.keys + hfByName.keys).toSortedSet()
        val out = ArrayList<ModelMarketItem>(names.size)
        for (name in names) {
            if (name.isEmpty() || name == "." || name == "..") continue
            val ms = msByName[name]
            val hf = hfByName[name]
            val fileSize = ms?.fileSize ?: labelsByName[name]?.fileSize ?: 0L
            val gated = truthy(ms?.gated) || truthy(hf?.gated)
            val isPrivate = truthy(ms?.isPrivate) || truthy(hf?.isPrivate)
            if (!isEligible(name, fileSize, gated, isPrivate)) continue
            val files = filesByName[name] ?: emptySet()
            if (!supportsFormat(name, files)) continue
            out.add(toItem(name, ms, hf, labelsByName[name]))
        }
        return out
    }

    /** Gate out private/gated repos, oversized packages and NPU-only builds. */
    internal fun isEligible(name: String, fileSize: Long, gated: Boolean, isPrivate: Boolean): Boolean {
        if (gated || isPrivate) return false
        if (fileSize < 0 || fileSize > MAX_FILE_SIZE) return false
        if (name.lowercase().endsWith("-npu")) return false
        return true
    }

    /** Match the packages the chat, Sana and diffusion entry points can load. */
    internal fun supportsFormat(name: String, files: Set<String>): Boolean {
        if (files.isEmpty()) return false
        val lower = name.lowercase()
        if (lower.contains("sana")) {
            return listOf("llm", "connector.mnn", "vae_encoder.mnn", "transformer.mnn").all(files::contains)
        }
        if (lower.contains("stable-diffusion")) {
            return listOf("text_encoder.mnn", "unet.mnn", "vae_decoder.mnn").all(files::contains)
        }
        return files.contains("config.json") && files.contains("llm.mnn")
    }

    /**
     * Compose one market item. CDN labels win for curated fields; API data fills
     * sources, size and fallback tags so newly discovered repos are usable.
     */
    internal fun toItem(
        name: String,
        ms: MsRepo?,
        hf: HfRepo?,
        label: ModelMarketItem?
    ): ModelMarketItem {
        val sources = LinkedHashMap<String, String>()
        label?.sources?.let { sources.putAll(it) }
        ms?.let { sources[ModelSources.sourceModelScope] = it.id }
        hf?.let { sources[ModelSources.sourceHuffingFace] = it.id }

        val paramsBillions = (ms?.params ?: 0L) / 1e9
        val sizeB = (label?.sizeB ?: 0.0).takeIf { it > 0.0 } ?: paramsBillions
        val fileSize = (ms?.fileSize ?: 0L).takeIf { it > 0L } ?: (label?.fileSize ?: 0L)

        val tags = (label?.tags.orEmpty() + apiTags(ms, hf)).distinct()
        val categories = label?.categories?.takeIf { it.isNotEmpty() }
            ?: (ms?.tasks?.takeIf { it.isNotEmpty() } ?: listOfNotNull(hf?.pipelineTag))

        return ModelMarketItem(
            modelName = name,
            vendor = label?.vendor?.takeIf { it.isNotBlank() } ?: inferVendor(name),
            sizeB = sizeB,
            tags = tags,
            categories = categories,
            sources = sources,
            minAppVersion = label?.minAppVersion,
            description = label?.description,
            fileSize = fileSize,
            extraTags = label?.extraTags ?: emptyList()
        )
    }

    /**
     * Family-prefix vendor inference for API-discovered models without CDN labels.
     * Mirrors the curated CDN vendor taxonomy so the vendor filter never shows blanks.
     * Longest prefix wins (e.g. "InternVL" before "Intern").
     */
    private val VENDOR_PREFIXES: List<Pair<String, String>> = listOf(
        "Baichuan" to "Baichuan",
        "DeepSeek" to "DeepSeek",
        "ERNIE" to "ERNIE",
        "FastVLM" to "FastVLM",
        "Fun-Audio" to "FunAudio",
        "GUI-Owl" to "Qwen",
        "Hunyuan" to "Hunyuan",
        "InternVL" to "InternLM",
        "InternLM" to "InternLM",
        "LFM" to "LFM",
        "Lingshu" to "Lingshu",
        "Llama" to "Llama",
        "Meta-Llama" to "Llama",
        "MiMo" to "MiMo",
        "MiniCPM" to "MiniCPM",
        "MiniMind" to "Nanbeige",
        "MNN-Pocket" to "MNN",
        "MobileLLM" to "MobileLLM",
        "Nanbeige" to "Nanbeige",
        "Phi" to "Phi",
        "Qwen" to "Qwen",
        "QwQ" to "Qwen",
        "Smol" to "Smol",
        "THUDM" to "THUDM",
        "TinyLlama" to "TinyLlama",
        "VibeThinker" to "VibeThinker",
        "YuFeng" to "YuFeng",
        "Gemma" to "Gemma",
        "GPT" to "GPT",
        "stable-diffusion" to "stability.ai",
        "MNN-Sana" to "stability.ai"
    ).sortedByDescending { it.first.length }

    internal fun inferVendor(name: String): String {
        for ((prefix, vendor) in VENDOR_PREFIXES) {
            if (name.startsWith(prefix, ignoreCase = true)) return vendor
        }
        return "Other"
    }

    private fun apiTags(ms: MsRepo?, hf: HfRepo?): List<String> {
        val tags = LinkedHashSet<String>()
        ms?.tasks?.let { tags.addAll(it) }
        ms?.tags?.forEach { tag ->
            tags.add(tag)
            if (tag.startsWith("custom_tag:")) tags.add(tag.removePrefix("custom_tag:"))
        }
        hf?.pipelineTag?.let { tags.add(it) }
        hf?.tags?.let { tags.addAll(it) }
        return tags.toList()
    }

    /** HuggingFace reports `gated` as false/true/"auto"; ModelScope uses a boolean. */
    private fun truthy(value: Any?): Boolean = when (value) {
        is Boolean -> value
        is String -> value.equals("true", true) || value.equals("auto", true)
        else -> false
    }

    // ==================== Network ====================

    private suspend fun listModelScope(client: OkHttpClient): List<MsRepo> = withContext(Dispatchers.IO) {
        // Page 1 discovers total_count; remaining pages are fetched in parallel.
        fun pageUrl(page: Int) = "$MS_LIST_URL?owner=$MS_OWNER&page_size=$MS_PAGE_SIZE&page_number=$page"
        fun parsePage(page: Int, body: String?): MsPageData? {
            if (body == null) return null
            return try {
                gson.fromJson(body, MsPage::class.java)?.data?.takeIf { it.pageNumber == page }
            } catch (e: Exception) {
                Log.w(TAG, "ModelScope page $page parse failed: ${e.message}")
                null
            }
        }

        val first = parsePage(1, httpGet(client, pageUrl(1))) ?: return@withContext emptyList()
        val out = ArrayList<MsRepo>(first.models)
        val lastPage = minOf(
            MS_MAX_PAGES,
            ((first.totalCount + MS_PAGE_SIZE - 1) / MS_PAGE_SIZE).coerceAtLeast(1)
        )
        if (lastPage > 1) {
            val semaphore = Semaphore(MS_FILE_CONCURRENCY)
            coroutineScope {
                (2..lastPage).map { page ->
                    async {
                        semaphore.withPermit {
                            page to parsePage(page, httpGet(client, pageUrl(page)))?.models
                        }
                    }
                }.awaitAll()
                    .sortedBy { it.first }
                    .forEach { (_, models) -> if (models != null) out.addAll(models) }
            }
        }
        out
    }

    private suspend fun listHuggingFace(client: OkHttpClient): List<HfRepo> = withContext(Dispatchers.IO) {
        // Primary endpoint first; hf-mirror.com mirrors the same API for networks
        // where huggingface.co is slow or blocked.
        for (base in listOf("https://huggingface.co", "https://hf-mirror.com")) {
            val body = httpGet(client, "$base/api/models?author=taobao-mnn&limit=1000&full=true") ?: continue
            try {
                val repos = gson.fromJson(body, Array<HfRepo>::class.java).toList()
                if (repos.isNotEmpty()) {
                    Log.i(TAG, "HuggingFace list from $base: ${repos.size}")
                    return@withContext repos
                }
            } catch (e: Exception) {
                Log.w(TAG, "HuggingFace list parse failed ($base): ${e.message}")
            }
        }
        emptyList()
    }

    private suspend fun fetchModelScopeFiles(client: OkHttpClient, repoId: String): Set<String>? =
        withContext(Dispatchers.IO) {
            val body = httpGet(client, String.format(MS_FILES_URL, repoId)) ?: return@withContext null
            try {
                gson.fromJson(body, MsFiles::class.java)?.data?.files?.map { it.name }?.toSet()
            } catch (e: Exception) {
                Log.w(TAG, "ModelScope files parse failed for $repoId: ${e.message}")
                null
            }
        }

    private fun httpGet(client: OkHttpClient, url: String): String? {
        return try {
            val request = Request.Builder()
                .url(url)
                .addHeader("Cache-Control", "no-cache, no-store, must-revalidate")
                .build()
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) response.body?.string() else null
            }
        } catch (e: Exception) {
            Log.w(TAG, "GET $url failed: ${e.message}")
            null
        }
    }

    // ==================== DTOs ====================

    internal data class MsRepo(
        val id: String,
        @SerializedName("last_modified") val lastModified: String? = null,
        @SerializedName("file_size") val fileSize: Long = 0L,
        val params: Long = 0L,
        val tasks: List<String>? = null,
        val tags: List<String>? = null,
        val gated: Any? = null,
        @SerializedName("private") val isPrivate: Any? = null
    ) {
        val name: String get() = id.substringAfterLast('/')
    }

    internal data class HfRepo(
        val id: String,
        @SerializedName("lastModified") val lastModified: String? = null,
        @SerializedName("pipeline_tag") val pipelineTag: String? = null,
        val tags: List<String>? = null,
        val gated: Any? = null,
        @SerializedName("private") val isPrivate: Any? = null,
        val siblings: List<HfSibling>? = null
    ) {
        val name: String get() = id.substringAfterLast('/')
        fun fileNames(): Set<String> = siblings?.map { it.rfilename }?.toSet() ?: emptySet()
    }

    internal data class HfSibling(@SerializedName("rfilename") val rfilename: String)

    private data class MsPageData(
        val models: List<MsRepo> = emptyList(),
        @SerializedName("total_count") val totalCount: Int = 0,
        @SerializedName("page_number") val pageNumber: Int = 0
    )

    private data class MsPage(val data: MsPageData? = null)

    private data class MsFileEntry(@SerializedName("Name") val name: String)
    private data class MsFilesData(@SerializedName("Files") val files: List<MsFileEntry>? = null)
    private data class MsFiles(@SerializedName("Data") val data: MsFilesData? = null)
}
