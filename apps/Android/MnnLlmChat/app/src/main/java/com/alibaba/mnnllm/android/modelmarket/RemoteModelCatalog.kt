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
    private const val HF_LIST_URL = "https://huggingface.co/api/models?author=taobao-mnn&limit=1000&full=true"

    private const val MS_OWNER = "MNN"
    private const val MS_PAGE_SIZE = 50
    private const val MS_MAX_PAGES = 40
    private const val MS_FILE_CONCURRENCY = 8

    private val gson = Gson()

    // ==================== Public entry point ====================

    /**
     * Enumerate both hubs, verify package format and merge into market items.
     * [labels] are the curated CDN entries keyed by model name; they may be null.
     * Returns null when neither hub could be reached so the caller can fall back
     * to the CDN index.
     */
    suspend fun fetchCatalog(
        client: OkHttpClient,
        labels: List<ModelMarketItem>?
    ): List<ModelMarketItem>? = withContext(Dispatchers.IO) {
        try {
            val ms = listModelScope(client)
            val hf = listHuggingFace(client)
            if (ms.isEmpty() && hf.isEmpty()) {
                Log.w(TAG, "fetchCatalog: both hubs returned nothing")
                return@withContext null
            }
            val labelsByName = labels?.associateBy { it.modelName } ?: emptyMap()
            val msByName = ms.associateBy { it.name }
            val hfByName = hf.associateBy { it.name }

            // HuggingFace already ships the file list (siblings) in the list response,
            // so only ModelScope-exclusive candidates need an extra repo/files call.
            val filesByName = HashMap<String, Set<String>>()
            val needMsFiles = ArrayList<String>()
            for (name in (msByName.keys + hfByName.keys).toSet()) {
                val hfFiles = hfByName[name]?.fileNames()
                if (!hfFiles.isNullOrEmpty()) {
                    filesByName[name] = hfFiles
                } else if (msByName.containsKey(name)) {
                    needMsFiles.add(name)
                }
            }
            if (needMsFiles.isNotEmpty()) {
                val semaphore = Semaphore(MS_FILE_CONCURRENCY)
                coroutineScope {
                    needMsFiles.map { name ->
                        async {
                            semaphore.withPermit {
                                val files = fetchModelScopeFiles(client, msByName[name]!!.id)
                                if (files != null) name to files else null
                            }
                        }
                    }.awaitAll().filterNotNull().forEach { (name, files) -> filesByName[name] = files }
                }
            }
            val merged = merge(ms, hf, filesByName, labelsByName)
            Log.i(
                TAG,
                "fetchCatalog: ms=${ms.size} hf=${hf.size} files=${filesByName.size} merged=${merged.size}"
            )
            merged
        } catch (e: Exception) {
            Log.e(TAG, "fetchCatalog failed", e)
            null
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

        val tags = label?.tags?.takeIf { it.isNotEmpty() } ?: apiTags(ms, hf)
        val categories = label?.categories?.takeIf { it.isNotEmpty() }
            ?: (ms?.tasks?.takeIf { it.isNotEmpty() } ?: listOfNotNull(hf?.pipelineTag))

        return ModelMarketItem(
            modelName = name,
            vendor = label?.vendor ?: "",
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

    private fun apiTags(ms: MsRepo?, hf: HfRepo?): List<String> {
        val tags = LinkedHashSet<String>()
        ms?.tasks?.let { tags.addAll(it) }
        ms?.tags?.forEach { if (it.startsWith("custom_tag:")) tags.add(it.removePrefix("custom_tag:")) }
        if (tags.isEmpty()) {
            hf?.tags?.forEach { if (!it.contains(":")) tags.add(it) }
        }
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
        val out = ArrayList<MsRepo>()
        var page = 1
        while (page <= MS_MAX_PAGES) {
            val url = "$MS_LIST_URL?owner=$MS_OWNER&page_size=$MS_PAGE_SIZE&page_number=$page"
            val body = httpGet(client, url) ?: break
            val data = try {
                gson.fromJson(body, MsPage::class.java)?.data
            } catch (e: Exception) {
                Log.w(TAG, "ModelScope page $page parse failed: ${e.message}")
                null
            } ?: break
            if (data.pageNumber != page) break
            out.addAll(data.models)
            if (data.models.isEmpty() || page.toLong() * MS_PAGE_SIZE >= data.totalCount) break
            page++
        }
        out
    }

    private suspend fun listHuggingFace(client: OkHttpClient): List<HfRepo> = withContext(Dispatchers.IO) {
        val body = httpGet(client, HF_LIST_URL) ?: return@withContext emptyList()
        try {
            gson.fromJson(body, Array<HfRepo>::class.java).toList()
        } catch (e: Exception) {
            Log.w(TAG, "HuggingFace list parse failed: ${e.message}")
            emptyList()
        }
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
