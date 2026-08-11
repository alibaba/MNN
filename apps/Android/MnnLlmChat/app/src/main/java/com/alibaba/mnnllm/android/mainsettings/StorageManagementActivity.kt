// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.mainsettings

import android.os.Bundle
import android.view.MenuItem
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.recyclerview.widget.LinearLayoutManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.ActivityStorageManagementBinding
import com.alibaba.mnnllm.android.modelist.ModelDeletionHelper
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.utils.MmapUtils
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.io.File

/**
 * Displays storage usage grouped by model. Tap a group to expand and show indented child rows
 * (model dir, config dir, mmap dir); each child has a delete button. Bottom button cleans all cleanable cache with confirmation.
 */
class StorageManagementActivity : AppCompatActivity() {

    private lateinit var binding: ActivityStorageManagementBinding
    private lateinit var adapter: StorageListAdapter

    /** Which entry.modelId are expanded; value is cached detail when loaded. */
    private val expandedDetails = mutableMapOf<String, ModelDeletionHelper.ModelStorageDetail?>()
    private val expandedSet = mutableSetOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityStorageManagementBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val toolbar: Toolbar = binding.toolbar
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        title = getString(R.string.storage_management)

        adapter = StorageListAdapter(
            formatSize = { formatSize(it) },
            getStatusCleanable = { getString(R.string.storage_status_orphan) },
            getStatusTracked = { getString(R.string.storage_status_tracked) },
            onGroupClick = { group -> onGroupClick(group) },
            onChildDelete = { child -> onChildDelete(child) }
        )
        binding.storageRecycler.layoutManager = LinearLayoutManager(this)
        binding.storageRecycler.adapter = adapter

        binding.btnCleanCleanableCache.setOnClickListener { showCleanCleanableConfirm() }

        loadSummary()
        refreshList()
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == android.R.id.home) {
            finish()
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    private fun formatSize(bytes: Long): String {
        return when {
            bytes >= 1024L * 1024L * 1024L -> String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
            bytes >= 1024L * 1024L -> String.format("%.2f MB", bytes / (1024.0 * 1024.0))
            bytes >= 1024L -> String.format("%.2f KB", bytes / 1024.0)
            else -> "$bytes B"
        }
    }

    private fun loadSummary() {
        val analysis = ModelDeletionHelper.getStorageAnalysis(applicationContext)
        val free = analysis.internalStorageTotal - analysis.internalStorageUsed

        binding.tvInternalTotal.text = formatSize(analysis.internalStorageTotal)
        binding.tvInternalUsed.text = formatSize(analysis.internalStorageUsed)
        binding.tvInternalFree.text = formatSize(free.coerceAtLeast(0))

        binding.tvModelFiles.text = formatSize(analysis.modelStorageSize)
        binding.tvMmapCache.text = formatSize(analysis.totalMmapCacheSize)
        binding.tvOrphanCache.text = formatSize(analysis.totalOrphanSize)
    }

    private fun buildFlatList(): List<StorageListItem> {
        val analysis = ModelDeletionHelper.getStorageAnalysis(applicationContext)
        val entries = analysis.mmapCacheEntries.sortedByDescending { it.sizeBytes }
        val list = mutableListOf<StorageListItem>()

        for (entry in entries) {
            val expanded = expandedSet.contains(entry.modelId)
            var detail = expandedDetails[entry.modelId]
            if (expanded && detail == null) {
                detail = ModelDeletionHelper.getStorageDetailForEntry(this, entry)
                expandedDetails[entry.modelId] = detail
            }
            list.add(StorageListItem.Group(entry = entry, expanded = expanded, detail = detail))

            if (expanded && detail != null) {
                // Child: Model directory (only if resolved)
                if (detail.resolvedModelId != null && detail.modelDirPath != null) {
                    list.add(
                        StorageListItem.Child(
                            type = StorageListItem.ChildType.MODEL_DIR,
                            label = getString(R.string.storage_detail_model_dir) + ": " + detail.modelDirPath,
                            path = detail.modelDirPath,
                            sizeBytes = detail.modelDirSize,
                            resolvedModelId = detail.resolvedModelId,
                            entry = entry
                        )
                    )
                }
                // Child: Config directory
                if (detail.resolvedModelId != null && detail.configDirPath != null) {
                    val configSize = detail.configFiles.sumOf { it.second }
                    list.add(
                        StorageListItem.Child(
                            type = StorageListItem.ChildType.CONFIG_DIR,
                            label = getString(R.string.storage_detail_config_dir) + ": " + detail.configDirPath,
                            path = detail.configDirPath,
                            sizeBytes = configSize,
                            resolvedModelId = detail.resolvedModelId,
                            entry = entry
                        )
                    )
                }
                // Child: Mmap directory (always present)
                list.add(
                    StorageListItem.Child(
                        type = StorageListItem.ChildType.MMAP_DIR,
                        label = getString(R.string.storage_detail_mmap_dir) + ": " + detail.mmapPath,
                        path = detail.mmapPath,
                        sizeBytes = detail.mmapSize,
                        resolvedModelId = detail.resolvedModelId,
                        entry = entry
                    )
                )
            }
        }
        return list
    }

    private fun refreshList() {
        adapter.submitList(buildFlatList())
    }

    private fun onGroupClick(group: StorageListItem.Group) {
        if (group.expanded) {
            expandedSet.remove(group.entry.modelId)
        } else {
            expandedSet.add(group.entry.modelId)
        }
        refreshList()
    }

    private fun onChildDelete(child: StorageListItem.Child) {
        when (child.type) {
            StorageListItem.ChildType.MODEL_DIR -> {
                val modelId = child.resolvedModelId
                if (modelId != null) {
                    val result = ModelDeletionHelper.deleteModelWithCleanup(this, modelId)
                    if (!result.modelDeleted && result.errors.isNotEmpty()) {
                        Toast.makeText(this, result.errors.joinToString(), Toast.LENGTH_LONG).show()
                    }
                    expandedSet.remove(child.entry.modelId)
                    expandedDetails.remove(child.entry.modelId)
                }
            }
            StorageListItem.ChildType.CONFIG_DIR -> {
                val modelId = child.resolvedModelId
                if (modelId != null) {
                    val dir = File(ModelConfig.getModelConfigDir(modelId))
                    if (dir.exists()) dir.deleteRecursively()
                    expandedDetails[child.entry.modelId] = null
                    refreshList()
                }
            }
            StorageListItem.ChildType.MMAP_DIR -> {
                if (child.entry.isOrphan) {
                    File(child.entry.path).deleteRecursively()
                    expandedSet.remove(child.entry.modelId)
                    expandedDetails.remove(child.entry.modelId)
                } else {
                    child.resolvedModelId?.let { modelId ->
                        MmapUtils.clearMmapCache(modelId)
                        expandedDetails[child.entry.modelId] = null
                        refreshList()
                    }
                }
            }
        }
        loadSummary()
        refreshList()
    }

    private fun showCleanCleanableConfirm() {
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.storage_clean_cleanable_confirm_title)
            .setMessage(R.string.storage_clean_cleanable_confirm_message)
            .setPositiveButton(R.string.storage_clean_cleanable_cache) { _, _ ->
                val result = ModelDeletionHelper.cleanOrphanMmapCaches(applicationContext)
                Toast.makeText(
                    this,
                    getString(R.string.storage_clean_cleanable_cache) + ": " + formatSize(result.bytesFreed),
                    Toast.LENGTH_SHORT
                ).show()
                loadSummary()
                refreshList()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }
}
