// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.mainsettings

import com.alibaba.mnnllm.android.modelist.ModelDeletionHelper

/**
 * Item for the storage list RecyclerView: either a group (model entry) or a child (storage unit with delete).
 */
sealed class StorageListItem {

    data class Group(
        val entry: ModelDeletionHelper.MmapCacheEntry,
        val expanded: Boolean,
        val detail: ModelDeletionHelper.ModelStorageDetail? = null
    ) : StorageListItem()

    enum class ChildType { MODEL_DIR, CONFIG_DIR, MMAP_DIR }

    data class Child(
        val type: ChildType,
        val label: String,
        val path: String,
        val sizeBytes: Long?,
        val resolvedModelId: String?,
        val entry: ModelDeletionHelper.MmapCacheEntry
    ) : StorageListItem()
}
