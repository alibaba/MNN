package com.alibaba.mnnllm.android.modelist

import com.alibaba.mls.api.ModelItem

interface ModelItemListener {
    fun onItemClicked(hfModelItem: ModelItem)
}
