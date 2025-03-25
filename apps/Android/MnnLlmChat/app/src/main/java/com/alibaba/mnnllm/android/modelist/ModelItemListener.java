package com.alibaba.mnnllm.android.modelist;

import com.alibaba.mls.api.HfRepoItem;

public interface ModelItemListener {
    void onItemClicked(HfRepoItem hfRepoItem);
}
