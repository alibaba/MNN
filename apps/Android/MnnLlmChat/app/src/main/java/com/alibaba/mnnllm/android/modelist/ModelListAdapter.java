// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelist;

import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.alibaba.mls.api.HfRepoItem;
import com.alibaba.mnnllm.android.R;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class ModelListAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    private List<HfRepoItem> items;
    private List<HfRepoItem> filteredItems;
    private ModelItemListener modelListListener;
    private Map<String, ModelItemState> modelItemStatesMap;
    private Set<ModelItemHolder> modelItemHolders = new HashSet<>();

    public ModelListAdapter(List<HfRepoItem> items) {
        this.items = items;
    }

    void setModelListListener(ModelItemListener modelListListener) {
        this.modelListListener = modelListListener;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.recycle_item_model, parent, false);
        ModelItemHolder holder = new ModelItemHolder(view, modelListListener);
        modelItemHolders.add(holder);
        return holder;
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        ((ModelItemHolder)holder).bind(getItems().get(position), modelItemStatesMap.get(getItems().get(position).getModelId()));
    }


    @Override
    public int getItemCount() {
        return getItems().size();
    }

    public void updateItems(List<HfRepoItem> hfRepoItems, Map<String, ModelItemState> modelItemStatesMap) {
        this.modelItemStatesMap = modelItemStatesMap;
        this.items.clear();
        this.items.addAll(hfRepoItems);
        notifyDataSetChanged();
    }

    public void updateItem(String modelId) {
        List<HfRepoItem> currentItems = getItems();
        int position = -1;
        for (int i = 0; i < currentItems.size(); i++) {
            if (currentItems.get(i).getModelId().equals(modelId)) {
                position = i;
                break;
            }
        }
        if (position >= 0) {
            notifyItemChanged(position);
        }
    }

    public void updateProgres(String modelId, double progress) {
        for (ModelItemHolder modelItemHolder : modelItemHolders) {
            if (modelItemHolder.itemView.getTag() == null) {
                continue;
            }
            String tempModelId = ((HfRepoItem)modelItemHolder.itemView.getTag()).getModelId();
            if (TextUtils.equals(tempModelId, modelId)) {
                modelItemHolder.updateProgress(progress);
            }
        }
    }

    public List<HfRepoItem> getItems() {
        return filteredItems != null ? filteredItems : items;
    }

    public void filter(String query) {
        List<HfRepoItem> filtered = this.items.stream()
                .filter(hfRepoItem -> {
                    String modelName = hfRepoItem.getModelName().toLowerCase();
                    return modelName.contains(query.toLowerCase()) ||
                            hfRepoItem.getNewTags().stream().anyMatch(
                                    tag -> tag.toLowerCase().contains(query.toLowerCase())
                            );
                })
                .collect(Collectors.toList());
        if (filtered.size() != this.items.size()) {
            this.filteredItems = filtered;
        } else {
            this.filteredItems = null;
        }
        notifyDataSetChanged();
    }

    public void unfilter() {
        this.filteredItems = null;
        notifyDataSetChanged();
    }
}
