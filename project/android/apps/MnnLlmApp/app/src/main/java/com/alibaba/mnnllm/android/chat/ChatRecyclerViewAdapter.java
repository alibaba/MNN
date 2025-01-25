// Created by ruoyi.sjd on 2025/01/03.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import static com.alibaba.mnnllm.android.chat.ChatViewHolders.ASSISTANT;
import static com.alibaba.mnnllm.android.chat.ChatViewHolders.HEADER;
import static com.alibaba.mnnllm.android.chat.ChatViewHolders.USER;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.alibaba.mnnllm.android.R;

import java.util.List;

public class ChatRecyclerViewAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {
    private final List<ChatDataItem> items;
    private final String modelName;

    public ChatRecyclerViewAdapter(Context context, List<ChatDataItem> items, String modelName) {
        this.items = items;
        this.modelName = modelName;
    }

    @Override
    public int getItemCount() { return items.size(); }

    @Override
    public int getItemViewType(int position) {
        return items.get(position).getType();
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(parent.getContext());
        View view;
        switch (viewType) {
            case HEADER:
                view = inflater.inflate(R.layout.item_holder_chatheader, parent, false);
                return new ChatViewHolders.HeaderViewHolder(view);
            case ASSISTANT:
                view = inflater.inflate(R.layout.item_holder_assistant, parent, false);
                return new ChatViewHolders.AssistantViewHolder(view);
            case USER:
                default:
                view = inflater.inflate(R.layout.item_holder_user, parent, false);
                return new ChatViewHolders.UserViewHolder(view);
        }
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        int viewType = getItemViewType(position);
        if (viewType == HEADER) {
            ((ChatViewHolders.HeaderViewHolder)holder).bind(items.get(position));
        } else if (viewType == ASSISTANT) {
            ((ChatViewHolders.AssistantViewHolder)holder).bind(items.get(position), modelName);
        } else if (viewType == USER) {
            ((ChatViewHolders.UserViewHolder)holder).bind(items.get(position));
        }
    }

    public void addItem(ChatDataItem item) {
        items.add(item);
        notifyItemInserted(items.size() - 1);
    }

    public ChatDataItem getRecentItem() {
        return !items.isEmpty() ? items.get(items.size() - 1) : null;
    }

    public void updateRecentItem(ChatDataItem item) {
        notifyItemChanged(items.size() - 1);
    }

    public boolean reset() {
        if (items.size() > 2) {
            int size = items.size();
            items.subList(2, size).clear();
            notifyItemRangeRemoved(2, size - 2);
            return true;
        }
        return false;
    }
}
