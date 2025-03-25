// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.history;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.chat.SessionItem;

import java.util.List;

public class HistoryListAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    private List<SessionItem> historySessionList;
    private OnHistoryCallback onHistoryCallback;

    public HistoryListAdapter() {

    }

    public void setOnHistoryClick(OnHistoryCallback onHistoryCallback) {
        this.onHistoryCallback = onHistoryCallback;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.recycle_item_history, parent, false);
        ViewHolder holder = new ViewHolder(view);
        holder.setOnHistoryClick(new OnHistoryCallback() {
            @Override
            public void onSessionHistoryClick(SessionItem sessionItem) {
                if (HistoryListAdapter.this.onHistoryCallback != null) {
                    HistoryListAdapter.this.onHistoryCallback.onSessionHistoryClick(sessionItem);
                }
            }

            @Override
            public void onSessionHistoryDelete(SessionItem sessionItem) {
                if (HistoryListAdapter.this.onHistoryCallback != null) {
                    HistoryListAdapter.this.onHistoryCallback.onSessionHistoryDelete(sessionItem);
                }
                int index = historySessionList.indexOf(sessionItem);
                historySessionList.remove(index);
                notifyItemRemoved(index);
            }
        });
        return holder;
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        SessionItem sessionItem = historySessionList.get(position);
        ((ViewHolder)holder).bind(sessionItem);
    }

    @Override
    public int getItemCount() {
        return this.historySessionList == null ? 0 : this.historySessionList.size();
    }

    public void updateItems(List<SessionItem> historySessionList) {
        this.historySessionList = historySessionList;
        notifyDataSetChanged();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {

        public View itemView;
        public TextView textHistory;
        public View viewDelete;

        private OnHistoryCallback onHistoryCallback;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            this.itemView = itemView;
            this.itemView.setOnClickListener(this);
            this.viewDelete = itemView.findViewById(R.id.iv_delete_history);
            this.viewDelete.setOnClickListener(this);
            textHistory = itemView.findViewById(R.id.text_history);
        }

        public void bind(SessionItem sessionItem) {
            textHistory.setText(sessionItem.getTitle());
            itemView.setTag(sessionItem);
            viewDelete.setTag(sessionItem);
        }

        @Override
        public void onClick(View v) {
            SessionItem sessionItem = (SessionItem) v.getTag();
            if (v.getId() == R.id.iv_delete_history) {
                if (onHistoryCallback != null) {
                    onHistoryCallback.onSessionHistoryDelete(sessionItem);
                }
            } else {//itemView
                if (onHistoryCallback != null) {
                    onHistoryCallback.onSessionHistoryClick(sessionItem);
                }
            }
        }

        public void setOnHistoryClick(OnHistoryCallback onHistoryCallback) {
            this.onHistoryCallback = onHistoryCallback;
        }
    }

    public interface OnHistoryCallback {
        void onSessionHistoryClick(SessionItem sessionItem);
        void onSessionHistoryDelete(SessionItem sessionItem);
    }
}
