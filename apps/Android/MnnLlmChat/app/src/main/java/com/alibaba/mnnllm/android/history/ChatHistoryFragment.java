// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.history;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.chat.ChatDataManager;
import com.alibaba.mnnllm.android.chat.SessionItem;
import com.alibaba.mnnllm.android.MainActivity;

import java.util.List;

public class ChatHistoryFragment extends Fragment {

    public static final String TAG = "ChatHistoryFragment";
    private RecyclerView chatListRecyclerView;
    private TextView textNoHistory;

    private HistoryListAdapter chatListAdapter;
    private ChatDataManager chatDataManager;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_historylist, container, false);
        chatListRecyclerView = view.findViewById(R.id.chat_history_recycler_view);
        textNoHistory = view.findViewById(R.id.text_no_history);
        chatListRecyclerView.setLayoutManager(new LinearLayoutManager(getContext(), LinearLayoutManager.VERTICAL, false));
        chatListAdapter = new HistoryListAdapter();
        chatDataManager = ChatDataManager.getInstance(getContext());
        chatListAdapter.setOnHistoryClick(new HistoryListAdapter.OnHistoryCallback() {
            @Override
            public void onSessionHistoryClick(SessionItem sessionItem) {
                ((MainActivity)getActivity()).runModel(null, sessionItem.getModelId(), sessionItem.getSessionId());
            }

            @Override
            public void onSessionHistoryDelete(SessionItem sessionItem) {
                HistoryUtils.deleteHistory(getContext(), chatDataManager, sessionItem.getSessionId());
                Toast.makeText(getContext(), R.string.history_delete_success, Toast.LENGTH_SHORT).show();
            }
        });
        chatListRecyclerView.setAdapter(chatListAdapter);
        return view;
    }

    @Override
    public void onResume() {
        super.onResume();
        onLoad();
    }

    public void onLoad() {
        List<SessionItem> historySessionList = chatDataManager.getAllSessions();
        chatListAdapter.updateItems(historySessionList);
        if (historySessionList.isEmpty()) {
            textNoHistory.setVisibility(View.VISIBLE);
        } else {
            textNoHistory.setVisibility(View.GONE);
        }
    }
}
