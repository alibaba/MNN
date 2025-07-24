// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.history

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.main.MainActivity
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatRouter
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.chat.model.ChatDataManager.Companion.getInstance
import com.alibaba.mnnllm.android.chat.model.SessionItem
import com.alibaba.mnnllm.android.history.HistoryListAdapter.OnHistoryCallback

class ChatHistoryFragment : Fragment() {
    private lateinit var chatListRecyclerView: RecyclerView
    private var textNoHistory: TextView? = null

    private var chatListAdapter: HistoryListAdapter? = null
    private var chatDataManager: ChatDataManager? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_historylist, container, false)
        chatListRecyclerView = view.findViewById(R.id.chat_history_recycler_view)
        textNoHistory = view.findViewById(R.id.text_no_history)
        chatListRecyclerView.setLayoutManager(
            LinearLayoutManager(
                context,
                LinearLayoutManager.VERTICAL,
                false
            )
        )
        chatListAdapter = HistoryListAdapter()
        chatDataManager = getInstance(requireContext())
        chatListAdapter!!.setOnHistoryClick(object : OnHistoryCallback {
            override fun onSessionHistoryClick(sessionItem: SessionItem) {
                ChatRouter.startRun(requireContext(), sessionItem.modelId, null, sessionItem.sessionId)
            }

            override fun onSessionHistoryDelete(sessionItem: SessionItem) {
                HistoryUtils.deleteHistory(context!!, chatDataManager!!, sessionItem.sessionId)
                Toast.makeText(context, R.string.history_delete_success, Toast.LENGTH_SHORT).show()
            }
        })
        chatListRecyclerView.setAdapter(chatListAdapter)
        return view
    }

    override fun onResume() {
        super.onResume()
        onLoad()
    }

    fun onLoad() {
        val historySessionList = chatDataManager!!.allSessions
        chatListAdapter!!.updateItems(historySessionList)
        if (historySessionList.isEmpty()) {
            textNoHistory!!.visibility = View.VISIBLE
        } else {
            textNoHistory!!.visibility = View.GONE
        }
    }

    companion object {
        const val TAG: String = "ChatHistoryFragment"
    }
}
