// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat.chatlist

import android.util.Log
import android.view.WindowInsets
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.chat.ChatActivity.Companion.TAG
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.databinding.ActivityChatBinding
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import java.util.Date
import kotlin.math.abs

class ChatListComponent(private val chatActivity: ChatActivity,
                        private val binding: ActivityChatBinding) {
    private lateinit var recyclerView: RecyclerView
    private lateinit var linearLayoutManager: LinearLayoutManager
    private lateinit var adapter: ChatRecyclerViewAdapter
    private var isUserScrolling: Boolean = false

    init {
        setupRecyclerView()
        smoothScrollToBottom()
    }

    val recentItem: ChatDataItem?
        get() = adapter.recentItem

    private fun setupRecyclerView() {
        recyclerView = binding.recyclerView
        recyclerView.setItemAnimator(null)
        linearLayoutManager = LinearLayoutManager(chatActivity)
        recyclerView.setLayoutManager(linearLayoutManager)
        binding.layoutBottomContainer.addOnLayoutChangeListener { v, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom ->
            val insets: WindowInsets? = v.rootWindowInsets
            val bottomInset = insets!!.systemWindowInsetBottom
            recyclerView.setPadding(recyclerView.paddingLeft, recyclerView.paddingTop, recyclerView.paddingRight,
                bottomInset +  binding.layoutBottomContainer.height)
            insets.consumeSystemWindowInsets()
        }
        adapter = ChatRecyclerViewAdapter(chatActivity, initData(), chatActivity.modelName)
        recyclerView.setAdapter(adapter)
        recyclerView.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrollStateChanged(recyclerView: RecyclerView, newState: Int) {
                super.onScrollStateChanged(recyclerView, newState)
            }

            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                super.onScrolled(recyclerView, dx, dy)
                if (abs(dy.toDouble()) > 0) {
                    isUserScrolling = true
                }
            }

        })
    }

    private fun initData(): MutableList<ChatDataItem> {
        val data: MutableList<ChatDataItem> = ArrayList()
        data.add(ChatDataItem(chatActivity.dateFormat!!.format(Date()), ChatViewHolders.HEADER, ""))
        data.add(
            ChatDataItem(
                chatActivity.dateFormat!!.format(Date()), ChatViewHolders.ASSISTANT,
                chatActivity.getString(
                    if (ModelUtils.isDiffusionModel(chatActivity.modelName))
                        R.string.model_hello_prompt_diffusion else
                        R.string.model_hello_prompt,
                    chatActivity.modelName
                )
            )
        )
        if (chatActivity.chatSession is LlmSession
            && (chatActivity.chatSession as LlmSession).savedHistory?.isNotEmpty() == true) {
            val savedHistory = (chatActivity.chatSession as LlmSession).savedHistory!!
            data.addAll(savedHistory)
        }
        return data
    }

    private fun smoothScrollToBottom() {
        Log.d(TAG, "smoothScrollToBottom")
        recyclerView.post {
            val position = adapter.itemCount - 1
            recyclerView.scrollToPosition(position)
            recyclerView.post { recyclerView.scrollToPosition(position) }
        }
    }

    private fun scrollToEnd() {
        recyclerView.postDelayed({
            val position = adapter.itemCount - 1
            linearLayoutManager.scrollToPositionWithOffset(position, -9999)
        }, 100)
    }

    private fun addResponsePlaceholder() {
        val holderItem = ChatDataItem(chatActivity.dateFormat!!.format(Date()), ChatViewHolders.ASSISTANT, "")
        holderItem.hasOmniAudio = chatActivity.chatSession.supportOmni
        adapter.addItem(holderItem)
        smoothScrollToBottom()
    }

    fun updateAssistantResponse(chatDataItem: ChatDataItem) {
        adapter.updateRecentItem(chatDataItem)
        if (!isUserScrolling) {
            scrollToEnd()
        }
    }

    fun onStartSendMessage(userData: ChatDataItem) {
        isUserScrolling = false
        adapter.addItem(userData)
        addResponsePlaceholder()
        smoothScrollToBottom()
    }

    fun toggleShowPerformanceMetrics(show:Boolean) {
        PreferenceUtils.setBoolean(
            chatActivity,
            PreferenceUtils.KEY_SHOW_PERFORMACE_METRICS,
            show
        )
        adapter.notifyItemRangeChanged(0, adapter.itemCount)
    }

    fun reset(): Boolean {
        return this.adapter.reset()
    }
}