// Created by ruoyi.sjd on 2025/01/03.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.chatlist

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders.AssistantViewHolder
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders.UserViewHolder
import com.alibaba.mnnllm.android.chat.model.ChatDataItem

class ChatRecyclerViewAdapter(
    context: Context?
) :
    RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private var items: MutableList<ChatDataItem> = ArrayList()
    override fun getItemCount(): Int {
        return items.size
    }
    var modelName: String? = null

    fun updateModelNameAndItems(modelName: String, items: MutableList<ChatDataItem>) {
        this.modelName = modelName
        this.items = items
        notifyDataSetChanged()
    }

    override fun getItemViewType(position: Int): Int {
        return items[position].type
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        val view: View
        when (viewType) {
            ChatViewHolders.HEADER -> {
                view = inflater.inflate(R.layout.item_holder_chatheader, parent, false)
                return ChatViewHolders.HeaderViewHolder(view)
            }

            ChatViewHolders.ASSISTANT -> {
                view = inflater.inflate(R.layout.item_holder_assistant, parent, false)
                return AssistantViewHolder(view)
            }

            ChatViewHolders.USER -> {
                view = inflater.inflate(R.layout.item_holder_user, parent, false)
                return UserViewHolder(view)
            }

            else -> {
                view = inflater.inflate(R.layout.item_holder_user, parent, false)
                return UserViewHolder(view)
            }
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val viewType = getItemViewType(position)
        if (viewType == ChatViewHolders.HEADER) {
            (holder as ChatViewHolders.HeaderViewHolder).bind(items[position])
        } else if (viewType == ChatViewHolders.ASSISTANT) {
            (holder as AssistantViewHolder).bind(items[position], modelName, null)
        } else if (viewType == ChatViewHolders.USER) {
            (holder as UserViewHolder).bind(items[position])
        }
    }

    override fun onBindViewHolder(
        holder: RecyclerView.ViewHolder,
        position: Int,
        payloads: List<Any>
    ) {
        super.onBindViewHolder(holder, position, payloads)
        val viewType = getItemViewType(position)
        if (viewType == ChatViewHolders.HEADER) {
            (holder as ChatViewHolders.HeaderViewHolder).bind(items[position])
        } else if (viewType == ChatViewHolders.ASSISTANT) {
            (holder as AssistantViewHolder).bind(items[position], modelName, payloads)
        } else if (viewType == ChatViewHolders.USER) {
            (holder as UserViewHolder).bind(items[position])
        }
    }

    fun addItem(item: ChatDataItem) {
        items.add(item)
        notifyItemInserted(items.size - 1)
    }

    val recentItem: ChatDataItem?
        get() = if (!items.isEmpty()) items[items.size - 1] else null

    fun updateRecentItem(item: ChatDataItem?) {
        notifyItemChanged(items.size - 1, Any())
    }

    fun reset(): Boolean {
        if (items.size > 0) {
            val size = items.size
            items.clear()
            notifyItemRangeRemoved(0, size)
            return true
        }
        return false
    }

    fun getCurrentChatHistory(): List<ChatDataItem> {
        return ArrayList(items)
    }
}
