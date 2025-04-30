// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.history

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.SessionItem

class HistoryListAdapter : RecyclerView.Adapter<RecyclerView.ViewHolder>() {
    private var historySessionList: MutableList<SessionItem>? = null
    private var onHistoryCallback: OnHistoryCallback? = null

    fun setOnHistoryClick(onHistoryCallback: OnHistoryCallback?) {
        this.onHistoryCallback = onHistoryCallback
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.recycle_item_history, parent, false)
        val holder = ViewHolder(view)
        holder.setOnHistoryClick(object : OnHistoryCallback {
            override fun onSessionHistoryClick(sessionItem: SessionItem) {
                if (this@HistoryListAdapter.onHistoryCallback != null) {
                    onHistoryCallback!!.onSessionHistoryClick(sessionItem)
                }
            }

            override fun onSessionHistoryDelete(sessionItem: SessionItem) {
                if (this@HistoryListAdapter.onHistoryCallback != null) {
                    onHistoryCallback!!.onSessionHistoryDelete(sessionItem)
                }
                val index = historySessionList!!.indexOf(sessionItem)
                historySessionList!!.removeAt(index)
                notifyItemRemoved(index)
            }
        })
        return holder
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val sessionItem = historySessionList!![position]
        (holder as ViewHolder).bind(sessionItem)
    }

    override fun getItemCount(): Int {
        return if (this.historySessionList == null) 0 else historySessionList!!.size
    }

    fun updateItems(historySessionList: MutableList<SessionItem>?) {
        this.historySessionList = historySessionList
        notifyDataSetChanged()
    }

    class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView), View.OnClickListener {
        var textHistory: TextView
        var viewDelete: View

        private var onHistoryCallback: OnHistoryCallback? = null

        init {
            this.itemView.setOnClickListener(this)
            this.viewDelete = itemView.findViewById(R.id.iv_delete_history)
            viewDelete.setOnClickListener(this)
            textHistory = itemView.findViewById(R.id.text_history)
        }

        fun bind(sessionItem: SessionItem) {
            textHistory.text = sessionItem.title
            itemView.tag = sessionItem
            viewDelete.tag = sessionItem
        }

        override fun onClick(v: View) {
            val sessionItem = v.tag as SessionItem
            if (v.id == R.id.iv_delete_history) {
                if (onHistoryCallback != null) {
                    onHistoryCallback!!.onSessionHistoryDelete(sessionItem)
                }
            } else { //itemView
                if (onHistoryCallback != null) {
                    onHistoryCallback!!.onSessionHistoryClick(sessionItem)
                }
            }
        }

        fun setOnHistoryClick(onHistoryCallback: OnHistoryCallback?) {
            this.onHistoryCallback = onHistoryCallback
        }
    }

    interface OnHistoryCallback {
        fun onSessionHistoryClick(sessionItem: SessionItem)
        fun onSessionHistoryDelete(sessionItem: SessionItem)
    }
}
