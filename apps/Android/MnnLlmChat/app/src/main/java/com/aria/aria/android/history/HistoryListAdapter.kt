// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.history

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.model.SessionItem
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.text.SimpleDateFormat
import java.util.*

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
        var textTimestamp: TextView
        var textModelName: TextView
        var modelAvatarView: ImageView
        var viewDelete: View

        private var onHistoryCallback: OnHistoryCallback? = null

        init {
            this.itemView.setOnClickListener(this)
            this.viewDelete = itemView.findViewById(R.id.iv_delete_history)
            viewDelete.setOnClickListener(this)
            textHistory = itemView.findViewById(R.id.text_history)
            textTimestamp = itemView.findViewById(R.id.text_timestamp)
            textModelName = itemView.findViewById(R.id.text_model_name)
            modelAvatarView = itemView.findViewById(R.id.model_avatar_view)
        }

        fun bind(sessionItem: SessionItem) {
            textHistory.text = sessionItem.title
            textTimestamp.text = formatTimestamp(sessionItem.lastChatTime)
            setModelAvatar(sessionItem.modelId)
            textModelName.text = getModelDisplayName(sessionItem.modelId)
            
            itemView.tag = sessionItem
            viewDelete.tag = sessionItem
        }

        private fun setModelAvatar(modelId: String) {
            val drawableId = ModelUtils.getDrawableId(modelId)
            if (drawableId != 0) {
                modelAvatarView.visibility = View.VISIBLE
                modelAvatarView.setImageResource(drawableId)
            } else {
                modelAvatarView.visibility = View.GONE
            }
        }

        private fun getModelDisplayName(modelId: String): String {
            return ModelUtils.getVendor(modelId)
        }

        private fun formatTimestamp(timestamp: Long): String {
            if (timestamp == 0L) {
                return ""
            }

            val now = System.currentTimeMillis()
            val chatDate = Date(timestamp)
            val today = Date(now)
            
            // Determine whether it's the same day
            val isSameDay = isSameDay(chatDate, today)
            
            val formattedTime = if (isSameDay) {
                // Chat occurred today, display hours and minutes, e.g., 8:30
                val timeFormat = SimpleDateFormat("H:mm", Locale.getDefault())
                timeFormat.format(chatDate)
            } else {
                // Chat did not occur today, display date, supports both Chinese and English
                val locale = Locale.getDefault()
                val dateFormat = if (locale.language == "zh") {
                    SimpleDateFormat("M月d日", locale)
                } else {
                    SimpleDateFormat("MMM d", locale) // For example: Jun 20, Dec 15
                }
                dateFormat.format(chatDate)
            }
            
            return formattedTime
        }
        
        private fun isSameDay(date1: Date, date2: Date): Boolean {
            val cal1 = java.util.Calendar.getInstance()
            val cal2 = java.util.Calendar.getInstance()
            cal1.time = date1
            cal2.time = date2
            return cal1.get(java.util.Calendar.YEAR) == cal2.get(java.util.Calendar.YEAR) &&
                    cal1.get(java.util.Calendar.DAY_OF_YEAR) == cal2.get(java.util.Calendar.DAY_OF_YEAR)
        }

        override fun onClick(v: View) {
            val sessionItem = v.tag as SessionItem
            if (v.id == R.id.iv_delete_history) {
                showDeleteConfirmDialog(sessionItem)
            } else {
                if (onHistoryCallback != null) {
                    onHistoryCallback!!.onSessionHistoryClick(sessionItem)
                }
            }
        }

        private fun showDeleteConfirmDialog(sessionItem: SessionItem) {
            val context = itemView.context
            MaterialAlertDialogBuilder(context)
                .setTitle(R.string.delete_history_title)
                .setMessage(R.string.delete_history_message)
                .setPositiveButton(android.R.string.ok) { _, _ ->
                    if (onHistoryCallback != null) {
                        onHistoryCallback!!.onSessionHistoryDelete(sessionItem)
                    }
                    val index = (itemView.parent as RecyclerView).getChildAdapterPosition(itemView)
                    if (index != RecyclerView.NO_POSITION) {
                        (itemView.parent as RecyclerView).adapter?.notifyItemRemoved(index)
                    }
                }
                .setNegativeButton(android.R.string.cancel, null)
                .show()
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
