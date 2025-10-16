// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat.chatlist

import android.content.Context
import android.util.Log
import android.view.View
import android.view.WindowInsets
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity.Companion.TAG
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.databinding.ActivityChatBinding
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.alibaba.mnnllm.android.widgets.ModelAvatarView
import java.text.DateFormat
import java.util.Date
import kotlin.math.abs

class ChatListComponent(private val context: Context,
                        private val dateFormat: DateFormat,
                        private val binding: ActivityChatBinding) {
    
    companion object {
        private const val SCROLL_TO_BOTTOM_THRESHOLD_DP = 30
        private const val TAG = "ChatListComponent"
    }
    private lateinit var recyclerView: RecyclerView
    private lateinit var linearLayoutManager: LinearLayoutManager
    private lateinit var adapter: ChatRecyclerViewAdapter
    private lateinit var emptyView: View
    private lateinit var emptyModelAvatarView: ModelAvatarView
    private lateinit var emptyMessageTextView: TextView
    private lateinit var btnScrollToBottom: View
    private var isUserScrolling: Boolean = false
    private var modelName:String? = null
    private var historyData: List<ChatDataItem>? = null

    init {
        setupRecyclerView()
        setupEmptyView()
        setupScrollToBottomButton()
    }

    fun setup(modelName:String, historyData: List<ChatDataItem>?) {
        Log.d(TAG, "setup: modelName=$modelName, historyData size=${historyData?.size ?: 0}")
        this.modelName = modelName
        this.historyData = historyData
        val initData = initData()
        Log.d(TAG, "setup: initData size=${initData.size}")
        adapter.updateModelNameAndItems(modelName, initData)
        updateEmptyViewContent()
        updateEmptyViewVisibility()
        if (adapter.itemCount > 0) {
            Log.d(TAG, "setup: scrolling to bottom, adapter item count=${adapter.itemCount}")
            smoothScrollToBottom()
        }
        updateScrollToBottomButtonVisibility()
    }

    /**
     * Update the model name and refresh related UI components
     */
    fun updateModel(newModelName: String) {
        this.modelName = newModelName
        // Update adapter with new model name
        adapter.modelName = newModelName
        // Update empty view content for new model
        updateEmptyViewContent()
        // Notify adapter to refresh items that might be model-dependent
        adapter.notifyDataSetChanged()
    }

    val recentItem: ChatDataItem?
        get() = adapter.recentItem

    private fun setupRecyclerView() {
        recyclerView = binding.recyclerView
        recyclerView.setItemAnimator(null)
        linearLayoutManager = LinearLayoutManager(context)
        recyclerView.setLayoutManager(linearLayoutManager)
        binding.layoutBottomContainer.addOnLayoutChangeListener { v, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom ->
            val insets: WindowInsets? = v.rootWindowInsets
            val bottomInset = insets!!.systemWindowInsetBottom
            recyclerView.setPadding(recyclerView.paddingLeft, recyclerView.paddingTop, recyclerView.paddingRight,
                bottomInset +  binding.layoutBottomContainer.height)
            insets.consumeSystemWindowInsets()
        }
        adapter = ChatRecyclerViewAdapter(context)
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
                updateScrollToBottomButtonVisibility()
            }

        })
    }

    private fun setupEmptyView() {
        emptyView = binding.emptyView.root
        emptyModelAvatarView = binding.emptyView.modelAvatarView
        emptyMessageTextView = binding.emptyView.tvEmptyMessage
    }

    private fun setupScrollToBottomButton() {
        btnScrollToBottom = binding.btnScrollToBottom
        btnScrollToBottom.setOnClickListener {
            scrollToBottom()
        }
    }

    private fun updateEmptyViewContent() {
        modelName?.let { name ->
            emptyModelAvatarView.setModelName(name)
            emptyMessageTextView.text = if (ModelTypeUtils.isDiffusionModel(name))  {
                context.getString(R.string.model_hello_prompt_diffusion)
            } else {
                context.getString(R.string.model_hello_prompt)
            }
        }
    }

    private fun updateEmptyViewVisibility() {
        val isEmpty = adapter.itemCount == 0
        emptyView.visibility = if (isEmpty) View.VISIBLE else View.GONE
        recyclerView.visibility = if (isEmpty) View.GONE else View.VISIBLE
    }

    private fun initData(): MutableList<ChatDataItem> {
        val data: MutableList<ChatDataItem> = ArrayList()
        if (historyData != null) {
            data.addAll(historyData!!)
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

    private fun scrollToBottom() {
        Log.d(TAG, "scrollToBottom")
        val position = adapter.itemCount - 1
        if (position >= 0) {
            // Use two steps to ensure scrolling to the very bottom
            recyclerView.post {
                // Step 1: First scroll to the last item
                recyclerView.scrollToPosition(position)
                // Step 2: Ensure scrolling to the very bottom
                recyclerView.post {
                    linearLayoutManager.scrollToPositionWithOffset(position, -99999)
                    // Add a short delay to ensure the scroll is complete before checking button state
                    recyclerView.postDelayed({
                        updateScrollToBottomButtonVisibility()
                    }, 50)
                }
            }
        }
    }

    private fun updateScrollToBottomButtonVisibility() {
        if (adapter.itemCount > 0) {
            val shouldShow = !isAtBottom()
            btnScrollToBottom.visibility = if (shouldShow) View.VISIBLE else View.GONE
        } else {
            btnScrollToBottom.visibility = View.GONE
        }
    }

    private fun isAtBottom(): Boolean {
        val totalItemCount = adapter.itemCount
        if (totalItemCount == 0) return true

        if (!recyclerView.canScrollVertically(1)) {
            return true
        }

        val thresholdPx = (SCROLL_TO_BOTTOM_THRESHOLD_DP * context.resources.displayMetrics.density).toInt()
        val range = recyclerView.computeVerticalScrollRange()
        val extent = recyclerView.computeVerticalScrollExtent()
        val offset = recyclerView.computeVerticalScrollOffset()
        val distanceFromBottom = range - extent - offset

        return distanceFromBottom < thresholdPx
    }

    private fun addResponsePlaceholder() {
        val holderItem = ChatDataItem(dateFormat.format(Date()), ChatViewHolders.ASSISTANT, "")
        holderItem.hasOmniAudio = ModelTypeUtils.isOmni(modelName!!)
        adapter.addItem(holderItem)
        smoothScrollToBottom()
    }

    fun updateAssistantResponse(chatDataItem: ChatDataItem) {
        adapter.updateRecentItem(chatDataItem)
        if (!isUserScrolling) {
            scrollToEnd()
        }
        updateEmptyViewVisibility()
        updateScrollToBottomButtonVisibility()
    }

    fun onStartSendMessage(userData: ChatDataItem) {
        isUserScrolling = false
        adapter.addItem(userData)
        addResponsePlaceholder()
        updateEmptyViewVisibility()
        smoothScrollToBottom()
        updateScrollToBottomButtonVisibility()
    }

    fun toggleShowPerformanceMetrics(show:Boolean) {
        PreferenceUtils.setBoolean(
            context,
            PreferenceUtils.KEY_SHOW_PERFORMACE_METRICS,
            show
        )
        adapter.notifyItemRangeChanged(0, adapter.itemCount)
    }

    fun reset(): Boolean {
        val wasReset = this.adapter.reset()
        if (wasReset) {
            updateEmptyViewVisibility()
            updateScrollToBottomButtonVisibility()
        }
        return wasReset
    }

    fun getCurrentChatHistory(): List<ChatDataItem> {
        return adapter.getCurrentChatHistory()
    }
}