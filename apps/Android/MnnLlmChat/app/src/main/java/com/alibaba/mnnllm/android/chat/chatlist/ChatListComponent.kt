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
import androidx.annotation.VisibleForTesting
import com.alibaba.mnnllm.android.BuildConfig
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity.Companion.TAG
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.databinding.ActivityChatBinding
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.alibaba.mnnllm.android.widgets.ModelAvatarView
import java.text.DateFormat
import java.util.Date

class ChatListComponent(private val context: Context,
                        private val dateFormat: DateFormat,
                        private val binding: ActivityChatBinding) {
    
    companion object {
        private const val SCROLL_TO_BOTTOM_THRESHOLD_DP = 30
        private const val TAG = "ChatListComponent"

        internal fun resolveUserScrollingState(
            currentUserScrolling: Boolean,
            newState: Int,
            isAtBottom: Boolean
        ): Boolean {
            return when {
                newState == RecyclerView.SCROLL_STATE_DRAGGING -> true
                newState == RecyclerView.SCROLL_STATE_IDLE && isAtBottom -> false
                else -> currentUserScrolling
            }
        }
    }
    private lateinit var recyclerView: RecyclerView
    private lateinit var linearLayoutManager: LinearLayoutManager
    private lateinit var adapter: ChatRecyclerViewAdapter
    private lateinit var emptyView: View
    private lateinit var emptyModelAvatarView: ModelAvatarView
    private lateinit var emptyMessageTextView: TextView
    private lateinit var btnScrollToBottom: View
    private var isUserScrolling: Boolean = false
    private var forceBottomDuringStreaming: Boolean = false
    private var modelName:String? = null
    private var historyData: List<ChatDataItem>? = null
    private var onResumeAutoScrollListener: (() -> Unit)? = null

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
                if (newState == RecyclerView.SCROLL_STATE_DRAGGING) {
                    forceBottomDuringStreaming = false
                }
                isUserScrolling = resolveUserScrollingState(isUserScrolling, newState, isAtBottom())
                updateScrollToBottomButtonVisibility()
                updateDebugProbe()
            }

            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                super.onScrolled(recyclerView, dx, dy)
                updateScrollToBottomButtonVisibility()
                updateDebugProbe()
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
            resumeAutoScroll()
        }
    }

    private fun updateEmptyViewContent() {
        modelName?.let { name ->
            emptyModelAvatarView.setModelName(name)
            emptyMessageTextView.text = if (ModelTypeUtils.isSanaModel(name)) {
                context.getString(R.string.model_hello_prompt_sana)
            } else if (ModelTypeUtils.isDiffusionModel(name))  {
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
            recyclerView.post {
                recyclerView.scrollToPosition(position)
                recyclerView.post {
                    linearLayoutManager.scrollToPositionWithOffset(position, -99999)
                    recyclerView.post(object : Runnable {
                        private var attempts = 0

                        override fun run() {
                            val remainingDistance = distanceFromBottomPx()
                            if (remainingDistance <= scrollToBottomThresholdPx() || attempts >= 4) {
                                updateScrollToBottomButtonVisibility()
                                return
                            }
                            attempts += 1
                            recyclerView.scrollBy(0, remainingDistance)
                            recyclerView.post(this)
                        }
                    })
                }
            }
        }
    }

    private fun resumeAutoScroll() {
        isUserScrolling = false
        forceBottomDuringStreaming = true
        scrollToBottom()
        updateScrollToBottomButtonVisibility()
        onResumeAutoScrollListener?.invoke()
    }

    private fun maintainBottomDuringStreaming() {
        recyclerView.post {
            val remainingDistance = distanceFromBottomPx()
            if (remainingDistance > 0) {
                recyclerView.scrollBy(0, remainingDistance)
            }
            updateScrollToBottomButtonVisibility()
        }
    }

    private fun updateScrollToBottomButtonVisibility() {
        if (adapter.itemCount > 0) {
            val shouldShow = !isAtBottom()
            btnScrollToBottom.visibility = if (shouldShow) View.VISIBLE else View.GONE
        } else {
            btnScrollToBottom.visibility = View.GONE
        }
        updateDebugProbe()
    }

    private fun isAtBottom(): Boolean {
        return distanceFromBottomPx() < scrollToBottomThresholdPx()
    }

    private fun scrollToBottomThresholdPx(): Int {
        return (SCROLL_TO_BOTTOM_THRESHOLD_DP * context.resources.displayMetrics.density).toInt()
    }

    private fun distanceFromBottomPx(): Int {
        val totalItemCount = adapter.itemCount
        if (totalItemCount == 0) return 0

        if (!recyclerView.canScrollVertically(1)) {
            return 0
        }

        val range = recyclerView.computeVerticalScrollRange()
        val extent = recyclerView.computeVerticalScrollExtent()
        val offset = recyclerView.computeVerticalScrollOffset()
        return (range - extent - offset).coerceAtLeast(0)
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
            if (forceBottomDuringStreaming) {
                maintainBottomDuringStreaming()
            } else {
                scrollToEnd()
            }
        }
        updateEmptyViewVisibility()
        updateScrollToBottomButtonVisibility()
    }

    fun onStartSendMessage(userData: ChatDataItem) {
        isUserScrolling = false
        forceBottomDuringStreaming = false
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

    @VisibleForTesting
    internal fun setUserScrollingForTest(value: Boolean) {
        isUserScrolling = value
    }

    @VisibleForTesting
    internal fun detachFromBottomForTest() {
        isUserScrolling = true
        recyclerView.post {
            val detachStep = recyclerView.height.coerceAtLeast(1)
            var leftBottom = false
            for (attempt in 0 until 4) {
                recyclerView.scrollBy(0, -detachStep)
                if (!isAtBottom()) {
                    leftBottom = true
                    break
                }
            }
            if (!leftBottom) {
                recyclerView.scrollBy(0, -detachStep)
            }
            updateScrollToBottomButtonVisibility()
        }
    }

    @VisibleForTesting
    internal fun isUserScrollingForTest(): Boolean = isUserScrolling

    fun getCurrentChatHistory(): List<ChatDataItem> {
        return adapter.getCurrentChatHistory()
    }

    fun setOnResumeAutoScrollListener(listener: (() -> Unit)?) {
        onResumeAutoScrollListener = listener
    }

    private fun updateDebugProbe() {
        if (!BuildConfig.DEBUG) {
            return
        }
        recyclerView.contentDescription = buildString {
            append("atBottom=").append(isAtBottom())
            append(";bottomGapPx=").append(distanceFromBottomPx())
            append(";userScrolling=").append(isUserScrolling)
            append(";assistantChars=").append(recentItem?.displayText?.length ?: 0)
        }
    }
}
