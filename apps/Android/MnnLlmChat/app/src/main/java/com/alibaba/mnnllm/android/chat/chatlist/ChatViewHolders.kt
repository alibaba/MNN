// Created by ruoyi.sjd on 2025/01/03.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.chatlist

import android.annotation.SuppressLint
import android.content.Intent
import android.text.TextUtils
import android.util.Log
import android.view.MenuItem
import android.view.MotionEvent
import android.view.View
import android.view.View.OnLongClickListener
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.widget.PopupMenu
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.chat.PromptUtils
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.SelectTextActivity
import com.alibaba.mnnllm.android.chat.chatlist.VideoPlayerComponent
import com.alibaba.mnnllm.android.utils.ClipboardUtils
import com.alibaba.mnnllm.android.utils.DeviceUtils
import com.alibaba.mnnllm.android.utils.GithubUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.alibaba.mnnllm.android.utils.UiUtils
import com.alibaba.mnnllm.android.widgets.FullScreenImageViewer
import com.alibaba.mnnllm.android.widgets.PopupWindowHelper
import io.noties.markwon.Markwon
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.util.Locale

object ChatViewHolders {
    const val HEADER: Int = 0
    const val ASSISTANT: Int = 1
    const val USER: Int = 2

    class HeaderViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val viewTime: TextView = itemView.findViewById(R.id.tv_date)

        fun bind(data: ChatDataItem) {
            viewTime.text = data.time
        }
    }

    class UserViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView), View.OnClickListener,
        OnLongClickListener {
        val audioLayout: View =
            itemView.findViewById(R.id.layout_audio)
        val viewText: TextView = itemView.findViewById(R.id.tv_chat_text)

        val chatImagesRecycler: RecyclerView =
            itemView.findViewById(R.id.rv_chat_images)

        val chatVideo: com.alibaba.mnnllm.android.widgets.VideoPreviewView =
            itemView.findViewById(R.id.tv_chat_video)

        val textDuration: TextView = itemView.findViewById(R.id.tv_chat_voice_duration)

        val iconPlayPause: ImageView =
            itemView.findViewById(R.id.iv_audio_play_pause)
        val audioSeekBar: SeekBar = itemView.findViewById(R.id.audio_seek_bar)

        init {
            iconPlayPause.setOnClickListener(this)
            viewText.setOnLongClickListener(this)
            audioLayout.setOnLongClickListener(this)
            chatVideo.setOnClickListener(this)
        }

        @SuppressLint("DefaultLocale")
        fun bind(data: ChatDataItem) {
            audioLayout.visibility =
                if (data.audioUri != null) View.VISIBLE else View.GONE
            audioLayout.tag = data
            iconPlayPause.tag = data
            itemView.tag = data
            chatVideo.tag = data
            viewText.text = data.text
            viewText.visibility =
                if (TextUtils.isEmpty(data.text)) View.GONE else View.VISIBLE
            textDuration.text = formatTime(data.audioDuration.toInt())
            
            val imageUris = data.imageUris
            chatImagesRecycler.visibility =
                if (!imageUris.isNullOrEmpty()) View.VISIBLE else View.GONE
            if (!imageUris.isNullOrEmpty()) {
                Log.d("UserViewHolder", "Binding ${imageUris.size} images")
                chatImagesRecycler.layoutManager = androidx.recyclerview.widget.LinearLayoutManager(itemView.context, androidx.recyclerview.widget.LinearLayoutManager.HORIZONTAL, false)
                chatImagesRecycler.adapter = ChatImageAdapter(imageUris)
            }

            val videoUri = data.videoUri
            Log.d("UserViewHolder", "Binding video data: videoUri=$videoUri")
            chatVideo.visibility =
                if (videoUri != null) View.VISIBLE else View.GONE
            if (videoUri != null) {
                // Set video thumbnail and play icon
                Log.d("UserViewHolder", "Setting video URI and making visible")
                chatVideo.setVideoUri(videoUri)
                chatVideo.setPlayIconVisible(true)
            }
            if (data.audioPlayComponent != null) {
                data.audioPlayComponent!!.bindViewHolder(this)
            }
        }

        override fun onClick(v: View) {
            Log.d("UserViewHolder", "onClick called for view: ${v.id}")
            val chatDataItem = v.tag as? ChatDataItem
            if (chatDataItem == null) {
                Log.e("UserViewHolder", "chatDataItem is null for view: ${v.id}")
                return
            }
            
            if (v.id == R.id.tv_chat_video && chatDataItem.videoUri != null) {
                // Handle video click
                Log.d("UserViewHolder", "Video clicked, videoUri: ${chatDataItem.videoUri}")
                val videoPlayerComponent = VideoPlayerComponent(chatDataItem)
                videoPlayerComponent.playVideo(v.context)
            } else if (chatDataItem.audioUri != null) {
                if (chatDataItem.audioPlayComponent == null) {
                    chatDataItem.audioPlayComponent = AudioPlayerComponent(chatDataItem)
                }
                chatDataItem.audioPlayComponent!!.bindViewHolder(this)
                chatDataItem.audioPlayComponent!!.onPlayPauseClicked()
            }
        }

        override fun onLongClick(v: View?): Boolean {
            val isAudio = (v?.id == R.id.layout_audio)
            val isText = (v?.id == R.id.tv_chat_text)
            val popupMenu = PopupMenu(v!!.context, viewText)
            val inflater = popupMenu.menuInflater
            inflater.inflate(R.menu.chat_context_menu_user, popupMenu.menu)
            popupMenu.menu.findItem(R.id.chat_user_copy_audio_info).isVisible = isAudio
            popupMenu.menu.findItem(R.id.chat_user_copy).isVisible = isText
            popupMenu.setOnMenuItemClickListener { item: MenuItem ->
                if (item.itemId == R.id.chat_user_copy) {
                    UiUtils.copyText(itemView.context, viewText)
                } else if (item.itemId == R.id.chat_user_copy_audio_info) {
                    val chatDataItem = v.tag as ChatDataItem
                    if (chatDataItem.audioUri != null) {
                        ClipboardUtils.copyToClipboard(
                            itemView.context,
                            PromptUtils.generateUserPrompt(chatDataItem)
                        )
                        UiUtils.showToast(itemView.context, itemView.context.getString(R.string.copied_to_clipboard))
                    }
                }
                true
            }
            popupMenu.show()
            return true
        }

        companion object {
            @SuppressLint("DefaultLocale")
            private fun formatTime(seconds: Int): String {
                val minutes = seconds / 60
                val remainingSeconds = seconds % 60
                return String.format("%d:%02d", minutes, remainingSeconds)
            }
        }
    }

    class AssistantViewHolder @SuppressLint("ClickableViewAccessibility") constructor(view: View) :
        RecyclerView.ViewHolder(view), View.OnClickListener, OnLongClickListener {
        private val viewText: TextView = view.findViewById(R.id.tv_chat_text)
        private val viewThinking: TextView = view.findViewById(R.id.tv_chat_thinking)
        private val thinkingContainer: View = view.findViewById(R.id.ll_thinking_container)
        private val thinkingMarker: View = view.findViewById(R.id.view_thinking_marker)
        private val benchmarkInfo: TextView = view.findViewById(R.id.tv_chat_benchmark)
        private val thinkingToggle: LinearLayout = view.findViewById(R.id.ll_thinking_toggle)
        private val textThinkingHeader:TextView = view.findViewById(R.id.tv_thinking_header)
        private val ivThinkingHeader: ImageView = view.findViewById(R.id.iv_thinking_header)
        private val headerIcon: ImageView =
            view.findViewById(R.id.ic_header)

        private val imageGenerated: ImageView =
            view.findViewById(R.id.image_generated)

        // Action buttons
        private val actionButtonsLayout: LinearLayout = view.findViewById(R.id.ll_action_buttons)
        private val reportIssueButton: View = view.findViewById(R.id.btn_report_issue)
        private val toggleBenchmarkButton: View = view.findViewById(R.id.btn_toggle_benchmark)
        private val replayAudioButton: View = view.findViewById(R.id.btn_replay_audio)
        private val shareImageButton: View = view.findViewById(R.id.btn_share_image)

        private val markdown = Markwon.create(itemView.context)
        var viewAssistantLoading: View =
            view.findViewById(R.id.view_assistant_loading)

        private var lastTouchX = 0
        private var lastTouchY = 0

        init {
            viewText.setOnLongClickListener(this)
            viewThinking.setOnLongClickListener(this)
            viewText.setOnTouchListener { v, event ->
                if (event.action == MotionEvent.ACTION_DOWN) {
                    updatePointerDownLocation(v, event)
                }
                false
            }
            viewThinking.setOnTouchListener { v, event ->
                if (event.action == MotionEvent.ACTION_DOWN) {
                    updatePointerDownLocation(v, event)
                }
                false
            }
            imageGenerated.setOnClickListener(this)
            thinkingToggle.setOnClickListener {
                val chatDataItem = it.tag as ChatDataItem
                chatDataItem.toggleThinking()
                updateThinkingView(chatDataItem, itemView.context)
                markdown.setMarkdown(viewText, chatDataItem.displayText!!)
            }

            // Setup action buttons
            reportIssueButton.setOnClickListener {
                val chatDataItem = it.tag as ChatDataItem
                MaterialAlertDialogBuilder(itemView.context)
                    .setTitle(R.string.report_issue_confirm_title)
                    .setMessage(R.string.report_issue_confirm_message)
                    .setPositiveButton(R.string.confirm) { dialog, _ ->
                        GithubUtils.reportIssue(itemView.context)
                        dialog.dismiss()
                    }
                    .setNegativeButton(R.string.cancel) { dialog, _ ->
                        dialog.dismiss()
                    }
                    .show()
            }
            
            toggleBenchmarkButton.setOnClickListener {
                val chatActivity = itemView.context as? com.alibaba.mnnllm.android.chat.ChatActivity
                chatActivity?.let { activity ->
                    // Access the chatListComponent to toggle performance metrics
                    val currentState = benchmarkInfo.visibility == View.VISIBLE
                    activity.chatListComponent.toggleShowPerformanceMetrics(!currentState)
                }
            }
            
            replayAudioButton.setOnClickListener {
                val chatDataItem = it.tag as ChatDataItem
                replayAudio(chatDataItem)
            }
            shareImageButton.setOnClickListener {
                val chatDataItem = it.tag as ChatDataItem
                shareImage(chatDataItem)
            }
        }

        private fun shareImage(chatDataItem: ChatDataItem) {
            val imageUri = chatDataItem.imageUri ?: return
            val context = itemView.context
            
            val shareUri = if (imageUri.scheme == "file") {
                androidx.core.content.FileProvider.getUriForFile(
                    context,
                    context.packageName + ".fileprovider",
                    java.io.File(imageUri.path!!)
                )
            } else {
                imageUri
            }
            
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "image/*"
                putExtra(Intent.EXTRA_STREAM, shareUri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            context.startActivity(Intent.createChooser(shareIntent, context.getString(R.string.share_image)))
        }

        private fun  updatePointerDownLocation(v:View, event: MotionEvent) {
            val location = IntArray(2)
            v.getLocationOnScreen(location)
            lastTouchX = location[0] + event.x.toInt()
            lastTouchY = location[1] + event.y.toInt()
        }

        fun bind(data: ChatDataItem, modelName: String?, payloads: List<Any?>?) {
            if (!payloads.isNullOrEmpty()) {
                if (data.thinkingText != null && !TextUtils.isEmpty(data.thinkingText)) {
                    updateThinkingView(data, itemView.context)
                }
                if (data.displayText != null) {
                    markdown.setMarkdown(viewText, data.displayText!!)
                }
                return
            }

            updateThinkingView(data, itemView.context)
            if (TextUtils.isEmpty(data.displayText)) {
                viewText.visibility = View.GONE
            } else {
                markdown.setMarkdown(viewText, data.displayText!!)
                viewText.visibility = View.VISIBLE
            }

            if (data.hasOmniAudio) {
                viewAssistantLoading.visibility = if (data.loading) {
                    View.VISIBLE
                } else {
                    View.GONE
                }
            } else {
                viewAssistantLoading.visibility = if (!TextUtils.isEmpty(data.displayText) || !TextUtils.isEmpty(data.thinkingText)) {
                  View.GONE
                } else {
                    View.VISIBLE
                }
            }
            val showMetrics = PreferenceUtils.getBoolean(
                itemView.context,
                PreferenceUtils.KEY_SHOW_PERFORMACE_METRICS,
                true
            )
            if (showMetrics && !TextUtils.isEmpty(data.benchmarkInfo)) {
                benchmarkInfo.visibility = View.VISIBLE
                benchmarkInfo.text = data.benchmarkInfo
            } else {
                benchmarkInfo.visibility = View.GONE
            }
            imageGenerated.visibility =
                if (data.imageUri != null) View.VISIBLE else View.GONE
            if (data.imageUri != null) {
                imageGenerated.setImageURI(data.imageUri)
            }
            val drawableId = ModelUtils.getDrawableId(modelName)
            headerIcon.setImageResource(if (drawableId > 0) drawableId else R.drawable.ic_launcher)
            imageGenerated.tag = data
            viewText.tag = data
            viewThinking.tag = data
            thinkingToggle.tag = data
            
            // Setup action buttons
            setupActionButtons(data, showMetrics)
            reportIssueButton.tag = data
            toggleBenchmarkButton.tag = data
            replayAudioButton.tag = data
            shareImageButton.tag = data
        }
        
        private fun updateThinkingView(data: ChatDataItem, context: android.content.Context) {
            val showThinking = data.showThinking
            thinkingToggle.visibility = if (TextUtils.isEmpty(data.thinkingText)) {
                View.GONE
            } else {
                View.VISIBLE
            }
            textThinkingHeader.text = if (data.thinkingFinishedTime >= 0)
                textThinkingHeader.resources.getString(R.string.r1_think_complete_template, (data.thinkingFinishedTime / 1000).toString())
            else textThinkingHeader.resources.getString(R.string.r1_thinking_message)
            if (showThinking && !TextUtils.isEmpty(data.thinkingText)) {
                val thinkingText = data.thinkingText!!
                thinkingContainer.visibility = View.VISIBLE
                viewThinking.visibility = View.VISIBLE
                // Legacy compatibility: if content starts with '>' assume preformatted blockquote
                val isLegacyBlockQuote = thinkingText.trimStart().startsWith(">")
                // Hide left marker if legacy content already has its own marker style
                thinkingMarker.visibility = if (isLegacyBlockQuote) View.GONE else View.VISIBLE
                markdown.setMarkdown(viewThinking, thinkingText)
                ivThinkingHeader.setImageResource(R.drawable.ic_arrow_up)
            } else {
                ivThinkingHeader.setImageResource(R.drawable.ic_arrow_down)
                viewThinking.visibility = View.GONE
                thinkingContainer.visibility = View.GONE
                // Reset marker visible for next binds by default
                thinkingMarker.visibility = View.VISIBLE
            }
        }

        override fun onClick(v: View) {
            val data = v.tag as ChatDataItem
            FullScreenImageViewer.showImagePopup(v.context, data.imageUri)
        }

        override fun onLongClick(v: View): Boolean {
            val textView = v as TextView
            val chatDataItem = v.getTag() as ChatDataItem
            PopupWindowHelper().showPopupWindow(
                v.getContext(), v, this.lastTouchX, this.lastTouchY
            ) { v ->
                if (v.id == R.id.assistant_text_copy) {
                    UiUtils.copyText(itemView.context, textView)
                } else if (v.id == R.id.assistant_text_select) {
                    val intent = Intent(v.context, SelectTextActivity::class.java)
                    intent.putExtra("content", chatDataItem.text)
                    v.context.startActivity(intent)
                } else if (v.id == R.id.assistant_text_report) {
                    val chatActivity = UiUtils.getActivity(v.context) as ChatActivity
                    ClipboardUtils.copyToClipboard(
                        itemView.context,
                        """
                            ${DeviceUtils.deviceInfo}
                            ${chatActivity.sessionDebugInfo}
                            """.trimIndent()
                    )
                    Toast.makeText(
                        chatActivity,
                        R.string.debug_message_copied,
                        Toast.LENGTH_LONG
                    ).show()
                    GithubUtils.reportIssue(v.context)
                }
            }
            return true
        }
        
        private fun setupActionButtons(data: ChatDataItem, showMetrics: Boolean) {
            // Show action buttons for completed assistant messages
            // Show buttons if not loading AND has any content (displayText or text or thinking content)
            val hasAnyContent = !TextUtils.isEmpty(data.displayText) || 
                               !TextUtils.isEmpty(data.text) || 
                               !TextUtils.isEmpty(data.thinkingText)
            val shouldShowButtons = !data.loading && hasAnyContent
            actionButtonsLayout.visibility = if (shouldShowButtons) View.VISIBLE else View.GONE
            
            if (shouldShowButtons) {
                // Show/hide replay button based on audio availability
                replayAudioButton.visibility = if (data.hasOmniAudio && !data.audioPath.isNullOrEmpty()) 
                    View.VISIBLE 
                else 
                    View.GONE
                
                // Show/hide share button based on image availability
                shareImageButton.visibility = if (data.imageUri != null) View.VISIBLE else View.GONE
            }
        }
        
        private fun replayAudio(chatDataItem: ChatDataItem) {
            if (chatDataItem.hasOmniAudio && !chatDataItem.audioPath.isNullOrEmpty()) {
                // Play the saved audio file
                val audioPlayService = com.alibaba.mnnllm.android.utils.AudioPlayService.instance
                audioPlayService?.playAudio(chatDataItem.audioPath, object : com.alibaba.mnnllm.android.utils.AudioPlayService.AudioPlayerCallback {
                    override fun onPlayStart() {
                        android.util.Log.d(TAG, "Started replaying audio")
                    }

                    override fun onPlayFinish() {
                        android.util.Log.d(TAG, "Finished replaying audio")
                    }

                    override fun onPlayError() {
                        android.util.Log.e(TAG, "Error replaying audio")
                        android.widget.Toast.makeText(itemView.context, "Error playing audio", android.widget.Toast.LENGTH_SHORT).show()
                    }

                    override fun onPlayProgress(progress: Float) {
                        // Not needed for our use case
                    }
                })
            } else {
                android.widget.Toast.makeText(itemView.context, "No audio available", android.widget.Toast.LENGTH_SHORT).show()
            }
        }

        companion object {
            const val TAG: String = "AssistantViewHolder"
        }
    }
}