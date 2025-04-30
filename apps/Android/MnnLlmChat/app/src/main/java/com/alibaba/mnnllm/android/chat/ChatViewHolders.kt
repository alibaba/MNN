// Created by ruoyi.sjd on 2025/01/03.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.annotation.SuppressLint
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.text.TextUtils
import android.view.MotionEvent
import android.view.View
import android.view.View.OnLongClickListener
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.ClipboardUtils
import com.alibaba.mnnllm.android.utils.DeviceUtils
import com.alibaba.mnnllm.android.utils.GithubUtils
import com.alibaba.mnnllm.android.utils.ModelUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.alibaba.mnnllm.android.utils.UiUtils
import com.alibaba.mnnllm.android.widgets.FullScreenImageViewer
import com.alibaba.mnnllm.android.widgets.PopupWindowHelper
import io.noties.markwon.Markwon

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

    class UserViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView), View.OnClickListener {
        val audioLayout: View =
            itemView.findViewById(R.id.layout_audio)
        val viewText: TextView = itemView.findViewById(R.id.tv_chat_text)

        val chatImage: ImageView =
            itemView.findViewById(R.id.tv_chat_image)

        val textDuration: TextView = itemView.findViewById(R.id.tv_chat_voice_duration)

        val iconPlayPause: ImageView =
            itemView.findViewById(R.id.iv_audio_play_pause)
        val audioSeekBar: SeekBar = itemView.findViewById(R.id.audio_seek_bar)

        init {
            iconPlayPause.setOnClickListener(this)
        }

        @SuppressLint("DefaultLocale")
        fun bind(data: ChatDataItem) {
            audioLayout.visibility =
                if (data.audioUri != null) View.VISIBLE else View.GONE
            audioLayout.tag = data
            iconPlayPause.tag = data
            viewText.text = data.text
            viewText.visibility =
                if (TextUtils.isEmpty(data.text)) View.GONE else View.VISIBLE
            textDuration.text = formatTime(data.audioDuration.toInt())
            val imageUri = data.imageUri
            chatImage.visibility =
                if (imageUri != null) View.VISIBLE else View.GONE
            if (imageUri != null) {
                chatImage.setImageURI(imageUri)
            }
            if (data.audioPlayComponent != null) {
                data.audioPlayComponent!!.bindViewHolder(this)
            }
        }

        override fun onClick(v: View) {
            val chatDataItem = v.tag as ChatDataItem
            if (chatDataItem.audioUri != null) {
                if (chatDataItem.audioPlayComponent == null) {
                    chatDataItem.audioPlayComponent = AudioPlayerComponent(chatDataItem)
                }
                chatDataItem.audioPlayComponent!!.bindViewHolder(this)
                chatDataItem.audioPlayComponent!!.onPlayPauseClicked()
            }
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
        private val benchmarkInfo: TextView = view.findViewById(R.id.tv_chat_benchmark)

        private val headerIcon: ImageView =
            view.findViewById(R.id.ic_header)

        private val imageGenerated: ImageView =
            view.findViewById(R.id.image_generated)
        private val markdown = Markwon.create(itemView.context)
        var viewAssistantLoading: View =
            view.findViewById(R.id.view_assistant_loading)

        private var lastTouchX = 0
        private var lastTouchY = 0

        init {
            viewText.setOnLongClickListener(this)
            viewText.setOnTouchListener { v, event ->
                if (event.action == MotionEvent.ACTION_DOWN) {
                    val location = IntArray(2)
                    v.getLocationOnScreen(location)
                    lastTouchX = location[0] + event.x.toInt()
                    lastTouchY = location[1] + event.y.toInt()
                }
                false
            }
            imageGenerated.setOnClickListener(this)
        }

        fun bind(data: ChatDataItem, modelName: String?, payloads: List<Any?>?) {
            if (!payloads.isNullOrEmpty()) {
                markdown.setMarkdown(viewText, data.displayText!!)
                return
            }
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
                viewAssistantLoading.visibility = if (!TextUtils.isEmpty(data.displayText)) {
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
                    copyText(itemView.context, textView)
                } else if (v.id == R.id.assistant_text_select) {
                    val intent = Intent(v.context, SelectTextActivity::class.java)
                    intent.putExtra("content", chatDataItem.text)
                    v.context.startActivity(intent)
                } else if (v.id == R.id.assistant_text_report) {
                    val chatActivity = UiUtils.getActivity(v.context) as ChatActivity
                    ClipboardUtils.copyToClipboard(
                        chatActivity,
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

        private fun copyText(context: Context, textView: TextView) {
            val content = textView.text.toString()
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("CopiedText", content)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(context, R.string.copy_success, Toast.LENGTH_SHORT).show()
        }

        companion object {
            const val TAG: String = "AssistantViewHolder"
        }
    }
}
