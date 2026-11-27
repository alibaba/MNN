package com.alibaba.mnnllm.android.chat.chatlist

import android.app.Application
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.widget.FrameLayout
import androidx.appcompat.app.AppCompatActivity
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28], application = Application::class)
class ReadOnlyHistoryHolderTest {
    @Test
    fun assistantReadOnlyActionsDoNotRequireAChatActivity() {
        val controller = Robolectric.buildActivity(AppCompatActivity::class.java)
        controller.get().setTheme(R.style.AppTheme)
        val activity = controller.setup().get()
        try {
            val view = LayoutInflater.from(activity).inflate(R.layout.item_holder_assistant, FrameLayout(activity), false)
            activity.setContentView(view)
            val holder = ChatViewHolders.AssistantViewHolder(view, readOnly = true)
            val data = ChatDataItem("123", ChatViewHolders.ASSISTANT, "answer").apply {
                hasOmniAudio = true
                audioUri = Uri.parse("file:///saved.wav")
            }
            holder.bind(data, "deleted-model", null)
            assertEquals(View.GONE, view.findViewById<View>(R.id.btn_report_issue).visibility)
            assertEquals(View.GONE, view.findViewById<View>(R.id.btn_toggle_benchmark).visibility)
            assertEquals(View.GONE, view.findViewById<View>(R.id.btn_replay_audio).visibility)
            assertTrue(holder.onLongClick(view.findViewById(R.id.tv_chat_text)))
        } finally {
            controller.pause().stop().destroy()
        }
    }

    @Test
    fun readOnlyAudioCannotConstructAudioPlaybackService() {
        val controller = Robolectric.buildActivity(AppCompatActivity::class.java)
        controller.get().setTheme(R.style.AppTheme)
        val activity = controller.setup().get()
        try {
            val view = LayoutInflater.from(activity).inflate(R.layout.item_holder_user, FrameLayout(activity), false)
            val holder = ChatViewHolders.UserViewHolder(view, readOnly = true)
            val data = ChatDataItem("123", ChatViewHolders.USER, "question").apply {
                audioUri = Uri.parse("file:///saved.wav")
            }
            holder.bind(data)
            assertEquals(View.GONE, holder.iconPlayPause.visibility)
            assertEquals(View.GONE, holder.audioSeekBar.visibility)
            holder.onClick(holder.iconPlayPause)
            assertNull(data.audioPlayComponent)
        } finally {
            controller.pause().stop().destroy()
        }
    }

    @Test
    fun defaultAssistantHolderKeepsExistingChatControls() {
        val controller = Robolectric.buildActivity(AppCompatActivity::class.java)
        controller.get().setTheme(R.style.AppTheme)
        val activity = controller.setup().get()
        try {
            val view = LayoutInflater.from(activity).inflate(R.layout.item_holder_assistant, FrameLayout(activity), false)
            ChatViewHolders.AssistantViewHolder(view).bind(
                ChatDataItem("123", ChatViewHolders.ASSISTANT, "answer"), "model", null
            )
            assertEquals(View.VISIBLE, view.findViewById<View>(R.id.btn_report_issue).visibility)
            assertEquals(View.VISIBLE, view.findViewById<View>(R.id.btn_toggle_benchmark).visibility)
        } finally {
            controller.pause().stop().destroy()
        }
    }
}
