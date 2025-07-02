// Created by ruoyi.sjd on 2025/3/26.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar

import android.content.Intent
import android.text.method.ScrollingMovementMethod
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.FileUtils.calculateSizeString
import com.alibaba.mnnllm.android.utils.GithubUtils
import com.taobao.meta.avatar.settings.MainSettings
import com.taobao.meta.avatar.settings.MainSettingsActivity
import com.taobao.meta.avatar.utils.UserPreferences
import java.io.File

class MainView(private val mainActivity: MainActivity, val callback:MainViewCallback) {
    val textResponse: TextView = mainActivity.findViewById(R.id.chat_text)
    private var buttonToggleShowTextImage: ImageView
    private var buttonToggleShowText:View
    private var buttonStartChat:Button
    private var buttonEndCall:View
    var textStatus:TextView
    var viewMask:View
    var viewRotateHint:View
    private var textStatusLayout:View
    private var rotationHintShowed = false
    private val buttonSettings:View
    private val buttonStarGithub:View
    private val enableSettingsAndGithub = true
    private var buttonDownload:Button
    private val textDownloadProgress:TextView
    private val textDebugInfo:TextView

    init {
        textResponse.movementMethod = ScrollingMovementMethod()
        buttonToggleShowText = mainActivity.findViewById(R.id.button_toggle_text)
        buttonToggleShowTextImage = buttonToggleShowText.findViewById(R.id.button_toggle_text_image)
        buttonToggleShowText.setOnClickListener{
            toggleShowLlmText()
        }
        updateShowLlmStatus(UserPreferences.isShowLlmText(mainActivity))
        viewRotateHint = mainActivity.findViewById(R.id.rotate_hint)
        textStatus = mainActivity.findViewById(R.id.text_status)
        textStatusLayout = mainActivity.findViewById(R.id.text_status_layout)

        viewMask = mainActivity.findViewById(R.id.not_available_mask)
        buttonEndCall = mainActivity.findViewById(R.id.button_end_call)
        buttonStartChat = mainActivity.findViewById(R.id.button_start_chat)
        buttonDownload = mainActivity.findViewById(R.id.button_download)
        textDownloadProgress = mainActivity.findViewById(R.id.text_download_progress)
        buttonDownload.setOnClickListener({
            buttonDownload.text = mainActivity.getString(R.string.download_prepareing)
            callback.onDownloadClicked()
        })
        buttonEndCall.setOnClickListener{
            callback.onEndCall()
        }
        textStatusLayout.setOnClickListener{
            callback.onStopAnswerClicked()
        }
        setupStartButton()
        buttonStarGithub = mainActivity.findViewById<View>(R.id.button_home_page).apply {
            setOnClickListener{
                GithubUtils.starProject(mainActivity)
            }
        }
        buttonSettings = mainActivity.findViewById<View>(R.id.button_settings).apply {
            setOnClickListener{
                mainActivity.startActivity(Intent(mainActivity, MainSettingsActivity::class.java))
            }
        }
        if (!enableSettingsAndGithub) {
            buttonSettings.visibility = View.GONE
            buttonStarGithub.visibility = View.GONE
        }
        textDebugInfo = mainActivity.findViewById(R.id.text_debug_info)
        if (MHConfig.DEBUG_SCREEN_SHOT) {
            buttonStartChat.visibility = View.INVISIBLE
            mainActivity.hideSystemBarsCompat()
        }
    }

    interface MainViewCallback{
        fun onEndCall()
        fun onStopAnswerClicked()
        fun onStartButtonClicked()
        fun onDownloadClicked()
    }


    private fun setupStartButton() {
        buttonStartChat.setOnClickListener{v->

            callback.onStartButtonClicked()
        }
    }

    fun setInitialiing() {
        buttonStartChat.visibility = View.INVISIBLE
        buttonEndCall.visibility = View.VISIBLE
        textStatus.visibility = View.VISIBLE
        buttonToggleShowText.visibility = View.VISIBLE
        buttonSettings.visibility = View.GONE
        buttonStarGithub.visibility = View.GONE
    }

    private fun toggleShowLlmText() {
        var isShowLlmText = UserPreferences.isShowLlmText(mainActivity)
        isShowLlmText = !isShowLlmText
        UserPreferences.setShowLlmText(mainActivity, isShowLlmText)
        updateShowLlmStatus(isShowLlmText)
    }

    fun updateShowLlmStatus(isShowLlmText: Boolean) {
        buttonToggleShowTextImage.setImageResource(if (isShowLlmText) R.drawable.text_on else R.drawable.text_off)
        textResponse.visibility = if (isShowLlmText) View.VISIBLE else View.GONE
    }

    fun onCallEnded() {
        buttonToggleShowText.visibility = View.GONE
        buttonStartChat.visibility = View.VISIBLE
        buttonEndCall.visibility = View.GONE
        textStatus.visibility = View.GONE
        viewMask.visibility = View.VISIBLE
        if (enableSettingsAndGithub) {
            buttonSettings.visibility = View.VISIBLE
            buttonStarGithub.visibility = View.VISIBLE
        }
    }

    fun onChatServiceStarted() {
        textStatus.text = ""
        viewMask.visibility = View.GONE
        if (!rotationHintShowed) {
            rotationHintShowed = true
            viewRotateHint.visibility = View.VISIBLE
        }


    }

    fun updateDownloadStatus(downloaded: Boolean) {
        if (downloaded) {
            buttonDownload.visibility = View.GONE
            textDownloadProgress.visibility = View.GONE
            buttonStartChat.visibility = View.VISIBLE
        } else {
            buttonDownload.visibility = View.VISIBLE
            buttonStartChat.visibility = View.INVISIBLE
            textDownloadProgress.visibility = View.VISIBLE
        }
    }

    fun updateDownloadProgress(currentBytes: Long, totalBytes: Long, speedInfo: String) {
        buttonDownload.text = mainActivity.getString(R.string.downloading)
        textDownloadProgress.text = mainActivity.getString(R.string.download_progress,
            FileUtils.formatFileSize(currentBytes), FileUtils.formatFileSize(totalBytes), speedInfo)
    }

    fun onDownloadError(error: Exception?) {
        Toast.makeText(mainActivity, error?.message, Toast.LENGTH_LONG).show()
        buttonDownload.text = mainActivity.getString(R.string.download_error)
    }

    fun updateDebugInfo() {
        if (MainSettings.isShowDebugInfo(mainActivity)) {
            if (textDebugInfo.text.isEmpty()) {
                textDebugInfo.text = mainActivity.getString(R.string.debug_model_size,
                    calculateSizeString(File(MHConfig.LLM_MODEL_DIR)),
                    calculateSizeString(File(MHConfig.TTS_MODEL_DIR)),
                    calculateSizeString(File(MHConfig.A2BS_MODEL_DIR)),
                    calculateSizeString(File(MHConfig.ASR_MODEL_DIR)),
                    calculateSizeString(File(MHConfig.NNR_MODEL_DIR))
                )
            }
        } else {
            textDebugInfo.text = ""
        }
    }
}