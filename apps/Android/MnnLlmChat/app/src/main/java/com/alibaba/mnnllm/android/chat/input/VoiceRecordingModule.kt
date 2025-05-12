// Created by ruoyi.sjd on 2025/01/08.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.input

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.Permissions.REQUEST_RECORD_AUDIO_PERMISSION
import com.alibaba.mnnllm.android.utils.UiUtils.getThemeColor
import com.github.squti.androidwaverecorder.WaveRecorder
import java.io.File
import com.google.android.material.card.MaterialCardView

class VoiceRecordingModule(private val activity: ChatActivity) {
    private lateinit var buttonVoiceRecording: View
    private lateinit var buttonSwitchVoice: ImageView
    private lateinit var inputCardContainer: MaterialCardView
    private var listener: VoiceRecordingListener? = null
    private var waveRecorder: WaveRecorder? = null
    private var voceRecordingWave: View? = null

    private var textVoiceHint: TextView? = null

    private var recordingFilePath: String? = null

    private var mStartRecordTime: Long = 0

    private var isCancelRecord = false
    private var enabled = false

    fun setOnVoiceRecordingListener(listener: VoiceRecordingListener?) {
        this.listener = listener
    }

    fun setup(isAudioModel: Boolean) {
        buttonSwitchVoice = activity.findViewById(R.id.bt_switch_audio)
        if (!isAudioModel) {
            buttonSwitchVoice.setVisibility(View.GONE)
        }
        inputCardContainer = activity.findViewById(R.id.input_card_container)
        voceRecordingWave = activity.findViewById(R.id.voice_recording_wav)
        buttonVoiceRecording = activity.findViewById(R.id.btn_voice_recording)
        textVoiceHint = activity.findViewById(R.id.text_voice_hint)
        buttonSwitchVoice.setOnClickListener { v: View? -> handleSwitch() }
        buttonVoiceRecording.setOnTouchListener { v: View, event: MotionEvent ->
            this.handleTouchEvent(
                v,
                event
            )
        }
        if (isAudioModel) {
            handleSwitch()
        }
    }

    private fun handleTouchEvent(v: View, event: MotionEvent): Boolean {
        if (!this.enabled) {
            return false
        }
        if (event.action == MotionEvent.ACTION_UP) {
            Log.d(TAG, "onTouch up stop recording")
            if (waveRecorder == null) {
                return false
            }
            endAudioRecording(isCancelRecord)
        } else if (event.action == MotionEvent.ACTION_DOWN) {
            if (ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    activity,
                    arrayOf(Manifest.permission.RECORD_AUDIO),
                    REQUEST_RECORD_AUDIO_PERMISSION
                )
            } else {
                startAudioRecording()
            }
        } else if (event.action == MotionEvent.ACTION_MOVE) {
            if (!isPointInsideView(event.rawX, event.rawY, v)) {
                isCancelRecord = true
                updateRecordingUI()
            } else {
                isCancelRecord = false
                updateRecordingUI()
            }
        } else if (event.action == MotionEvent.ACTION_CANCEL) {
            if (waveRecorder != null) {
                endAudioRecording(true)
            }
        }
        return true
    }

    private fun updateRecordingUI() {
        textVoiceHint!!.visibility = View.VISIBLE
        textVoiceHint!!.setText(if (isCancelRecord) R.string.release_to_cancel else R.string.release_to_send)
        voceRecordingWave!!.setBackgroundColor(
            if (isCancelRecord) Color.RED else voceRecordingWave!!.context.getThemeColor(androidx.appcompat.R.attr.colorPrimary)
        )
        inputCardContainer.setCardBackgroundColor(
            if (isCancelRecord) Color.RED else inputCardContainer.context.getThemeColor(androidx.appcompat.R.attr.colorPrimary)
        )
        textVoiceHint!!.setTextColor(if (isCancelRecord) Color.RED else voceRecordingWave!!.context.getThemeColor(
            com.google.android.material.R.attr.colorOnSurface))
    }

    private fun isPointInsideView(x: Float, y: Float, view: View): Boolean {
        val location = IntArray(2)
        view.getLocationOnScreen(location)
        val rect = Rect(
            location[0],
            location[1],
            location[0] + view.width,
            location[1] + view.height
        )
        return rect.contains(x.toInt(), y.toInt())
    }

    private fun endAudioRecording(cancel: Boolean) {
        inputCardContainer.setCardBackgroundColor(Color.TRANSPARENT)
        waveRecorder!!.stopRecording()
        voceRecordingWave!!.visibility = View.GONE
        if (listener != null) {
            val duration = (System.currentTimeMillis() - mStartRecordTime) / 1000f
            if (!cancel && duration > 1) {
                listener!!.onRecordSuccess(duration, this.recordingFilePath)
            } else {
                this.recordingFilePath?.let { File(it).delete() }
                listener!!.onRecordCanceled()
            }
        }
        textVoiceHint!!.visibility = View.GONE
        waveRecorder = null
        isCancelRecord = false
    }

    private fun startAudioRecording() {
        recordingFilePath = FileUtils.generateDestRecordFilePath(
            activity, activity.sessionId!!
        )
        Log.d(
            TAG,
            "onTouch down start recording: $recordingFilePath"
        )
        waveRecorder = WaveRecorder(recordingFilePath!!)
        mStartRecordTime = System.currentTimeMillis()
        waveRecorder!!.startRecording()
        voceRecordingWave!!.visibility = View.VISIBLE
        updateRecordingUI()
    }

    fun isRecordingMode():Boolean {
        if (buttonVoiceRecording.visibility == View.VISIBLE) {
            return true
        }
        return false
    }

    private fun handleSwitch() {
        if (buttonVoiceRecording.visibility == View.VISIBLE) {
            exitRecordingMode()
        } else {
            enterRecordingMode()
        }
    }

    fun enterRecordingMode() {
        if (buttonVoiceRecording.visibility == View.VISIBLE) {
            return
        }
        buttonVoiceRecording.visibility = View.VISIBLE
        if (listener != null) {
            listener!!.onEnterRecordingMode()
        }
        buttonSwitchVoice.setImageResource(R.drawable.ic_keyboard)
    }

    fun exitRecordingMode() {
        if (buttonVoiceRecording.visibility != View.VISIBLE) {
            return
        }
        buttonVoiceRecording.visibility = View.GONE
        if (listener != null) {
            listener!!.onLeaveRecordingMode()
        }
        buttonSwitchVoice.setImageResource(R.drawable.ic_audio)
    }

    fun handlePermissionAllowed() {
    }

    fun handlePermissionDenied() {
        Toast.makeText(this.activity, R.string.recording_permission_denied, Toast.LENGTH_LONG)
            .show()
    }

    fun onEnabled() {
        this.enabled = true
    }

    interface VoiceRecordingListener {
        fun onEnterRecordingMode()
        fun onLeaveRecordingMode()

        fun onRecordSuccess(duration: Float, recordingFilePath: String?)

        fun onRecordCanceled()
    }

    companion object {
        const val TAG: String = "VoiceRecordingModule"
    }
}
