// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.chatlist

import android.widget.SeekBar
import android.widget.SeekBar.OnSeekBarChangeListener
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders.UserViewHolder
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.utils.AudioPlayService
import com.alibaba.mnnllm.android.utils.AudioPlayService.AudioPlayerCallback

class AudioPlayerComponent(private val chatDataItem: ChatDataItem) {
    private var isPlaying = false
    private val audioPlayService: AudioPlayService = AudioPlayService.instance!!
    private var playProgress = 0f

    val TAG: String = "AudioPlayerComponent"
    private var viewHolder: UserViewHolder? = null

    fun onPlayPauseClicked() {
        if (isPlaying) {
            audioPlayService.pauseAudio(chatDataItem.audioPath)
            setPlaying(false)
        } else {
            audioPlayService.playAudio(chatDataItem.audioPath, object : AudioPlayerCallback {
                override fun onPlayFinish() {
                    setPlaying(false)
                    setProgress(0.0f)
                }

                override fun onPlayError() {
                    setPlaying(false)
                    setProgress(0.0f)
                }

                override fun onPlayStart() {
                    setPlaying(true)
                    if (viewHolder != null) {
                        viewHolder!!.iconPlayPause.setImageResource(R.drawable.ic_audio_pause)
                    }
                }

                override fun onPlayProgress(progress: Float) {
                    setProgress(progress)
                }
            })
        }
    }

    private fun setProgress(progress: Float) {
        this.playProgress = progress
        if (viewHolder != null) {
            viewHolder!!.audioSeekBar.progress = (progress * 100).toInt()
        }
    }

    private fun updateProgressView() {
        if (viewHolder != null) {
            viewHolder!!.audioSeekBar.progress = (playProgress * 100).toInt()
        }
    }

    private fun updatePlayingState() {
        viewHolder!!.iconPlayPause.setImageResource(if (isPlaying) R.drawable.ic_audio_pause else R.drawable.ic_audio_play)
    }

    private fun setPlaying(playing: Boolean) {
        if (playing != isPlaying) {
            isPlaying = playing
            updatePlayingState()
        }
    }

    private fun setSeekbarListener() {
        viewHolder!!.audioSeekBar.setOnSeekBarChangeListener(object : OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    audioPlayService.seekAudio(chatDataItem.audioPath!!, progress / 100f)
                }
            }

            override fun onStartTrackingTouch(seekBar: SeekBar) {
            }

            override fun onStopTrackingTouch(seekBar: SeekBar) {
            }
        })
    }


    fun bindViewHolder(viewHolder: UserViewHolder?) {
        this.viewHolder = viewHolder
        setSeekbarListener()
        updateProgressView()
        updatePlayingState()
    }
}
