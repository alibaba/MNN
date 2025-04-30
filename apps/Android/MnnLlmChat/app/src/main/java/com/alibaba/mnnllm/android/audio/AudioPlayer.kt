package com.alibaba.mnnllm.android.audio

import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

class AudioPlayer {
    private var audioTrack: AudioTrack? = null
    private var audioFormat = AudioFormat.ENCODING_PCM_16BIT
//    private var _sampleRate = 44100
    private var _sampleRate = 24000
    private var channelConfig = AudioFormat.CHANNEL_OUT_MONO
    private var playbackThread: Thread? = null
    private var totalSize = 0
    private var playbackSpeed = 1.0f
    private var playJob:Job = Job()
    private val executor:ThreadPoolExecutor =
        ThreadPoolExecutor(1, 5, 2L, TimeUnit.SECONDS, LinkedBlockingQueue())
            .apply {
                this.allowCoreThreadTimeOut(true)
            }
    private var playerScope = CoroutineScope(executor.asCoroutineDispatcher() + SupervisorJob())

    fun start() {
        totalSize = 0
        if (isPlaying) {
            return
        }
        val minBufferSize = AudioTrack.getMinBufferSize(
            _sampleRate,
            channelConfig,
            audioFormat
        )
        audioTrack = AudioTrack.Builder()
            .setAudioFormat(AudioFormat.Builder()
                .setEncoding(audioFormat)
                .setSampleRate(_sampleRate)
                .setChannelMask(channelConfig).build())
            .setBufferSizeInBytes(minBufferSize).build()

        if (state != AudioTrack.STATE_UNINITIALIZED) {
            val params = audioTrack!!.playbackParams
            try {
                audioTrack!!.playbackParams = params
            } catch (e: Exception) {
                Log.e(TAG, "set Audio speed failed", e)
            }
            play()
        } else {
            throw RuntimeException("Failed to initialize AudioTrack.1")
        }
    }

    fun setPlaybackSpeed(speed: Float) {
        if (speed <= 0.0f) {
            playbackSpeed = 0.5f
            return
        }
        playbackSpeed = speed
        if (audioTrack != null && isInited) {
            val params = audioTrack!!.playbackParams
            params.speed = playbackSpeed
            try {
                audioTrack!!.playbackParams = params
            } catch (e: Exception) {
                Log.e(TAG, "set Audio speed failed", e)
            }
        }
    }

    fun play() {
        if (audioTrack != null && isInited && !isPlaying) {
            audioTrack!!.play()
        }
    }

    suspend fun waitStop() {
        if (audioTrack == null) {
            Log.d(TAG, "Audio track already destroyed")
            return
        }
        if (audioTrack!!.playbackHeadPosition == totalSize) {
            return
        }
        suspendCoroutine { continuation ->
            audioTrack!!.setNotificationMarkerPosition(totalSize)
            audioTrack!!.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onPeriodicNotification(track: AudioTrack?) {
                }
                override fun onMarkerReached(track: AudioTrack?) {
                    Log.d(TAG, "Audio track end of file reached ${track?.playbackHeadPosition}")
                    continuation.resume(true)
                }
            })
        }
    }

    fun stop() {
        if (audioTrack != null && isInited && !isStopped) {
            audioTrack!!.stop()
            audioTrack!!.release()
            audioTrack = null
            playbackThread = null
        }
    }

    fun currentSize(): Int {
        return totalSize
    }

    fun setMarkerSizeListener(size:Int, listener: (Int) -> Unit) {
        audioTrack!!.setNotificationMarkerPosition(size)
        audioTrack!!.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
            override fun onPeriodicNotification(track: AudioTrack?) {
            }
            override fun onMarkerReached(track: AudioTrack?) {
                Log.d(TAG, "onMarkerReached ${track?.playbackHeadPosition}")
                listener(track?.playbackHeadPosition ?: 0)
            }
        })
    }
    private var chunkCounter = 0
    private val MAX_CHUNKS_TO_LOG = 100
    suspend fun playChunk(pcmData: FloatArray) {
        val shortData = ShortArray(pcmData.size)
        for (i in pcmData.indices) {
            val limitedSample = pcmData[i].coerceIn(-1.0f, 1.0f)
            shortData[i] = (limitedSample * 32767.0f).toInt().toShort()
        }
        playChunk(shortData)
    }

    suspend fun playChunk(pcmData: ShortArray) {
        totalSize += pcmData.size
        Log.d(TAG, "playChunk: ${pcmData.size}")
        try {
            playerScope.async {
                audioTrack?.write(pcmData, 0, pcmData.size)
            }.await()
        } catch (Exception: Exception) {
            Log.e(TAG, "playChunk: ", Exception)
        }
    }

    private fun stopNow() {
        totalSize = 0
        if (audioTrack != null && isInited && !isStopped) {
            audioTrack!!.pause()
            audioTrack!!.flush()
            audioTrack!!.stop()
            audioTrack!!.release()
            audioTrack = null
            playbackThread = null
        }
    }

    fun reset() {
        stop()
        start()
    }

    fun destroy() {
        stopNow()
        playJob.cancel()
        executor.shutdown()
    }

    val isPlaying: Boolean
        get() = (playState == AudioTrack.PLAYSTATE_PLAYING)

    val isStopped: Boolean
        get() = (playState == AudioTrack.PLAYSTATE_STOPPED)

    val isPaused: Boolean
        get() = (playState == AudioTrack.PLAYSTATE_PAUSED)

    private val isInited: Boolean
        get() = (state == AudioTrack.STATE_INITIALIZED)

    fun currentTime(): Long {
        return if (audioTrack != null && isInited && isPlaying) {
            (audioTrack!!.playbackHeadPosition * 1000.0 / _sampleRate).toLong()
        } else {
            0
        }
    }

    fun currentHeadPosition(): Int {
        return if (audioTrack != null && isInited && isPlaying) {
            audioTrack!!.playbackHeadPosition
        } else {
            0
        }
    }

    val channelCount: Int
        get() {
            if (audioTrack != null && isInited) {
                return audioTrack!!.channelCount
            } else {
                return AudioTrack.ERROR
            }
        }

    val playState: Int
        get() = if (audioTrack != null && isInited) {
            audioTrack!!.playState
        } else {
            AudioTrack.ERROR
        }

    val sampleRate: Int
        get() {
            return if (audioTrack != null && isInited) {
                audioTrack!!.sampleRate
            } else {
                _sampleRate
            }
        }

    val state: Int
        get() {
            if (audioTrack != null) {
                return audioTrack!!.state
            } else {
                Log.e(TAG, "AudioTrack is NOT initialized.")
                return AudioTrack.STATE_UNINITIALIZED
            }
        }

    fun totalTime(): Long {
        return if (isPlaying) ((totalSize * 1000L / channelCount) / _sampleRate) else 0
    }

    companion object {
        private const val TAG = "AudioPlayer"
    }
}
