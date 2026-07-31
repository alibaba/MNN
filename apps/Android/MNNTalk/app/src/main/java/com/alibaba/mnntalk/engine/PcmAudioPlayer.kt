package com.alibaba.mnntalk.engine

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.io.Closeable
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

class PcmAudioPlayer(private val sampleRate: Int) : Closeable {
    private val lock = Any()
    private val playbackGeneration = AtomicLong(0L)
    private var audioTrack: AudioTrack? = null
    private var queuedSamples = 0L

    fun begin(): Long {
        synchronized(lock) {
            releaseLocked()
            val minimum = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            check(minimum > 0) { "Unable to determine playback buffer size" }
            audioTrack = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .build()
                )
                .setBufferSizeInBytes(minimum * 2)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
                .also {
                    check(it.state == AudioTrack.STATE_INITIALIZED) {
                        it.release()
                        "Unable to initialize audio playback"
                    }
                    it.play()
                }
            queuedSamples = 0L
            return playbackGeneration.incrementAndGet()
        }
    }

    suspend fun write(generation: Long, samples: ShortArray): Boolean = withContext(Dispatchers.IO) {
        if (samples.isEmpty() || generation != playbackGeneration.get()) {
            return@withContext false
        }
        val track = synchronized(lock) { audioTrack } ?: return@withContext false
        val written = runCatching {
            track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
        }.getOrDefault(AudioTrack.ERROR)
        if (written > 0 && generation == playbackGeneration.get()) {
            synchronized(lock) {
                queuedSamples += written
            }
            true
        } else {
            false
        }
    }

    suspend fun awaitDrain(generation: Long) {
        while (generation == playbackGeneration.get()) {
            val snapshot = synchronized(lock) {
                val track = audioTrack ?: return
                track to queuedSamples
            }
            val played = snapshot.first.playbackHeadPosition.toLong() and 0xffffffffL
            if (played >= snapshot.second) {
                return
            }
            delay(30)
        }
    }

    fun interrupt() {
        synchronized(lock) {
            playbackGeneration.incrementAndGet()
            releaseLocked()
            queuedSamples = 0L
        }
    }

    private fun releaseLocked() {
        audioTrack?.let { track ->
            runCatching { track.pause() }
            runCatching { track.flush() }
            runCatching { track.stop() }
            track.release()
        }
        audioTrack = null
    }

    override fun close() {
        interrupt()
    }
}
