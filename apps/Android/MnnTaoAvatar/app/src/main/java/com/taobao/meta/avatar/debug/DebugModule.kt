// Created by ruoyi.sjd on 2025/3/12.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.debug

import android.util.Log
import android.view.View
import androidx.lifecycle.lifecycleScope
import com.k2fsa.sherpa.mnn.OfflineTtsConfig
import com.k2fsa.sherpa.mnn.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.mnn.OfflineTtsModelConfig
import com.taobao.meta.avatar.MHConfig
import com.taobao.meta.avatar.MainActivity
import com.taobao.meta.avatar.R
import com.taobao.meta.avatar.a2bs.A2BSService
import com.taobao.meta.avatar.a2bs.AudioBlendShapePlayer
import com.taobao.meta.avatar.audio.AudioChunksPlayer
import com.taobao.meta.avatar.tts.TtsService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class DebugModule {

    companion object {
        private const val TAG = "DebugModule"
        const val DEBUG_DISABLE_A2BS = false
        const val DEBUG_DISABLE_NNR = false
        const val DEBUG_DISABLE_SERVICE_AUTO_START = false
        const val DEBUG_USE_PRIVATE = false
        const val DEBUG_TTS_ENGLISH = true
        const val TTS_USE_SHERPA = false
    }

    private lateinit var ttsButton: View
    private lateinit var a2bsButton: View
    private lateinit var llmButton: View
    private lateinit var a2BSService: A2BSService
    private lateinit var ttsService: TtsService
    private lateinit var activity: MainActivity
    fun setupDebug(activity: MainActivity) {
        if (!MHConfig.DEBUG_MODE) {
            return
        }
        if (MHConfig.DEBUG_SCREEN_SHOT) {
            activity.mainView.viewMask.visibility = View.GONE
            return
        }
        this.activity = activity
        activity.findViewById<View>(R.id.debug_layout).visibility = View.VISIBLE
        ttsButton = activity.findViewById(R.id.test_tts_button)
        a2bsButton = activity.findViewById(R.id.test_a2bs_button)
        llmButton = activity.findViewById(R.id.test_llm_button)
        a2BSService = activity.getA2bsService()
        ttsService = activity.getTtsService()
        activity.findViewById<View>(R.id.test_asr_button).setOnClickListener { v: View? -> testAsr() }
        llmButton.setOnClickListener {
            testLlm()
        }
        a2bsButton.setOnClickListener {
            Log.d(TAG, "testA2bs begin")
            activity.lifecycleScope.launch {
                testA2bs()
            }
        }
        ttsButton.setOnClickListener {
            activity.lifecycleScope.launch {
                testTts("你好我是小淘给我讲个故事")
            }
        }
    }

    private fun testAsr() {
    }

    private fun testLlm() {
//        mLLMService!!.process("hello", mLLMTextViewHandler!!, mLLMResponseHandler!!)
    }

    private suspend fun testA2bs() {
        Log.d(TAG, "testA2bs begin")
        a2BSService.waitForInitComplete()
        Log.d(TAG, "testA2bs asr init completed")
        while (!activity.serviceInitialized()) {
            delay(200)
        }
        Log.d(TAG, "testA2bs all service init completed")
        val audioBsPlayer = activity.getAudioBlendShapePlayer()!!
//        testAudioBlendShapePlayer(audioBsPlayer, listOf(
//            "雨后的小巷总是弥漫着特别的气息",
//            "青石板路面上泛着湿润的光泽",
//            "墙角的绿萝显得格外翠绿。",
//            "远处传来断断续续的风铃声",
//            "那是街角那家老面馆的装饰",
//            "已经挂了整整十年",
//            "老板娘站在门口望着天空",
//            "七彩的霓虹灯牌在雨后的空气中晕染出柔和的光晕",
//            "几个匆匆归家的上班族快步走过",
//            "皮鞋踩过水洼发出清脆的声响。",
//            "夜幕慢慢降临，",
//            "街灯次第亮起",
//            "为这座城市描绘出另一番景象。",
//        ))
        testAudioBlendShapePlayer(audioBsPlayer, listOf(
            "123456789",
            "I am a english talker"
        ))
    }

    fun testAudioBlendShapePlayer(audioBsPlayer:AudioBlendShapePlayer, texts: List<String>) {
        audioBsPlayer.playSession(System.currentTimeMillis(), texts)
    }

    private suspend fun testTts(str:String) {
        if (TTS_USE_SHERPA) {
            CoroutineScope(Dispatchers.IO).launch {
                testKokoroZhEn()
            }
            return
        }
        var ttsService:TtsService? = null
        if (DEBUG_DISABLE_SERVICE_AUTO_START) {
            ttsService = TtsService()
            ttsService.init(MHConfig.TTS_MODEL_DIR, context = activity)
            ttsService.waitForInitComplete()
        } else {
            ttsService = activity.getTtsService()
        }
        var audioChunksPlayer = AudioChunksPlayer()
        audioChunksPlayer.start()
        if (DEBUG_TTS_ENGLISH) {
            listOf(
                "123 321",
                "I am a english talker"
            ).forEach {
                CoroutineScope(Dispatchers.Default).launch {
                    Log.d(TAG, "generate tts in thread: ${Thread.currentThread().name}")
                    audioChunksPlayer.playChunk(ttsService.process(it, 0))
                }
            }
        } else {
            listOf(
                "雨后的小巷总是弥漫着特别的气息",
                "青石板路面上泛着湿润的光泽",
                "墙角的绿萝显得格外翠绿。",
                "远处传来断断续续的风铃声",
                "那是街角那家老面馆的装饰",
                "已经挂了整整十年",
                "老板娘站在门口望着天空",
                "七彩的霓虹灯牌在雨后的空气中晕染出柔和的光晕",
                "几个匆匆归家的上班族快步走过",
                "皮鞋踩过水洼发出清脆的声响。",
                "夜幕慢慢降临，",
                "街灯次第亮起",
                "为这座城市描绘出另一番景象。",
            ).forEach {
                CoroutineScope(Dispatchers.Default).launch {
                    Log.d(TAG, "generate tts in thread: ${Thread.currentThread().name}")
                    audioChunksPlayer.playChunk(ttsService.process(it, 0))
                }
            }
        }


//        audioBsPlayer.playText("你好我是小小淘，两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船", 0, true)
//        audioBsPlayer.playText("你好我是小小淘，两个黄鹂鸣翠柳，一行白鹭上青天。", 0, true)

//        tesTtsList(audioBsPlayer, listOf("你好我是小淘", "两个黄鹂鸣翠柳", "一行白鹭上青天。"))
//        tesTtsList(audioBsPlayer, listOf(
//            "两个黄鹂鸣翠柳",
//            "你好我是小淘",
//            "一行白鹭上青天。",
//            "窗含西岭千秋雪",
//            "门泊东吴万里船"
//        ))
//        tesTtsList(audioBsPlayer, listOf(
//            "你好我是小小淘，两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船"
//        ))
//        tesTtsList(audioBsPlayer, listOf(
//            "床前明月光，疑是地上霜，举头望明月，低头思故乡。"
//        ))
//        tesTtsList(audioBsPlayer, listOf(
//            "床前明月光，疑是地上霜，举头望明月，低头思故乡。",
//        ))
//        tesTtsList(audioBsPlayer, listOf(
//            "床前明月光",
//            "疑是地上霜",
//            "举头望明月",
//            "低头思故乡"
//        ))


//        tesTtsList(audioBsPlayer, listOf(
//            "墙角的绿萝显得格外翠绿"
//        ))
//        tesTtsList(audioBsPlayer, listOf(
//            "雨后的小巷总是弥漫着特别的气息"
//        ))
//        tesTtsList(audioBsPlayer, listOf(
//            "青石板路面上泛着湿润的光泽",
//            "雨后的小巷总是弥漫着特别的气息"
//        ))

//        tesTtsList(audioBsPlayer, listOf(
//            "雨后的小巷总是弥漫着特别的气息，青石板路面上泛着湿润的光泽，墙角的绿萝显得格外翠绿。",
//        ))
//        tesTtsList(audioBsPlayer, listOf(
//            "雨后的小巷总是弥漫着特别的气息，青石板路面上泛着湿润的光泽，墙角的绿萝显得格外翠绿。远处传来断断续续的风铃声，那是街角那家老面馆的装饰，已经挂了整整十年。老板娘站在门口望着天空，七彩的霓虹灯牌在雨后的空气中晕染出柔和的光晕。几个匆匆归家的上班族快步走过，皮鞋踩过水洼发出清脆的声响。夜幕慢慢降临，街灯次第亮起，为这座城市描绘出另一番景象。",
//        ))
//        tesTtsList(audioBsPlayer, listOf(
//            "勇气岛上的居民都很勇敢",
//        ))
    }

    suspend fun testKokoroZhEn() {
        ttsService.init(MHConfig.TTS_MODEL_DIR, context = activity)
        ttsService.waitForInitComplete()
        val tts_path = "/data/local/tmp/kokoro-multi-lang-v1_0"
        val config = OfflineTtsConfig(
            model= OfflineTtsModelConfig(
                kokoro= OfflineTtsKokoroModelConfig(
                    model="${tts_path}/model.mnn",
                    voices="${tts_path}/voices.bin",
                    tokens="${tts_path}/tokens.txt",
                    dataDir="${tts_path}/espeak-ng-data",
                    dictDir="${tts_path}/dict",
                    lexicon="${tts_path}/lexicon-us-en.txt,${tts_path}/lexicon-zh.txt",
                ),
                numThreads=2,
                debug=true,
            ),
        )
        val tts = ttsService
        delay(1000)
        listOf(
            "雨后的小巷总是弥漫着特别的气息",
            "青石板路面上泛着湿润的光泽",
            "墙角的绿萝显得格外翠绿。",
            "远处传来断断续续的风铃声",
            "那是街角那家老面馆的装饰",
            "已经挂了整整十年",
            "老板娘站在门口望着天空",
            "七彩的霓虹灯牌在雨后的空气中晕染出柔和的光晕",
            "几个匆匆归家的上班族快步走过",
            "皮鞋踩过水洼发出清脆的声响。",
            "夜幕慢慢降临，",
            "街灯次第亮起",
            "为这座城市描绘出另一番景象。",
        ).forEach {
            CoroutineScope(Dispatchers.Default).launch {
               Log.d(TAG, "generate tts in thread: ${Thread.currentThread().name}")
               tts.processSherpa(it, 0)
            }
        }
//        val audioChunksPlayer = AudioChunksPlayer()
//        audioChunksPlayer.audioFormat = AudioFormat.ENCODING_PCM_FLOAT
//        audioChunksPlayer.sampleRate = audio.sampleRate
//        audioChunksPlayer.start()
//        delay(1000)
//        audioChunksPlayer.playChunk(audio.samples)
//        audio.save(filename="${activity.filesDir}/test-kokoro-zh-en.wav")
//        tts.release()
        println("Saved to test-kokoro-zh-en.wav")
    }
}