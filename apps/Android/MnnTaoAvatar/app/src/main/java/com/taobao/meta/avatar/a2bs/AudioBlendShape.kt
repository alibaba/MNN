package com.taobao.meta.avatar.a2bs

import com.k2fsa.sherpa.mnn.GeneratedAudio

class AudioBlendShape(
                      val id:Int,
                      val is_last:Boolean,
                      val text:String,
                      var audio: ShortArray,
                      var audio_sherpa: GeneratedAudio?,
                      var a2bs: AudioToBlendShapeData)
