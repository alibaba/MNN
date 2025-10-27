package com.alibaba.mnnllm.api.openai.network.processors

import android.content.Context
import kotlinx.coroutines.runBlocking
import timber.log.Timber
import java.util.Base64

/** * imagecachetesttoolclass，forverificationdualhashcachemechanism*/
class ImageCacheTest {
    
    companion object {
        private const val TAG = "ImageCacheTest"
        
        /** * testdualhashcachemechanism * simulatedifferentclienttosameimagedifferentBase64encodingmethod*/
        fun testDualHashCache(context: Context) {
            val processor = MnnImageProcessor.getInstance(context)
            
            //simulate onesimpleimage data (1x1 pixelred PNG)
            val originalImageBytes = byteArrayOf(
                0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  // PNG signature
                0x00, 0x00, 0x00, 0x0D,  // IHDR chunk length
                0x49, 0x48, 0x44, 0x52,  // IHDR
                0x00, 0x00, 0x00, 0x01,  // width: 1
                0x00, 0x00, 0x00, 0x01,  // height: 1
                0x08, 0x02, 0x00, 0x00, 0x00,  // bit depth, color type, compression, filter, interlace
                0x90.toByte(), 0x77, 0x53, 0xDE.toByte(),  // CRC
                0x00, 0x00, 0x00, 0x0C,  // IDAT chunk length
                0x49, 0x44, 0x41, 0x54,  // IDAT
                0x08, 0x99.toByte(), 0x01, 0x01, 0x00, 0x00, 0x00, 0xFF.toByte(), 0xFF.toByte(), 0x00, 0x00, 0x00,
                0x02, 0x00, 0x01,  // IDAT data
                0xE2.toByte(), 0x21, 0xBC.toByte(), 0x33,  // CRC
                0x00, 0x00, 0x00, 0x00,  // IEND chunk length
                0x49, 0x45, 0x4E, 0x44,  // IEND
                0xAE.toByte(), 0x42, 0x60, 0x82.toByte()   // CRC
            )
            
            //generatedifferentBase64encoding（simulatedifferentclientencodingdifferences）
            val base64Version1 = Base64.getEncoder().encodeToString(originalImageBytes)
            val base64Version2 = Base64.getEncoder().withoutPadding().encodeToString(originalImageBytes)
            val base64Version3 = Base64.getMimeEncoder().encodeToString(originalImageBytes)
            
            logInfo("开始测试双重哈希缓存机制")
            logInfo("原始图像数据大小: ${originalImageBytes.size} bytes")
            logInfo("Base64版本1长度: ${base64Version1.length}")
            logInfo("Base64版本2长度: ${base64Version2.length}")
            logInfo("Base64版本3长度: ${base64Version3.length}")
            
            runBlocking {
                //first timeprocess - shouldcreatenewcache
                logInfo("\n=== 第一次处理（版本1）===")
                val startTime1 = System.currentTimeMillis()
                val result1 = processor.processImageUrl("data:image/png;base64,$base64Version1")
                val time1 = System.currentTimeMillis() - startTime1
                logInfo("结果1: $result1")
                logInfo("耗时1: ${time1}ms")
                logInfo(processor.getCacheStats())
                
                //second timeprocesssamecontentbutdifferentencoding - shouldhitcontenthashcache
                logInfo("\n=== 第二次处理（版本2，相同内容不同编码）===")
                val startTime2 = System.currentTimeMillis()
                val result2 = processor.processImageUrl("data:image/png;base64,$base64Version2")
                val time2 = System.currentTimeMillis() - startTime2
                logInfo("结果2: $result2")
                logInfo("耗时2: ${time2}ms")
                logInfo("是否复用文件: ${result1 == result2}")
                logInfo(processor.getCacheStats())
                
                //third timeprocess - shouldhitstringhashcache
                logInfo("\n=== 第三次处理（版本1重复）===")
                val startTime3 = System.currentTimeMillis()
                val result3 = processor.processImageUrl("data:image/png;base64,$base64Version1")
                val time3 = System.currentTimeMillis() - startTime3
                logInfo("结果3: $result3")
                logInfo("耗时3: ${time3}ms")
                logInfo("是否复用文件: ${result1 == result3}")
                logInfo(processor.getCacheStats())
                
                //fourth timeprocessversion3 - shouldhitcontenthashcache
                logInfo("\n=== 第四次处理（版本3，MIME编码）===")
                val startTime4 = System.currentTimeMillis()
                val result4 = processor.processImageUrl("data:image/png;base64,$base64Version3")
                val time4 = System.currentTimeMillis() - startTime4
                logInfo("结果4: $result4")
                logInfo("耗时4: ${time4}ms")
                logInfo("是否复用文件: ${result1 == result4}")
                logInfo(processor.getCacheStats())
                
                //performancecomparisonanalysis
                logInfo("\n=== 性能分析 ===")
                logInfo("首次处理耗时: ${time1}ms（包含解码+保存）")
                logInfo("字符串哈希命中耗时: ${time3}ms（应该<5ms）")
                logInfo("内容哈希命中耗时: ${time2}ms, ${time4}ms（应该<解码时间）")
                
                val speedup2 = if (time2 > 0) time1.toFloat() / time2 else Float.MAX_VALUE
                val speedup3 = if (time3 > 0) time1.toFloat() / time3 else Float.MAX_VALUE
                val speedup4 = if (time4 > 0) time1.toFloat() / time4 else Float.MAX_VALUE
                
                logInfo("内容哈希加速比: ${String.format("%.2f", speedup2)}x, ${String.format("%.2f", speedup4)}x")
                logInfo("字符串哈希加速比: ${String.format("%.2f", speedup3)}x")
                
                //verificationallresultpoint tosamefile
                val allSame = result1 == result2 && result2 == result3 && result3 == result4
                logInfo("\n=== 缓存一致性验证 ===")
                logInfo("所有处理结果指向同一文件: $allSame")
                if (allSame) {
                    logInfo("✅ 双重哈希缓存机制工作正常")
                } else {
                    logInfo("❌ 双重哈希缓存机制存在问题")
                    logInfo("结果对比: $result1 vs $result2 vs $result3 vs $result4")
                }
            }
        }
        
        /**
         * testcachecleanupfunction*/
        fun testCacheCleanup(context: Context) {
            val processor = MnnImageProcessor.getInstance(context)
            
            logInfo("\n=== 缓存清理测试 ===")
            logInfo("清理前: ${processor.getCacheStats()}")
            
            //cleanupexpiredcache（settingvery shortexpiredtimetotest）
            processor.cleanupCache(1) //1msexpiredtime，forcecleanupallcache
            
            logInfo("清理后: ${processor.getCacheStats()}")
        }
        
        private fun logInfo(message: String) {
            Timber.tag(TAG).i(message)
            println("[$TAG] $message") //simultaneouslyoutputtoconsolefordebug
        }
    }
}