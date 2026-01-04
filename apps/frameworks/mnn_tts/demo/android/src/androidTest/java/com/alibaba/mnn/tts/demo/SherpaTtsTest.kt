package com.alibaba.mnn.tts.demo

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class SherpaTtsTest {

    @Test
    fun testSherpaTtsInitializationWithMissingModel() = runBlocking {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        
        val sherpaTts = SherpaTts()
        
        // Use a dummy path that definitely doesn't have the model
        val dummyPath = File(appContext.cacheDir, "non_existent_model_dir").absolutePath
        
        // This should not crash, but handle the error gracefully
        sherpaTts.init(dummyPath)
        
        // Attempt to process text
        val result = sherpaTts.process("Hello world")
        
        // Since initialization failed (or model missing), result should be null
        assertNull("Result should be null when model is missing", result)
        
        sherpaTts.release()
    }
    
    @Test
    fun testSherpaTtsInitializationWithNullPath() = runBlocking {
         val sherpaTts = SherpaTts()
         
         // Init with null
         sherpaTts.init(null)
         
         val result = sherpaTts.process("Hello world")
         assertNull("Result should be null when path is null", result)
         
         sherpaTts.release()
    }
}
