package com.alibaba.mnnllm.android.modelmarket

import com.google.gson.Gson
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class ModelMarketOpenClWarningTest {

    @Test
    fun `all qwen35 models include opencl warning extra tag`() {
        val jsonFile = File("src/main/assets/model_market.json")
        assertTrue("model_market.json should exist", jsonFile.exists())

        val root = Gson().fromJson(jsonFile.readText(), Map::class.java)
        val models = (root["models"] as? List<*>) ?: emptyList<Any>()

        val qwen35Models = models.mapNotNull { it as? Map<*, *> }
            .filter {
                val modelName = it["modelName"]?.toString() ?: ""
                modelName.contains("Qwen3.5", ignoreCase = true)
            }

        assertTrue("Expected at least one Qwen3.5 model", qwen35Models.isNotEmpty())

        qwen35Models.forEach { model ->
            val modelName = model["modelName"]?.toString() ?: "unknown"
            val extraTags = (model["extra_tags"] as? List<*>)?.mapNotNull { it?.toString() } ?: emptyList()
            val hasOpenclWarning = extraTags.any { it.equals("opencl_warning", ignoreCase = true) }
            assertTrue("$modelName should include opencl_warning in extra_tags", hasOpenclWarning)
        }
    }
}
