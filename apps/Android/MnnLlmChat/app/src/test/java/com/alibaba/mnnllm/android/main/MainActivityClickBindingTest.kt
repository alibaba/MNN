package com.alibaba.mnnllm.android.main

import android.view.View
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import org.junit.Assert.assertTrue
import org.junit.Test

class MainActivityClickBindingTest {

    @Test
    fun activityMainOnClickHandlersResolveOnMainActivity() {
        val layoutFile = File("src/main/res/layout/activity_main.xml")
        val missingHandlers = mutableListOf<String>()

        val document = DocumentBuilderFactory.newInstance()
            .newDocumentBuilder()
            .parse(layoutFile)
        val nodes = document.getElementsByTagName("*")

        for (i in 0 until nodes.length) {
            val node = nodes.item(i)
            val attributes = node.attributes ?: continue
            val onClick = attributes.getNamedItem("android:onClick")?.nodeValue ?: continue
            val methodExists = MainActivity::class.java.methods.any { method ->
                method.name == onClick &&
                    method.parameterTypes.contentEquals(arrayOf(View::class.java))
            }
            if (!methodExists) {
                missingHandlers += onClick
            }
        }

        assertTrue(
            "activity_main.xml declares missing MainActivity onClick handlers: $missingHandlers",
            missingHandlers.isEmpty()
        )
    }
}
