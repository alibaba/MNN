package com.alibaba.mnnllm.android.chat

import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class ChatActivityLifecycleTest {

    @Test
    fun destroy_withoutModelExtras_doesNotCrash() {
        val intent = Intent(ApplicationProvider.getApplicationContext(), ChatActivity::class.java)

        Robolectric.buildActivity(ChatActivity::class.java, intent)
            .create()
            .destroy()
    }
}
