package com.alibaba.mnnllm.android.chat

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.lifecycle.ActivityLifecycleCallback
import androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry
import androidx.test.runner.lifecycle.Stage
import androidx.test.uiautomator.By
import androidx.test.uiautomator.Configurator
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.agent.AgentCalculator
import com.alibaba.mnnllm.android.chat.agent.AgentPhase
import com.alibaba.mnnllm.android.chat.agent.AgentRunState
import com.alibaba.mnnllm.android.chat.agent.AgentStepKind
import com.alibaba.mnnllm.android.chat.agent.AgentStepStatus
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.main.MainActivity
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.manager.ServerEventManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

@RunWith(AndroidJUnit4::class)
class AgentChatDeviceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val device = UiDevice.getInstance(instrumentation)
    private val statusFailure = AtomicReference<Throwable?>()
    private val statusExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "agent-device-status").apply { isDaemon = true }
    }
    private var originalIdleTimeout = 0L
    private var currentCase = "AgentChatDeviceTest"
    private var workDeadline = Long.MAX_VALUE
    private var hardDeadline = Long.MAX_VALUE

    @Before
    fun disableUiIdleWait() {
        val now = SystemClock.elapsedRealtime()
        workDeadline = now + CASE_WORK_TIMEOUT
        hardDeadline = workDeadline + CLEANUP_TIMEOUT
        val configurator = Configurator.getInstance()
        originalIdleTimeout = configurator.waitForIdleTimeout
        configurator.waitForIdleTimeout = 0L
    }

    @After
    fun restoreUiIdleWait() {
        try {
            Configurator.getInstance().waitForIdleTimeout = originalIdleTimeout
            statusExecutor.shutdown()
            assertTrue("Instrumentation status delivery timed out",
                statusExecutor.awaitTermination(STATUS_TIMEOUT, TimeUnit.MILLISECONDS))
            statusFailure.get()?.let { throw AssertionError("Instrumentation status delivery failed", it) }
        } finally {
            statusExecutor.shutdownNow()
            checkpoint(currentCase, "test_teardown.end")
        }
    }

    @Test
    fun dumpCurrentUi() {
        beginCase("dumpCurrentUi")
        val output = ByteArrayOutputStream()
        device.dumpWindowHierarchy(output)
        val xml = output.toString(Charsets.UTF_8.name())
        assertTrue("No hierarchy was returned", xml.contains("<hierarchy"))
        val parts = xml.chunked(16_000)
        parts.forEachIndexed { index, part ->
            postStatus(Bundle().apply {
                putString("case", "dumpCurrentUi")
                putInt("ui_xml_part", index + 1)
                putInt("ui_xml_parts", parts.size)
                putString("ui_xml", part)
                putString("stream", "\nUI_XML ${index + 1}/${parts.size}\n$part\n")
            })
        }
    }

    @Test
    fun qwen35AgentCalculatesUsingRealTool() {
        val case = beginCase("qwen35AgentCalculatesUsingRealTool")
        val qwen = model("qwen", "Qwen3.5-0.8B-MNN")
        unchangedConfigs(listOf(qwen)) {
            withChat(qwen, case) { chat ->
                assertFalse(main { chat.presenter.isAgentMode })
                val toggle = device.wait(Until.findObject(By.res(context.packageName, "btn_agent_mode")), remainingBudget(UI_TIMEOUT))
                    ?: throw AssertionError("btn_agent_mode not found")
                assertTrue("Agent toggle is disabled", toggle.isEnabled)
                toggle.click()
                val title = context.getString(R.string.agent_consent_title)
                assertTrue("Agent disclosure missing", device.wait(Until.hasObject(By.text(title)), remainingBudget(UI_TIMEOUT)))
                val positive = device.wait(Until.findObject(By.res("android", "button1")), remainingBudget(UI_TIMEOUT))
                    ?: throw AssertionError("Agent disclosure positive button missing")
                assertTrue("Unexpected disclosure positive action", positive.text.equals(
                    context.getString(R.string.agent_consent_enable), ignoreCase = true))
                positive.click()
                awaitCondition("Agent confirmation and model reload", LOAD_TIMEOUT) {
                    chat.requireAlive()
                    main { chat.presenter.isAgentMode && chat.presenter.agentChat.mode.value.authorized } && chat.loaded()
                }
                runBlocking {
                    chat.presenter.agentChat.consent.confirmEnable(networkAllowed = false, persistenceAllowed = false)
                }
                assertTrue(main { chat.presenter.agentChat.mode.value.authorized })
                assertEquals(qwen.id, chat.session.modelId())
                assertEquals(appBackend(qwen), nativeBackend(chat.session, case))
                chat.configure(384)
                report(case, "consent", "UI-positive confirmed; network=false; memory/skill persistence=false")
                val prompt = "Use the calculator tool to calculate 17 * 23. Reply with the numeric result."
                val result = chat.send(prompt)
                val state = result["agent_state"] as? AgentRunState
                    ?: throw AssertionError("Real AgentRunState missing: ${result.keys}")
                val answer = chat.displayText()
                report(case, "result", "phase=${state.phase}; answer=$answer; steps=" +
                    state.steps.joinToString { "${it.kind}:${it.status}:${it.detail.take(64)}:${it.result.take(64)}" })
                assertSuccessful(result)
                assertSuccessful(state.metrics)
                assertEquals("Agent error", null, state.error)
                assertEquals(AgentPhase.COMPLETED, state.phase)
                assertTrue("Final answer does not contain 391: $answer", NUMBER_391.containsMatchIn(answer))
                assertCleanDisplay(answer)
                assertCleanDisplay(result["response"] as? String ?: "")
                assertTrue("No completed real calculator step", state.steps.any {
                    val expression = AgentCalculator.normalize(it.detail).replace(Regex("[\\s()]"), "")
                    it.kind == AgentStepKind.CALCULATOR && it.status == AgentStepStatus.DONE &&
                        NUMBER_391.containsMatchIn(it.result) && expression in setOf("17*23", "23*17")
                })
                assertTrue("Unexpected or unsuccessful tool step", state.steps.all {
                    it.kind in setOf(AgentStepKind.CALCULATOR, AgentStepKind.TIME) && it.status == AgentStepStatus.DONE
                })
                assertTrue("No real tool round occurred", state.toolCalls > 0 && state.pass >= 2)
                val history = ChatDataManager.getInstance(context).getChatDataBySession(chat.session.sessionId)
                assertEquals("User history was duplicated", listOf(prompt),
                    history.filter { it.type == ChatViewHolders.USER }.map { it.text })
                if (InstrumentationRegistry.getArguments().getString("checkToolStepStore") == "true") {
                    val records = runBlocking { chat.presenter.agentChat.store.toolStepsForSession(chat.session.sessionId) }
                    assertTrue("Calculator metadata not persisted", records.any { record ->
                        record.phase == AgentPhase.COMPLETED && record.steps.any {
                            it.kind == AgentStepKind.CALCULATOR && it.status == AgentStepStatus.DONE
                        }
                    })
                    report(case, "store", "calculator DONE metadata persisted")
                }
            }
        }
    }

    @Test
    fun hunyuanOpenclThenQwenFirstEntry() {
        val case = beginCase("hunyuanOpenclThenQwenFirstEntry")
        val hunyuan = model("hunyuan", "Hunyuan-0.5B-Instruct-MNN")
        val qwen = model("qwen", "Qwen3.5-0.8B-MNN")
        requireRealApp()
        assertEquals("Qwen App config must already select CPU; test does not save settings", "cpu", appBackend(qwen))
        unchangedConfigs(listOf(hunyuan, qwen)) {
            report(case, "scope", "direct Hunyuan OpenCL -> first Qwen CPU ChatActivity; NOT Hunyuan Activity/back navigation")
            val direct = DirectSession(hunyuan, case)
            try {
                direct.work("load") { direct.session.load() }
                assertTrue("Hunyuan native model not loaded", direct.session.isModelLoaded())
                assertEquals("opencl", nativeBackend(direct.session, case))
                configure(direct.session, 64)
                val processor = GenerateResultProcessor().apply { generateBegin() }
                val result = direct.work("generate") {
                    direct.session.generate(HELLO_PROMPT, emptyMap(), object : GenerateProgressListener {
                        override fun onProgress(progress: String?): Boolean {
                            processor.process(progress)
                            return direct.stop.get()
                        }
                    })
                }
                assertSuccessful(result)
                val item = ChatDataItem(ChatViewHolders.ASSISTANT).apply {
                    text = processor.getRawResult()
                    displayText = processor.getNormalOutput()
                }
                val display = item.displayText.orEmpty()
                report(case, "hunyuan", "nativeBackendConfig=opencl; displayText=$display")
                assertCleanDisplay(display)
            } finally {
                direct.close()
            }
            withChat(qwen, case) { chat ->
                assertEquals("cpu", nativeBackend(chat.session, case))
                chat.configure(64)
                assertSuccessful(chat.send(HELLO_PROMPT))
                assertCleanDisplay(chat.displayText())
                chat.requireAlive()
                report(case, "first-entry", "Qwen first launch remained alive and generated; answer=${chat.displayText()}")
            }
        }
    }

    @Test
    fun hunyuanAppThenQwenFirstEntry() {
        val case = beginCase("hunyuanAppThenQwenFirstEntry")
        val hunyuan = model("hunyuan", "Hunyuan-0.5B-Instruct-MNN")
        val qwen = model("qwen", "Qwen3.5-0.8B-MNN")
        unchangedConfigs(listOf(hunyuan, qwen)) {
            withChat(hunyuan, case) { chat ->
                report(case, "scope", "real Hunyuan Activity/back -> first Qwen Activity; Hunyuan saved backend=" +
                    nativeBackend(chat.session, case) + "; OpenCL App-return evidence ONLY if this is opencl")
                chat.configure(64)
                assertSuccessful(chat.send(HELLO_PROMPT))
                assertCleanDisplay(chat.displayText())
                report(case, "hunyuan", "displayText=${chat.displayText()}")
                assertTrue("Back key was not delivered", device.pressBack())
                awaitCondition("Hunyuan Activity back/destroy", UI_TIMEOUT) { main { chat.activity.isDestroyed } }
            }
            withChat(qwen, case) { chat ->
                chat.configure(64)
                assertSuccessful(chat.send(HELLO_PROMPT))
                assertCleanDisplay(chat.displayText())
                chat.requireAlive()
                report(case, "first-entry", "first Qwen Activity remained alive; displayText=${chat.displayText()}")
            }
        }
    }

    private data class DeviceModel(val id: String, val name: String, val config: File)

    private fun model(argumentPrefix: String, name: String): DeviceModel {
        val id = InstrumentationRegistry.getArguments().getString("${argumentPrefix}ModelId")
            ?: "ModelScope/MNN/$name"
        assertTrue("Expected the official ModelScope model ID, got $id", id in setOf("ModelScope/MNN/$name", "MNN/$name"))
        val directory = File(context.filesDir, ".mnnmodels/modelscope/$name")
        val config = File(directory, "config.json")
        assertTrue("Model config missing/empty; finish the model-market download: $config", config.isFile && config.length() > 0)
        val json = JSONObject(config.readText())
        for ((key, fallback) in listOf("llm_model" to "llm.mnn", "llm_weight" to "llm.mnn.weight")) {
            val path = json.optString(key, fallback)
            assertTrue("Empty $key in $config", path.isNotBlank())
            val file = if (File(path).isAbsolute) File(path) else File(directory, path)
            assertTrue("Required $key missing/empty: $file", file.isFile && file.canRead() && file.length() > 0)
        }
        report(currentCase, "model", "modelId=$id; config=$config")
        checkpoint(currentCase, "model_returned", "modelId=$id")
        return DeviceModel(id, name, config)
    }

    private fun requireRealApp() {
        traced(currentCase, "requireRealApp.prefs") {
            assertFalse("Disable API service explicitly before this test; no settings are changed", MainSettings.isApiServiceEnabled(context))
        }
        traced(currentCase, "requireRealApp.server_state") {
            assertFalse("An API server is already running", ServerEventManager.getInstance().isServerRunning())
        }
        traced(currentCase, "requireRealApp.llm_clinit") {
            assertFalse("Native mockLatex must not be enabled", LlmSession.mockLatex)
        }
        val runtime = traced(currentCase, "requireRealApp.runtime_lookup") {
            ServiceLocator.getLlmRuntimeController()
        }
        traced(currentCase, "requireRealApp.active_model") {
            assertEquals("Return from the current chat before testing; never reuse user history", null, runtime.getActiveModelId())
        }
    }

    private fun appBackend(model: DeviceModel): String = traced(currentCase, "merged_config", model.id) {
        val overridePath = ModelConfig.getExtraConfigFile(model.id)
        val config = ModelConfig.loadMergedConfig(model.config.absolutePath, overridePath)
            ?: throw AssertionError("Cannot read merged model config")
        val backend = config.backendType ?: "cpu"
        assertTrue("Only CPU/OpenCL are permitted in this test: $backend", backend in setOf("cpu", "opencl"))
        backend
    }

    private fun nativeBackend(session: LlmSession, case: String): String {
        val backend = JSONObject(session.dumpConfig()).getString("backend_type")
        assertTrue("Unexpected native backend config: $backend", backend in setOf("cpu", "opencl"))
        report(case, "backend", "modelId=${session.modelId()}; nativeBackendConfig=$backend; actual GPU dispatch/fallback not exposed by this API")
        return backend
    }

    private fun configure(session: LlmSession, tokens: Int) {
        val jinja = JSONObject(session.dumpConfig()).optJSONObject("jinja") ?: JSONObject()
        val jinjaContext = jinja.optJSONObject("context") ?: JSONObject()
        jinja.put("context", jinjaContext.put("enable_thinking", false))
        session.updateConfig(JSONObject().put("sampler_type", "greedy").put("jinja", jinja).toString())
        session.updateMaxNewTokens(tokens)
        session.setEnableAudioOutput(false)
        val runtime = JSONObject(session.dumpConfig())
        assertEquals("greedy", runtime.getString("sampler_type"))
        assertFalse(runtime.getJSONObject("jinja").getJSONObject("context").getBoolean("enable_thinking"))
        assertEquals(tokens, session.captureAgentConfig().maxNewTokens)
    }

    private fun isVisibleAppActivity(activity: Activity): Boolean =
        activity.packageName == context.packageName && (activity is MainActivity || activity is ChatActivity) &&
            !activity.isFinishing && !activity.isDestroyed && activity.window.decorView.isShown && activity.hasWindowFocus()

    private fun resumedAppActivity(): Activity? = main {
        ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(Stage.RESUMED)
            .firstOrNull { isVisibleAppActivity(it) }
    }

    private fun ensureForegroundUi(case: String): Activity {
        val appLabel = context.packageManager.getApplicationLabel(context.applicationInfo).toString()
        return traced(case, "foreground_UI", "appLabel=$appLabel; package=${context.packageName}") {
            assertTrue("Application label is empty", appLabel.isNotBlank())
            val existing = resumedAppActivity()
            if (existing != null) {
                checkpoint(case, "foreground_UI.already_resumed", "activity=${existing.javaClass.simpleName}")
                existing
            } else {
                val observedMain = AtomicReference<MainActivity?>()
                val monitor = main { ActivityLifecycleMonitorRegistry.getInstance() }
                val callback = ActivityLifecycleCallback { activity, stage ->
                    if (activity is MainActivity && activity.packageName == context.packageName) {
                        if (stage == Stage.RESUMED) observedMain.set(activity)
                        checkpoint(case, "foreground_UI.callback", "appLabel=$appLabel; lifecycle=$stage")
                    }
                }
                var registered = false
                try {
                    main { monitor.addLifecycleCallback(callback); registered = true }
                    assertTrue("Home key was not delivered; please open $appLabel manually", device.pressHome())
                    val launcherPackage = device.launcherPackageName
                        ?: throw AssertionError("Cannot identify the launcher; please open $appLabel manually")
                    var icon: UiObject2? = null
                    awaitCondition("No visible desktop icon for '$appLabel' (${context.packageName}). " +
                        "Place this app's icon on the current desktop, or manually open MNN's main screen before testing. " +
                        "No other desktop pages, apps or system settings will be searched.", UI_TIMEOUT) {
                        val texts = device.findObjects(By.pkg(launcherPackage).text(appLabel))
                        val descriptions = if (texts.isEmpty()) device.findObjects(By.pkg(launcherPackage).desc(appLabel)) else emptyList()
                        val exactPackage = device.findObject(By.pkg(launcherPackage).desc(context.packageName))
                        if (texts.size > 1 || descriptions.size > 1) {
                            icon = exactPackage ?: throw AssertionError("Ambiguous desktop icons for '$appLabel'. " +
                                "Please manually open ${context.packageName} on MNN's main screen before testing.")
                        } else {
                            icon = texts.singleOrNull() ?: descriptions.singleOrNull() ?: exactPackage
                        }
                        icon != null
                    }
                    checkpoint(case, "foreground_UI.icon_click", "appLabel=$appLabel; launcher=$launcherPackage")
                    checkNotNull(icon).click()
                    awaitCondition("MNN MainActivity did not become visibly RESUMED after the desktop icon click. " +
                        "Please manually open '$appLabel' on its main screen; this test will not change permissions or retry the launch.",
                        LAUNCH_TIMEOUT) {
                        main {
                            observedMain.get()?.let {
                                isVisibleAppActivity(it) && monitor.getLifecycleStageOf(it) == Stage.RESUMED
                            } == true
                        }
                    }
                    checkNotNull(observedMain.get())
                } finally {
                    if (registered) main { monitor.removeLifecycleCallback(callback) }
                }
            }
        }
    }

    private fun <T> withChat(model: DeviceModel, case: String, action: (LiveChat) -> T): T {
        requireRealApp()
        val foreground = ensureForegroundUi(case)
        val expectedBackend = appBackend(model)
        val id = "agent-device-${UUID.randomUUID()}"
        val intent = Intent(foreground, ChatActivity::class.java).apply {
            putExtra("modelId", model.id)
            putExtra("modelName", model.name)
            putExtra("configFilePath", model.config.absolutePath)
            putExtra("chatSessionId", id)
        }
        val observed = AtomicReference<ChatActivity?>()
        val lifecycleStage = AtomicReference<Stage?>()
        val callbackFailure = AtomicReference<Throwable?>()
        val callback = ActivityLifecycleCallback { activity, stage ->
            if (activity is ChatActivity && activity.packageName == context.packageName &&
                activity.intent.getStringExtra("chatSessionId") == id) {
                observed.compareAndSet(null, activity)
                if (observed.get() !== activity) {
                    callbackFailure.compareAndSet(null, AssertionError("Unexpected test Activity recreation"))
                    activity.finish()
                } else {
                    lifecycleStage.set(stage)
                }
                checkpoint(case, "launch_callback", "sessionId=$id; lifecycle=$stage")
            }
        }
        val monitor = traced(case, "lifecycle_monitor") { main { ActivityLifecycleMonitorRegistry.getInstance() } }
        var registered = false
        var chat: LiveChat? = null
        try {
            traced(case, "lifecycle_register") { main { monitor.addLifecycleCallback(callback); registered = true } }
            checkpoint(case, "launch.begin", "sessionId=$id; model=${model.name}")
            main {
                assertTrue("MNN is no longer visibly RESUMED; refusing a background Activity start",
                    isVisibleAppActivity(foreground) && monitor.getLifecycleStageOf(foreground) == Stage.RESUMED)
                foreground.startActivity(intent)
            }
            checkpoint(case, "launch.dispatched", "sessionId=$id")
            awaitCondition("Target ChatActivity RESUMED", LAUNCH_TIMEOUT) {
                callbackFailure.get()?.let { throw it }
                val target = observed.get() ?: return@awaitCondition false
                main {
                    assertFalse("ChatActivity finished during first launch", target.isFinishing || target.isDestroyed)
                    assertEquals(id, target.intent.getStringExtra("chatSessionId"))
                    assertEquals(model.id, target.intent.getStringExtra("modelId"))
                    lifecycleStage.get() == Stage.RESUMED
                }
            }
            checkpoint(case, "launch.end", "sessionId=$id; lifecycle=RESUMED")
            checkpoint(case, "presenter_available.begin", "sessionId=$id")
            val live = LiveChat(checkNotNull(observed.get()), case)
            chat = live
            assertEquals("Test sessionFactory injection is not allowed", null,
                main { field<Any?>(live.presenter, "sessionFactory") })
            checkpoint(case, "presenter_available.end", "sessionId=$id")
            awaitCondition("${model.name} first model load", LOAD_TIMEOUT) { live.requireAlive(); live.loaded() }
            assertEquals("A cached/user conversation was reused", id, live.session.sessionId)
            assertEquals(id, main { live.activity.sessionId })
            assertTrue("Expected empty new conversation", main { live.presenter.getChatHistory().isEmpty() })
            assertEquals(expectedBackend, nativeBackend(live.session, case))
            report(case, "first_loaded", "sessionId=$id; model=${model.name}; firstLaunch=true")
            return action(live)
        } finally {
            try {
                val target = observed.get()
                if (chat != null) {
                    chat!!.close()
                } else if (target != null) {
                    if (main { field<ChatPresenter?>(target, "chatPresenter") } != null) {
                        LiveChat(target, case).close()
                    } else {
                        main { target.finish() }
                        awaitCondition("Uninitialized target Activity destroyed", CLEANUP_TIMEOUT, cleanup = true) {
                            main { target.isDestroyed }
                        }
                    }
                }
            } finally {
                if (registered) main { monitor.removeLifecycleCallback(callback) }
                checkpoint(case, "launch_callback.unregistered", "sessionId=$id")
            }
        }
    }

    private inner class LiveChat(val activity: ChatActivity, val case: String) {
        val presenter: ChatPresenter = main { field(activity, "chatPresenter") }
        private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
        private var turn: Deferred<HashMap<String, Any>>? = null
        private val sessions = linkedSetOf<LlmSession>()
        val session: LlmSession get() = main { activity.chatSession as? LlmSession }
            ?.also { sessions.add(it) } ?: throw AssertionError("Real LlmSession missing")

        fun requireAlive() = main {
            assertFalse("ChatActivity finished on first entry", activity.isFinishing || activity.isDestroyed)
        }

        fun loaded(): Boolean = main {
            (activity.chatSession as? LlmSession)?.also { sessions.add(it) }?.isModelLoaded() == true &&
                !activity.isLoading && jobsQuiet(presenter)
        }

        fun configure(tokens: Int) {
            this@AgentChatDeviceTest.configure(session, tokens)
        }

        fun send(prompt: String): HashMap<String, Any> {
            requireAlive()
            assertTrue("Chat model is not ready", loaded())
            assertEquals("Only one send per test conversation", null, turn)
            checkpoint(case, "presenter_send.begin")
            val pending = scope.async { presenter.sendMessage(prompt) }
            turn = pending
            awaitCondition("Real presenter/inference/tool completion", GENERATION_TIMEOUT) { pending.isCompleted }
            val result = runBlocking { pending.await() }
            checkpoint(case, "presenter_send.end")
            awaitCondition("Generation UI completion", UI_TIMEOUT) { main { !activity.isGenerating } }
            requireAlive()
            return result
        }

        fun displayText(): String = main { activity.chatListComponent.recentItem?.displayText.orEmpty() }

        private fun rememberSessions() = main {
            (activity.chatSession as? LlmSession)?.let { sessions.add(it) }
            for (name in listOf("chatSession", "loadingSession", "ownedAgentSession")) {
                (field<Any?>(presenter, name) as? LlmSession)?.let { sessions.add(it) }
            }
        }

        fun close() {
            checkpoint(case, "cleanup.begin")
            try {
                rememberSessions()
                main {
                    presenter.stopGenerate()
                    if (!activity.isDestroyed) activity.finish()
                }
                checkpoint(case, "finish.dispatched")
                awaitCondition("Actual generation/native release and return to visible MNN after destroy", CLEANUP_TIMEOUT, cleanup = true) {
                    rememberSessions()
                    turn?.isCompleted != false && jobsQuiet(presenter) && sessions.all(::nativeReleased) &&
                        !presenter.agentChat.consent.state.value.enabled && main { activity.isDestroyed } && resumedAppActivity() != null
                }
                report(case, "cleanup", "Test ChatActivity destroyed; generation completed; native release acknowledged; MNN remains visible")
            } finally {
                scope.cancel()
            }
        }
    }

    private inner class DirectSession(model: DeviceModel, private val case: String) {
        val session = LlmSession("local/agent-device-${UUID.randomUUID()}/${model.name}",
            "agent-device-${UUID.randomUUID()}", model.config.absolutePath, null,
            backendType = "opencl", useCustomConfig = false)
        val stop = AtomicBoolean(false)
        private var workDone: CountDownLatch? = null

        fun <T> work(stage: String, block: () -> T): T {
            val timeout = remainingBudget(if (stage == "load") LOAD_TIMEOUT else GENERATION_TIMEOUT)
            checkpoint(case, "direct_$stage.begin", "timeoutMs=$timeout")
            val done = CountDownLatch(1)
            val result = AtomicReference<Result<T>>()
            workDone = done
            Thread({
                try { result.set(runCatching(block)) }
                finally { done.countDown() }
            }, "agent-device-$stage").start()
            assertTrue("Real native $stage timed out; safe stop happens in finally", done.await(timeout, TimeUnit.MILLISECONDS))
            checkpoint(case, "direct_$stage.end")
            return result.get().getOrThrow()
        }

        fun close() {
            checkpoint(case, "direct_cleanup.begin")
            stop.set(true)
            session.cancelContextWork()
            val done = workDone
            if (done != null && done.count > 0) {
                assertTrue("Native worker did not exit; refusing to release in-flight native code",
                    done.await(remainingBudget(CLEANUP_TIMEOUT, cleanup = true), TimeUnit.MILLISECONDS))
            }
            session.release()
            awaitCondition("Hunyuan native release acknowledgment", CLEANUP_TIMEOUT, cleanup = true) { nativeReleased(session) }
            report(case, "cleanup", "direct OpenCL worker exited; native release acknowledged")
        }
    }

    private fun jobsQuiet(presenter: ChatPresenter): Boolean = listOf("loadJob", "activeGeneration", "agentGeneration")
        .all { field<Job?>(presenter, it)?.isCompleted != false }

    private fun nativeReleased(session: LlmSession): Boolean {
        val service = ChatService.provide()
        return synchronized(field<Any>(service, "loadLock")) {
            field<Any?>(service, "residentSession") !== session && field<Long>(session, "nativePtr") == 0L
        }
    }

    private fun remainingBudget(timeout: Long, cleanup: Boolean = false): Long {
        val deadline = if (cleanup) hardDeadline else workDeadline
        val remaining = deadline - SystemClock.elapsedRealtime()
        assertTrue("Case ${if (cleanup) "cleanup" else "work"} deadline exceeded", remaining > 0)
        return minOf(timeout, remaining)
    }

    private fun awaitCondition(description: String, timeout: Long, cleanup: Boolean = false, condition: () -> Boolean) {
        assertFalse("Condition waits must not block the main thread", Looper.myLooper() == Looper.getMainLooper())
        val budget = remainingBudget(timeout, cleanup)
        checkpoint(currentCase, "await.begin", "$description; timeoutMs=$budget")
        val completed = runBlocking {
            withTimeoutOrNull(budget) {
                while (!condition()) delay(100)
                true
            }
        } == true
        checkpoint(currentCase, "await.end", "$description; completed=$completed")
        assertTrue("Timed out after ${budget}ms: $description", completed)
    }

    private fun assertSuccessful(result: Map<String, Any>) {
        for (key in listOf("error", "session_invalid", "session_unavailable", "context_rejected", "agent_persistence_failed")) {
            assertFalse("$key: ${result["message"]}", result[key] == true)
        }
        assertFalse("Native generation failed", result["success"] == false)
        assertFalse("Agent request rejected", result.containsKey("agent_rejected"))
    }

    private fun assertCleanDisplay(text: String) {
        assertTrue("Empty displayText", text.isNotBlank())
        for (marker in listOf("<answer", "</answer>", "tool_call", "<function=", "<think", "</think>",
            "```", "{", "}", "[TOOL_ERROR]", "Error:", "generation_failed")) {
            assertFalse("Protocol/error leaked into displayText: $text", text.contains(marker, ignoreCase = true))
        }
    }

    private fun <T> unchangedConfigs(models: List<DeviceModel>, block: () -> T): T {
        val before = traced(currentCase, "config-snapshot") {
            val files = models.flatMap { model ->
                val extra = traced(currentCase, "extra_config_path", model.id) { ModelConfig.getExtraConfigFile(model.id) }
                listOf(model.config, File(extra))
            }.distinct()
            files.associateWith { file ->
                traced(currentCase, "config_snapshot_read", file.absolutePath) {
                    if (file.exists()) {
                        assertTrue("Config is not a regular file: $file", file.isFile)
                        file.readBytes()
                    } else null
                }
            }
        }
        try {
            return block()
        } finally {
            traced(currentCase, "config_verify") {
                before.forEach { (file, bytes) ->
                    if (bytes == null) assertFalse("Test created a user config: $file", file.exists())
                    else {
                        assertTrue("User config disappeared: $file", file.isFile)
                        assertArrayEquals("User config was modified: $file", bytes, file.readBytes())
                    }
                }
            }
        }
    }

    private fun beginCase(name: String): String {
        currentCase = name
        checkpoint(name, "test.begin", "workBudgetMs=$CASE_WORK_TIMEOUT; cleanupReserveMs=$CLEANUP_TIMEOUT")
        return name
    }

    private fun checkpoint(case: String, stage: String, details: String = "") {
        Log.i(TAG, "case=$case stage=$stage uptimeMs=${SystemClock.elapsedRealtime()} " +
            "thread=${Thread.currentThread().name} main=${Looper.myLooper() == Looper.getMainLooper()} ${details.take(1800)}")
    }

    private fun <T> traced(case: String, stage: String, details: String = "", block: () -> T): T {
        checkpoint(case, "$stage.begin", details)
        try {
            val result = block()
            checkpoint(case, "$stage.end", details)
            return result
        } catch (failure: Throwable) {
            checkpoint(case, "$stage.fail", failure.javaClass.simpleName)
            throw failure
        }
    }

    private fun postStatus(bundle: Bundle) {
        statusExecutor.execute {
            try {
                instrumentation.sendStatus(2, bundle)
            } catch (failure: Throwable) {
                statusFailure.compareAndSet(null, failure)
                checkpoint(currentCase, "status.fail", failure.javaClass.simpleName)
            }
        }
    }

    private fun report(case: String, stage: String, details: String) {
        val text = details.take(1800)
        checkpoint(case, stage, text)
        postStatus(Bundle().apply {
            putString("case", case)
            putString("stage", stage)
            putString("details", text)
            putString("stream", "\n$TAG case=$case stage=$stage $text\n")
        })
    }

    private fun <T> main(block: () -> T): T {
        if (Looper.myLooper() == Looper.getMainLooper()) return block()
        var result: Result<T>? = null
        instrumentation.runOnMainSync { result = runCatching(block) }
        return checkNotNull(result).getOrThrow()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T> field(instance: Any, name: String): T = instance.javaClass.getDeclaredField(name)
        .apply { isAccessible = true }.get(instance) as T

    companion object {
        private const val TAG = "AgentChatDeviceTest"
        private const val UI_TIMEOUT = 15_000L
        private const val LAUNCH_TIMEOUT = 30_000L
        private const val LOAD_TIMEOUT = 120_000L
        private const val GENERATION_TIMEOUT = 120_000L
        private const val CASE_WORK_TIMEOUT = 360_000L
        private const val CLEANUP_TIMEOUT = 60_000L
        private const val STATUS_TIMEOUT = 5_000L
        private const val HELLO_PROMPT = "Say hello in one short sentence."
        private val NUMBER_391 = Regex("(?<![0-9])391(?![0-9])")
    }
}
