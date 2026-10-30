package com.alibaba.mnnllm.android.history

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatRouter
import com.alibaba.mnnllm.android.chat.chatlist.ChatRecyclerViewAdapter
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.chat.model.SessionItem
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** Browsing a saved conversation must not construct an inference session or start downloads. */
class HistoryViewerActivity : AppCompatActivity() {
    private lateinit var adapter: ChatRecyclerViewAdapter
    private lateinit var status: TextView
    private lateinit var continueButton: Button
    private var session: SessionItem? = null
    private var loadJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history_viewer)
        setSupportActionBar(findViewById<Toolbar>(R.id.history_toolbar))
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        title = getString(R.string.history_title)
        status = findViewById(R.id.history_status)
        continueButton = findViewById(R.id.history_continue)
        adapter = ChatRecyclerViewAdapter(this, readOnly = true)
        adapter.stateRestorationPolicy = RecyclerView.Adapter.StateRestorationPolicy.PREVENT_WHEN_EMPTY
        findViewById<RecyclerView>(R.id.history_messages).apply {
            layoutManager = LinearLayoutManager(this@HistoryViewerActivity)
            adapter = this@HistoryViewerActivity.adapter
        }
        continueButton.setOnClickListener {
            session?.let {
                ChatRouter.startRun(this, it.modelId, null, it.sessionId, requireExistingModel = true)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        loadHistory()
    }

    private fun loadHistory() {
        loadJob?.cancel()
        session = null
        continueButton.isEnabled = false
        val sessionId = intent.getStringExtra(EXTRA_SESSION_ID)
        if (sessionId.isNullOrBlank()) {
            showUnavailable()
            return
        }
        loadJob = lifecycleScope.launch {
            val loaded = try {
                withContext(Dispatchers.IO) {
                    val manager = ChatDataManager.getInstance(applicationContext)
                    val storedSession = manager.allSessions.firstOrNull { it.sessionId == sessionId }
                    storedSession?.let {
                        val items = HistoryDisplayData.prepare(
                            manager.getChatDataBySession(sessionId),
                            getString(R.string.history_attachment_unavailable),
                            ::isAttachmentAvailable
                        )
                        it to items
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
                showUnavailable()
                return@launch
            }
            if (loaded == null) {
                showUnavailable()
                return@launch
            }
            session = loaded.first
            title = loaded.first.title.ifBlank { getString(R.string.history_title) }
            supportActionBar?.subtitle = loaded.first.modelId
            adapter.updateModelNameAndItems(loaded.first.modelId, loaded.second)
            status.setText(if (loaded.second.isEmpty()) R.string.no_chat_history else R.string.history_read_only)
            continueButton.isEnabled = true
        }
    }

    private fun showUnavailable() {
        adapter.updateModelNameAndItems("", mutableListOf())
        status.setText(R.string.history_session_unavailable)
        continueButton.isEnabled = false
    }

    private fun isAttachmentAvailable(uri: Uri): Boolean = try {
        when (uri.scheme) {
            "file" -> uri.path?.let { File(it).isFile && File(it).canRead() } == true
            "content" -> contentResolver.openAssetFileDescriptor(uri, "r")?.use { true } ?: false
            else -> false
        }
    } catch (_: Exception) {
        false
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    companion object {
        const val EXTRA_SESSION_ID = "historySessionId"

        fun start(context: Context, sessionId: String) {
            val intent = Intent(context, HistoryViewerActivity::class.java)
                .putExtra(EXTRA_SESSION_ID, sessionId)
            if (context !is Activity) intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
        }
    }
}
