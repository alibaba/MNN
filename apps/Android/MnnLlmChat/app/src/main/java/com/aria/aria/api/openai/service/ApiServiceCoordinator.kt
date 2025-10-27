package com.alibaba.mnnllm.api.openai.service

import android.content.Context
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.manager.ApiNotificationManager
import com.alibaba.mnnllm.api.openai.network.application.OpenAIApplication
import com.alibaba.mnnllm.android.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import timber.log.Timber

/** * unifiedschedulingmanager,responsible for coordinatingnotificationbarserviceandservicelifecycle*/
class ApiServiceCoordinator(private val context: Context) {
    private val TAG = this::class.java.simpleName
    private val networkServiceScope = CoroutineScope(Dispatchers.IO)
    private var notificationManager: ApiNotificationManager? = null
    private var application: OpenAIApplication? = null

    private var _isInitialized = false
    val isInitialized: Boolean get() = _isInitialized

    private var _isServerRunning = false
    val isServerRunning: Boolean get() = _isServerRunning

    /** * initializecoordinator*/
    fun initialize(): Boolean {
        return runCatching {
            if (_isInitialized) {
                Timber.Forest.tag(TAG).w("Coordinator already initialized")
                return true
            }

            //initializenotificationmanager
            notificationManager = ApiNotificationManager(context)

            _isInitialized = true
            Timber.Forest.tag(TAG).i("Coordinator initialized successfully")
            true
        }.getOrElse { e ->
            Timber.Forest.tag(TAG).e(e, "Failed to initialize coordinator")
            false
        }
    }

    /** * startserviceandnotification*/
    fun startServer(): Boolean {
        if (!_isInitialized) {
            Timber.Forest.tag(TAG).w("Coordinator not initialized")
            return false
        }

        return runCatching {
            val session = ServiceLocator.getChatSessionProvider().getLlmSession()
            if (session == null) {
                Timber.Forest.tag(TAG).w("No active LlmSession found")
                notificationManager?.updateNotification(
                    context.getString(R.string.api_service_not_started),
                    context.getString(R.string.no_active_session)
                )
                return false
            }

            //ensureServerEventManagerstate correctlyinitialize
            //if previouslybeenreset,neededre-preparestate
            val serverEventManager = com.alibaba.mnnllm.api.openai.manager.ServerEventManager.getInstance()
            if (serverEventManager.getCurrentState() == com.alibaba.mnnllm.api.openai.manager.ServerEventManager.ServerState.STOPPED) {
                Timber.Forest.tag(TAG).d("ServerEventManager is in STOPPED state, ready for new server")
            }

            //startservice
            val app = OpenAIApplication(networkServiceScope, context)
            app.start()
            application = app

            //updatenotification
            notificationManager?.updateNotification(
                context.getString(R.string.api_service_running),
                "", //let NotificationManager usedefault IP addressdisplay
                app.getPort()
            )

            _isServerRunning = true
            Timber.Forest.tag(TAG).i("Server started successfully on port ${app.getPort()}")
            true
        }.getOrElse { e ->
            Timber.Forest.tag(TAG).e(e, "Failed to start server: ${e.message}")
            notificationManager?.updateNotification(
                context.getString(R.string.api_service_start_failed),
                context.getString(R.string.api_service_error, e.message)
            )
            false
        }
    }

    /** * stop serverandnotification*/
    fun stopServer() {
        runCatching {
            //stop server
            application?.stop()
            application = null

            com.alibaba.mnnllm.api.openai.manager.ServerEventManager.getInstance().resetRuntimeState()
            Timber.Forest.tag(TAG).d("ServerEventManager state reset after application is nullified.")

            //cancelnotification
            notificationManager?.cancelNotification()

            _isServerRunning = false
            Timber.Forest.tag(TAG).i("Server stopped successfully")
        }.onFailure { e ->
            Timber.Forest.tag(TAG).e(e, "Error stopping server: ${e.message}")
        }
    }

    /**
     * updatenotificationcontent*/
    fun updateNotification(title: String, content: String, port: Int = 8080) {
        notificationManager?.updateNotification(title, content, port)
    }

    /** * getnotificationobject（forforegroundservice）*/
    fun getNotification(
        title: String = context.getString(R.string.api_service_running),
        content: String = context.getString(R.string.api_service_port, 8080),
        port: Int = 8080
    ) = notificationManager?.buildNotification(title, content, port)

    /** * getserviceport*/
    fun getServerPort(): Int? = application?.getPort()

    /** * checkservicewhetherrunning*/
    fun checkServerStatus(): Boolean = application?.isRunning() ?: false

    /** * checkservicewhetherready*/
    fun checkServerReady(): Boolean = application?.isReady() ?: false

    /** * getservicestate*/
    fun getServerState() = application?.getServerState()

    /** * getserviceinfo*/
    fun getServerInfo() = application?.getServerInfo()

    /**
     * cleanupresource*/
    fun cleanup() {
        runCatching {
            val appToStop = application
            if (appToStop != null) {
                Timber.Forest.tag(TAG).i("Cleanup: Requesting server stop for application: $appToStop")
                appToStop.stop() //issuestoprequest
                
                Timber.Forest.tag(TAG).i("Cleanup: Waiting 5 seconds for server to stop gracefully...")
                try {
                    Thread.sleep(3000) //increasewaittimeto 5 seconds
                } catch (e: InterruptedException) {
                    Thread.currentThread().interrupt()
                    Timber.Forest.tag(TAG).w("Cleanup delay interrupted after server stop request.")
                }
                Timber.Forest.tag(TAG).i("Cleanup: Finished waiting. Proceeding with coordinator cleanup.")
                application = null //atwaitafter set to null
            }

            //ensureeven if appToStop as null，alsotryresetstateandcancelnotification
            com.alibaba.mnnllm.api.openai.manager.ServerEventManager.getInstance().resetRuntimeState()
            Timber.Forest.tag(TAG).d("ServerEventManager state reset during cleanup.")
            notificationManager?.cancelNotification()
            
            Timber.Forest.tag(TAG).i("Cleanup: Cancelling networkServiceScope.")
            networkServiceScope.cancel() //finallycancelscope
            notificationManager = null
            _isInitialized = false
            Timber.Forest.tag(TAG).i("Coordinator cleaned up")
        }.onFailure { e ->
            Timber.Forest.tag(TAG).e(e, "Error during cleanup: ${e.message}")
        }
    }
}