package com.alibaba.mnnllm.api.openai.ui

import android.app.Dialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import androidx.recyclerview.widget.LinearLayoutManager
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.Toast
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Job
import com.alibaba.mnnllm.android.databinding.FragmentApiConsoleSheetBinding
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import com.alibaba.mnnllm.api.openai.manager.ServerEventManager
import com.alibaba.mnnllm.api.openai.network.logging.LogCollector
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import timber.log.Timber
import java.text.SimpleDateFormat
import java.util.*

class ApiConsoleBottomSheetFragment : BottomSheetDialogFragment() {

    private var _binding: FragmentApiConsoleSheetBinding? = null
    private val binding get() = _binding!!

    private val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val logAdapter = LogAdapter()
    private val serverEventManager = ServerEventManager.getInstance()
    private val logCollector = LogCollector.getInstance()

    // 管理协程订阅
    private var serverStateJob: Job? = null
    private var serverInfoJob: Job? = null
    private var logCollectorJob: Job? = null

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentApiConsoleSheetBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onStart() {
        super.onStart()
        val dialog: Dialog? = dialog
        if (dialog != null) {
            val bottomSheet: FrameLayout? = dialog.findViewById(com.google.android.material.R.id.design_bottom_sheet)
            if (bottomSheet != null) {
                val behavior = BottomSheetBehavior.from(bottomSheet)
                bottomSheet.post {
                    behavior.state = BottomSheetBehavior.STATE_EXPANDED
                }
                behavior.skipCollapsed = false
            }
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        Timber.tag("ApiConsoleUI").d("[UI] Initializing view components")
        setupServiceStatus()
        setupConfigSummary()
        setupLogArea()
        setupActionButtons()
        observeServerEvents()
    }

    override fun onResume() {
        super.onResume()
        // 重新订阅事件，确保状态监听正常工作
        //observeServerEvents()


        // 每次Fragment可见时刷新状态，确保显示最新的服务器状态
       // updateServiceStatus()

        // 延迟再次检查状态，确保服务重启后能正确获取状态
        //binding.root.postDelayed({
           // if (isAdded && !isDetached) {
          //     updateServiceStatus()
           // }
       // }, 1000)

    }

    private fun setupServiceStatus() {
        updateServiceStatus()
    }

    private fun updateServiceStatus() {
        val context = requireContext()
        val isServiceEnabled = MainSettings.isApiServiceEnabled(context)
        val serverState = serverEventManager.getCurrentState()
        val serverInfo = serverEventManager.getCurrentInfo()

        // 获取配置的IP和端口，用于显示API端点
        val configuredHost = ApiServerConfig.getIpAddress(context)
        val configuredPort = ApiServerConfig.getPort(context)

        Timber.tag("ApiConsoleUI").d("[Service] Updating service status - enabled: $isServiceEnabled, state: $serverState, info: $serverInfo")

        when {
            !isServiceEnabled -> {
                Timber.tag("ApiConsoleUI").d("[Service] Service is disabled")
                binding.textServiceStatus.text = "已禁用"
                binding.textServiceStatus.setTextColor(resources.getColor(android.R.color.darker_gray, null))
                binding.textListenAddress.visibility = View.GONE
                binding.labelListenAddress.visibility = View.GONE
                binding.textApiEndpoint.visibility = View.GONE
                binding.labelApiEndpoint.visibility = View.GONE
            }
            serverState == ServerEventManager.ServerState.READY -> {
                Timber.tag("ApiConsoleUI").d("[Service] Server is ready")
                binding.textServiceStatus.text = "运行中"
                binding.textServiceStatus.setTextColor(resources.getColor(android.R.color.holo_green_dark, null))
                binding.textListenAddress.visibility = View.GONE
                binding.labelListenAddress.visibility = View.GONE
                // 使用实际运行的服务器信息，如果为空则使用配置信息
                val displayHost = if (serverInfo.host.isNotEmpty()) serverInfo.host else configuredHost
                val displayPort = if (serverInfo.port > 0) serverInfo.port else configuredPort
                val endpointUrl = "http://${displayHost}:${displayPort}/v1/chat/completions"
                binding.textApiEndpoint.text = endpointUrl
                binding.textApiEndpoint.visibility = View.VISIBLE
                binding.labelApiEndpoint.visibility = View.VISIBLE
                Timber.tag("ApiConsoleUI").d("[Service] API endpoint displayed: $endpointUrl")
            }
            serverState in listOf(ServerEventManager.ServerState.STARTING, ServerEventManager.ServerState.STARTED) -> {
                Timber.tag("ApiConsoleUI").d("[Service] Server is starting")
                binding.textServiceStatus.text = "启动中"
                binding.textServiceStatus.setTextColor(resources.getColor(android.R.color.holo_orange_dark, null))
                binding.textListenAddress.visibility = View.GONE
                binding.labelListenAddress.visibility = View.GONE
                binding.textApiEndpoint.visibility = View.GONE
                binding.labelApiEndpoint.visibility = View.GONE
            }
            serverState in listOf(ServerEventManager.ServerState.STOP_PREPARING, ServerEventManager.ServerState.STOPPING) -> {
                Timber.tag("ApiConsoleUI").d("[Service] Server is stopping")
                binding.textServiceStatus.text = "停止中"
                binding.textServiceStatus.setTextColor(resources.getColor(android.R.color.holo_orange_dark, null))
                binding.textListenAddress.visibility = View.GONE
                binding.labelListenAddress.visibility = View.GONE
                binding.textApiEndpoint.visibility = View.GONE
                binding.labelApiEndpoint.visibility = View.GONE
            }
            else -> {
                Timber.tag("ApiConsoleUI").d("[Service] Server is stopped")
                binding.textServiceStatus.text = "已停止"
                binding.textServiceStatus.setTextColor(resources.getColor(android.R.color.holo_red_dark, null))
                binding.textListenAddress.visibility = View.GONE
                binding.labelListenAddress.visibility = View.GONE
                binding.textApiEndpoint.visibility = View.GONE
                binding.labelApiEndpoint.visibility = View.GONE
            }
        }
    }

    private fun setupConfigSummary() {
        val context = requireContext()

        val corsEnabled = ApiServerConfig.isCorsEnabled(context)
        val authEnabled = ApiServerConfig.isAuthEnabled(context)

        Timber.tag("ApiConsoleUI").d("[Config] CORS enabled: $corsEnabled, Auth enabled: $authEnabled")
        
        binding.textCorsStatus.text = if (corsEnabled) "已启用" else "已禁用"
        binding.textAuthStatus.text = if (authEnabled) "API Key已开启" else "无认证"

        // 设置折叠/展开功能
        binding.layoutConfigHeader.setOnClickListener {
            val isVisible = binding.layoutConfigDetails.isVisible
            binding.layoutConfigDetails.isVisible = !isVisible
            binding.iconConfigExpand.rotation = if (isVisible) 0f else 180f
            Timber.tag("ApiConsoleUI").d("[Config] Header clicked, visibility: ${!isVisible}")
        }
    }

    private fun setupLogArea() {
        // 设置RecyclerView
        binding.recyclerLogContent.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = logAdapter
            // 设置触摸事件拦截，防止滑动冲突
            setOnTouchListener { view, event ->
                // 请求父容器不要拦截触摸事件
                view.parent.requestDisallowInterceptTouchEvent(true)
                false
            }
        }

        Timber.tag("ApiConsoleUI").d("[Log] Initializing log area")
        // 添加初始日志
        addLogMessage("API服务控制台已启动")

        val serverState = serverEventManager.getCurrentState()
        when (serverState) {
            ServerEventManager.ServerState.READY -> {
                addLogMessage("服务器正在运行中")
                addLogMessage("等待客户端连接...")
            }
            ServerEventManager.ServerState.STOPPED -> {
                addLogMessage("服务器未启动")
            }
            else -> {
                addLogMessage("服务器状态: ${serverState.name}")
            }
        }

        // 订阅实时日志
        logCollector.logFlow
            .onEach { logEntry ->
                if (isAdded && !isDetached) {
                    val (formattedLog, clickableInfo) = logCollector.formatLogEntryWithClickableInfo(logEntry)
                    addRawLogEntryWithClickInfo(formattedLog, clickableInfo)
                }
            }
            .launchIn(lifecycleScope)

        // 设置折叠/展开功能
        binding.layoutLogHeader.setOnClickListener {
            val isVisible = binding.layoutLogContent.isVisible
            binding.layoutLogContent.isVisible = !isVisible
            binding.iconLogExpand.rotation = if (isVisible) 0f else 180f
            Timber.tag("ApiConsoleUI").d("[Log] Log header clicked, visibility: ${!isVisible}")
        }
    }

    private fun addLogMessage(message: String) {
        val timestamp = dateFormat.format(Date())
        val logEntry = "[$timestamp] $message"
        logAdapter.addLogMessage(logEntry)

        // 自动滚动到底部
        scrollToBottom()
    }

    private fun addRawLogMessage(message: String) {
        logAdapter.addLogMessage(message)

        // 自动滚动到底部
        scrollToBottom()
    }
    
    private fun addRawLogEntryWithClickInfo(message: String, clickableInfo: String?) {
        val logEntry = LogAdapter.LogEntryData(message, clickableInfo)
        logAdapter.addLogEntry(logEntry)

        // 自动滚动到底部
        scrollToBottom()
    }

    private fun scrollToBottom() {
        binding.recyclerLogContent.post {
            if (logAdapter.itemCount > 0) {
                binding.recyclerLogContent.smoothScrollToPosition(logAdapter.itemCount - 1)
            }
        }
    }



    private fun setupActionButtons() {
        binding.buttonClose.setOnClickListener {
            dismiss()
        }

        binding.buttonClearLog.setOnClickListener {
            clearLog()
        }

        binding.buttonCopyLog.setOnClickListener {
            copyLogToClipboard()
        }
        
        // 添加测试按钮（长按清空日志按钮）
        binding.buttonClearLog.setOnLongClickListener {
            addTestLogWithCodeLocation()
            true
        }
    }
    
    /**
     * 添加测试日志，包含代码行号信息
     */
    private fun addTestLogWithCodeLocation() {
        Timber.tag("TestLog").i("这是一条测试日志，用于验证代码行号显示和点击跳转功能")
        Timber.tag("TestLog").d("调试信息：代码行号功能测试")
        Timber.tag("TestLog").w("警告信息：请确保Android Studio已打开项目")
        addLogMessage("已添加测试日志，包含代码行号信息")
        Toast.makeText(requireContext(), "已添加测试日志\n长按日志条目可尝试跳转到代码", Toast.LENGTH_LONG).show()
    }

    private fun clearLog() {
        logAdapter.clearLogs()
        Toast.makeText(requireContext(), "日志已清空", Toast.LENGTH_SHORT).show()
    }

    private fun copyLogToClipboard() {
        val logText = logAdapter.getAllLogs().joinToString("\n")
        val clipboard = requireContext().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("API Console Log", logText)
        clipboard.setPrimaryClip(clip)
        Toast.makeText(requireContext(), "日志已复制到剪贴板", Toast.LENGTH_SHORT).show()
    }

    private fun observeServerEvents() {
        // 取消之前的订阅
        cancelObservations()
        Timber.tag("ApiConsoleUI").d("[ServerEvent] Subscribing to server events")

        // 立即获取当前状态并更新UI
        updateServiceStatus()

        // 观察服务器状态变化
        serverStateJob = serverEventManager.serverState
            .onEach { state ->
                if (isAdded && !isDetached) {
                    // 强制更新UI状态
                    updateServiceStatus()

                    // 根据状态变化添加日志
                    when (state) {
                        ServerEventManager.ServerState.STARTING -> {
                            addLogMessage("服务器启动中...")
                        }
                        ServerEventManager.ServerState.STARTED -> {
                            addLogMessage("服务器已启动")
                        }
                        ServerEventManager.ServerState.READY -> {
                            addLogMessage("服务器就绪，可接受连接")
                        }
                        ServerEventManager.ServerState.STOP_PREPARING -> {
                            addLogMessage("服务器准备停止...")
                        }
                        ServerEventManager.ServerState.STOPPING -> {
                            addLogMessage("服务器停止中...")
                        }
                        ServerEventManager.ServerState.STOPPED -> {
                            addLogMessage("服务器已停止")
                        }
                    }
                }
            }
            .launchIn(lifecycleScope)

        // 观察服务器信息变化
        serverInfoJob = serverEventManager.serverInfo
            .onEach { info ->
                if (isAdded && !isDetached) {
                    // 当服务器信息变化时强制更新状态
                    updateServiceStatus()

                    if (info.isRunning) {
                        addLogMessage("服务器运行在 ${info.host}:${info.port}")
                    }
                }
            }
            .launchIn(lifecycleScope)
    }

    private fun cancelObservations() {
        serverStateJob?.cancel()
        serverInfoJob?.cancel()
        logCollectorJob?.cancel()
        Timber.tag("ApiConsoleUI").d("[Subscription] All subscriptions cancelled")
        serverStateJob = null
        serverInfoJob = null
        logCollectorJob = null
    }

    override fun onDestroyView() {
        cancelObservations()
        super.onDestroyView()
        _binding = null
    }

    companion object {
        const val TAG = "ApiConsoleBottomSheetFragment"
    }
}