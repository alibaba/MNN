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
import com.alibaba.mnnllm.android.R
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
import com.alibaba.mnnllm.android.chat.ChatActivity

class ApiConsoleBottomSheetFragment : BottomSheetDialogFragment() {

    private var _binding: FragmentApiConsoleSheetBinding? = null
    private val binding get() = _binding!!
    private var chatActivity: ChatActivity? = null

    companion object {
        const val TAG = "ApiConsoleBottomSheetFragment"
        fun newInstance(chatActivity: ChatActivity): ApiConsoleBottomSheetFragment {
            return ApiConsoleBottomSheetFragment().apply {
                this.chatActivity = chatActivity
            }
        }
    }

    private val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val logAdapter = LogAdapter()
    private val serverEventManager = ServerEventManager.getInstance()
    private val logCollector = LogCollector.getInstance()

    // Manage coroutine subscriptions
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
                
                // Optimize touch event handling to reduce conflicts with ScrollView
                behavior.isDraggable = true
                behavior.isHideable = false
                
                // Set up touch event listener to optimize scrolling experience
                setupBottomSheetTouchHandling(bottomSheet, behavior)
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
        
        // Resolve scrolling conflicts between ScrollView and BottomSheetDialog
        setupScrollViewTouchHandling()
    }

    override fun onResume() {
        super.onResume()
        // Re-subscribe to events to ensure status monitoring works correctly
        //observeServerEvents()


        // Refresh status every time the fragment is visible to ensure the latest server status is displayed
       // updateServiceStatus()

        // Check status again after a delay to ensure correct status is obtained after service restart
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
        val context = chatActivity ?: requireContext()
        val isServiceEnabled = MainSettings.isApiServiceEnabled(context)
        val serverState = serverEventManager.getCurrentState()
        val serverInfo = serverEventManager.getCurrentInfo()

        // Get configured IP and port to display the API endpoint
        val configuredHost = ApiServerConfig.getIpAddress(context)
        val configuredPort = ApiServerConfig.getPort(context)

        Timber.tag("ApiConsoleUI").d("[Service] Updating service status - enabled: $isServiceEnabled, state: $serverState, info: $serverInfo")

        when {
            !isServiceEnabled -> {
                Timber.tag("ApiConsoleUI").d("[Service] Service is disabled")
                binding.textServiceStatus.text = getString(R.string.service_disabled)
                binding.textServiceStatus.setTextColor(resources.getColor(android.R.color.darker_gray, null))
                binding.textListenAddress.visibility = View.GONE
                binding.labelListenAddress.visibility = View.GONE
                binding.textApiEndpoint.visibility = View.GONE
                binding.labelApiEndpoint.visibility = View.GONE
            }
            serverState == ServerEventManager.ServerState.READY -> {
                Timber.tag("ApiConsoleUI").d("[Service] Server is ready")
                binding.textServiceStatus.text = getString(R.string.service_running)
                binding.textServiceStatus.setTextColor(resources.getColor(android.R.color.holo_green_dark, null))
                binding.textListenAddress.visibility = View.GONE
                binding.labelListenAddress.visibility = View.GONE
                // Use actual running server info, otherwise use configured info
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
                binding.textServiceStatus.text = getString(R.string.service_starting)
                binding.textServiceStatus.setTextColor(resources.getColor(android.R.color.holo_orange_dark, null))
                binding.textListenAddress.visibility = View.GONE
                binding.labelListenAddress.visibility = View.GONE
                binding.textApiEndpoint.visibility = View.GONE
                binding.labelApiEndpoint.visibility = View.GONE
            }
            serverState in listOf(ServerEventManager.ServerState.STOP_PREPARING, ServerEventManager.ServerState.STOPPING) -> {
                Timber.tag("ApiConsoleUI").d("[Service] Server is stopping")
                binding.textServiceStatus.text = getString(R.string.service_stopping)
                binding.textServiceStatus.setTextColor(resources.getColor(android.R.color.holo_orange_dark, null))
                binding.textListenAddress.visibility = View.GONE
                binding.labelListenAddress.visibility = View.GONE
                binding.textApiEndpoint.visibility = View.GONE
                binding.labelApiEndpoint.visibility = View.GONE
            }
            else -> {
                Timber.tag("ApiConsoleUI").d("[Service] Server is stopped")
                binding.textServiceStatus.text = getString(R.string.service_stopped)
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
        
        binding.textCorsStatus.text = if (corsEnabled) getString(R.string.cors_enabled) else getString(R.string.cors_disabled_status)
        binding.textAuthStatus.text = if (authEnabled) getString(R.string.api_key_enabled) else getString(R.string.no_authentication)

        // Set up collapse/expand functionality
        binding.layoutConfigHeader.setOnClickListener {
            val isVisible = binding.layoutConfigDetails.isVisible
            binding.layoutConfigDetails.isVisible = !isVisible
            binding.iconConfigExpand.rotation = if (isVisible) 0f else 180f
            Timber.tag("ApiConsoleUI").d("[Config] Header clicked, visibility: ${!isVisible}")
        }
    }

    private fun setupLogArea() {
        // Set up RecyclerView
        binding.recyclerLogContent.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = logAdapter
            // Intercept touch events to prevent scrolling conflicts
            setOnTouchListener {
                view, event ->
                // Request parent container not to intercept touch events
                view.parent.requestDisallowInterceptTouchEvent(true)
                false
            }
        }

        Timber.tag("ApiConsoleUI").d("[Log] Initializing log area")
        // Add initial log message
        addLogMessage(getString(R.string.console_started))

        val serverState = serverEventManager.getCurrentState()
        when (serverState) {
            ServerEventManager.ServerState.READY -> {
                addLogMessage(getString(R.string.server_running_message))
                addLogMessage(getString(R.string.waiting_for_connections))
            }
            ServerEventManager.ServerState.STOPPED -> {
                addLogMessage(getString(R.string.server_not_started))
            }
            else -> {
                addLogMessage(getString(R.string.server_status_template, serverState.name))
            }
        }

        // Subscribe to real-time logs
        logCollector.logFlow
            .onEach {
                logEntry ->
                if (isAdded && !isDetached) {
                    val (formattedLog, clickableInfo) = logCollector.formatLogEntryWithClickableInfo(logEntry)
                    addRawLogEntryWithClickInfo(formattedLog, clickableInfo)
                }
            }
            .launchIn(lifecycleScope)

        // Set up collapse/expand functionality
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

        // Auto-scroll to the bottom
        scrollToBottom()
    }

    private fun addRawLogMessage(message: String) {
        logAdapter.addLogMessage(message)

        // Auto-scroll to the bottom
        scrollToBottom()
    }
    
    private fun addRawLogEntryWithClickInfo(message: String, clickableInfo: String?) {
        val logEntry = LogAdapter.LogEntryData(message, clickableInfo)
        logAdapter.addLogEntry(logEntry)

        // Auto-scroll to the bottom
        scrollToBottom()
    }

    private fun scrollToBottom() {
        binding.recyclerLogContent.post {
            if (logAdapter.itemCount > 0) {
                binding.recyclerLogContent.smoothScrollToPosition(logAdapter.itemCount - 1)
            }
        }
    }



    private fun setupScrollViewTouchHandling() {
        // Set up touch event handling for the ScrollView to resolve scrolling conflicts with the BottomSheetDialog
        binding.settingsScrollView.setOnTouchListener {
            view, event ->
            when (event.action) {
                android.view.MotionEvent.ACTION_DOWN -> {
                    // When touch starts, request parent container not to intercept touch events
                    view.parent.requestDisallowInterceptTouchEvent(true)
                }
                android.view.MotionEvent.ACTION_MOVE -> {
                    // Check if scrolling is needed
                    val scrollView = view as androidx.core.widget.NestedScrollView
                    if (scrollView.canScrollVertically(-1) || scrollView.canScrollVertically(1)) {
                        // If it can scroll, continue to request no interception
                        view.parent.requestDisallowInterceptTouchEvent(true)
                    }
                }
                android.view.MotionEvent.ACTION_UP, android.view.MotionEvent.ACTION_CANCEL -> {
                    // When touch ends, allow parent container to intercept touch events
                    view.parent.requestDisallowInterceptTouchEvent(false)
                }
            }
            false // Return false to let the ScrollView continue handling the touch event
        }
    }

    private fun setupBottomSheetTouchHandling(bottomSheet: FrameLayout, behavior: BottomSheetBehavior<FrameLayout>) {
        // Set up touch event handling for the BottomSheet to optimize interaction with the ScrollView
        bottomSheet.setOnTouchListener {
            view, event ->
            when (event.action) {
                android.view.MotionEvent.ACTION_DOWN -> {
                    // When touch starts, check the touch position
                    val scrollView = binding.settingsScrollView
                    if (scrollView.canScrollVertically(-1) || scrollView.canScrollVertically(1)) {
                        // If the ScrollView can scroll, let the ScrollView handle the touch event
                        scrollView.requestDisallowInterceptTouchEvent(true)
                    }
                }
                android.view.MotionEvent.ACTION_MOVE -> {
                    // When moving, if the ScrollView is scrolling, let it continue to handle it
                    val scrollView = binding.settingsScrollView
                    if (scrollView.canScrollVertically(-1) || scrollView.canScrollVertically(1)) {
                        scrollView.requestDisallowInterceptTouchEvent(true)
                    }
                }
            }
            false // Return false to let the BottomSheet continue handling the touch event
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
        
        // Add test button (long press to clear log)
        binding.buttonClearLog.setOnLongClickListener {
            addTestLogWithCodeLocation()
            true
        }
    }
    
    /**
     * Add test log, including code line number information
     */
    private fun addTestLogWithCodeLocation() {
        Timber.tag("TestLog").i(getString(R.string.test_log_message1))
        Timber.tag("TestLog").d(getString(R.string.test_log_message2))
        Timber.tag("TestLog").w(getString(R.string.test_log_message3))
        addLogMessage(getString(R.string.test_log_added_info))
        
        Toast.makeText(requireContext(), getString(R.string.test_log_added_with_code_jump), Toast.LENGTH_LONG).show()
    }

    private fun clearLog() {
        logAdapter.clearLogs()
        Toast.makeText(requireContext(), getString(R.string.log_cleared), Toast.LENGTH_SHORT).show()
    }

    private fun copyLogToClipboard() {
        val logText = logAdapter.getAllLogs().joinToString("\n")
        val clipboard = requireContext().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("API Console Log", logText)
        clipboard.setPrimaryClip(clip)
        Toast.makeText(requireContext(), getString(R.string.log_copied_to_clipboard), Toast.LENGTH_SHORT).show()
    }

    private fun observeServerEvents() {
        // Cancel previous subscriptions
        cancelObservations()
        Timber.tag("ApiConsoleUI").d("[ServerEvent] Subscribing to server events")

        // Immediately get the current status and update the UI
        updateServiceStatus()

        // Observe server status changes
        serverStateJob = serverEventManager.serverState
            .onEach {
                state ->
                if (isAdded && !isDetached) {
                    // Force update UI status
                    updateServiceStatus()

                    // Add log based on status change
                    when (state) {
                        ServerEventManager.ServerState.STARTING -> {
                            addLogMessage(getString(R.string.server_starting_message))
                        }
                        ServerEventManager.ServerState.STARTED -> {
                            addLogMessage(getString(R.string.server_started_message))
                        }
                        ServerEventManager.ServerState.READY -> {
                            addLogMessage(getString(R.string.server_ready_message))
                        }
                        ServerEventManager.ServerState.STOP_PREPARING -> {
                            addLogMessage(getString(R.string.server_preparing_stop))
                        }
                        ServerEventManager.ServerState.STOPPING -> {
                            addLogMessage(getString(R.string.server_stopping_message))
                        }
                        ServerEventManager.ServerState.STOPPED -> {
                            addLogMessage(getString(R.string.server_stopped_message))
                        }
                    }
                }
            }
            .launchIn(lifecycleScope)

        // Observe server info changes
        serverInfoJob = serverEventManager.serverInfo
            .onEach {
                info ->
                if (isAdded && !isDetached) {
                    // Force update status when server info changes
                    updateServiceStatus()

                    if (info.isRunning) {
                        addLogMessage(getString(R.string.server_running_at_template, info.host, info.port))
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

}