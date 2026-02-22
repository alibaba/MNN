package com.alibaba.mnnllm.android.hotspot

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.model.ModelUtils
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ChatServerFragment : Fragment() {

    private lateinit var statusCard: View
    private lateinit var runningCard: View
    private lateinit var tvStatus: TextView
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var ivWifiQr: ImageView
    private lateinit var ivUrlQr: ImageView
    private lateinit var tvUrl: TextView
    private lateinit var tvUsers: TextView
    private lateinit var webView: WebView
    private lateinit var btnOpenBrowser: Button
    private lateinit var tvDebugPrompt: TextView
    private lateinit var tvDebugOutput: TextView
    private lateinit var debugPanel: View
    private lateinit var debugScroll: View
    private lateinit var tvQrWifiLabel: LabelCyclerView
    private lateinit var tvQrUrlLabel: LabelCyclerView

    private var hotspotManager: LocalHotspotManager? = null
    private var serverManager: ChatServerManager? = null
    private var hotspotJob: kotlinx.coroutines.Job? = null
    private var inferenceDebugJob: kotlinx.coroutines.Job? = null

    // Local loopback URL used for WebView / "open in browser"
    private var localLoopbackUrl: String? = null

    // Must be registered before onStart; property initialisation satisfies that requirement.
    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
            if (grants.values.all { it }) {
                pickModelAndStart()
            } else {
                Toast.makeText(
                    requireContext(),
                    R.string.chat_server_permission_denied,
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    // ── Service binding ────────────────────────────────────────────────────────
    private val serviceConnection = object : android.content.ServiceConnection {
        override fun onServiceConnected(name: android.content.ComponentName, service: IBinder) {
            serverManager = (service as ChatServerService.LocalBinder).getManager()
            updateUi()
        }
        override fun onServiceDisconnected(name: android.content.ComponentName) {
            serverManager = null
        }
    }
    private var serviceBound = false

    private val exportBridge = WebExportBridge(this) // For export

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View = inflater.inflate(R.layout.fragment_chat_server, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        statusCard = view.findViewById(R.id.card_status)
        runningCard = view.findViewById(R.id.card_running)
        tvStatus = view.findViewById(R.id.tv_status)
        btnStart = view.findViewById(R.id.btn_start_server)
        btnStop = view.findViewById(R.id.btn_stop_server)
        ivWifiQr = view.findViewById(R.id.iv_wifi_qr)
        ivUrlQr = view.findViewById(R.id.iv_url_qr)
        tvUrl = view.findViewById(R.id.tv_chat_url)
        tvUsers = view.findViewById(R.id.tv_users)
        webView = view.findViewById(R.id.webview_chat)
        btnOpenBrowser = view.findViewById(R.id.btn_open_browser)
        debugPanel = view.findViewById(R.id.debug_panel)
        tvDebugPrompt = view.findViewById(R.id.tv_debug_prompt)
        tvDebugOutput = view.findViewById(R.id.tv_debug_output)
        tvQrWifiLabel = view.findViewById(R.id.tv_qr_wifi_label)
        tvQrUrlLabel  = view.findViewById(R.id.tv_qr_url_label)

        debugScroll = view.findViewById(R.id.debug_scroll)
        debugScroll.isNestedScrollingEnabled = true
        debugScroll.setOnTouchListener { v, event ->
            v.parent.requestDisallowInterceptTouchEvent(true)
            false
        }

        btnStart.setOnClickListener { checkAndRequestHotspotPermissions() }
        btnStop.setOnClickListener { stopServer() }
        btnOpenBrowser.setOnClickListener {
            // Open the loopback URL in the default browser when available.
            val url = localLoopbackUrl ?: tvUrl.text.toString()
            if (url.isNotEmpty()) {
                startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
            }
        }

        setupWebView()
        updateUi()
    }

    // ── Permissions ────────────────────────────────────────────────────────────

    /**
     * Checks for the permission required by [WifiManager.startLocalOnlyHotspot]:
     *  - API 33+: NEARBY_WIFI_DEVICES
     *  - API 26–32: ACCESS_FINE_LOCATION
     *
     * Proceeds to [pickModelAndStart] immediately when already granted, otherwise
     * triggers the system permission dialog via [permissionLauncher].
     */
    private fun checkAndRequestHotspotPermissions() {
        val required = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(Manifest.permission.NEARBY_WIFI_DEVICES)
        } else {
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
        }
        val missing = required.filter {
            ContextCompat.checkSelfPermission(requireContext(), it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            pickModelAndStart()
        } else {
            permissionLauncher.launch(missing.toTypedArray())
        }
    }

    // ── WebView ────────────────────────────────────────────────────────────────

    private fun setupWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = true
            allowContentAccess = true
        }
        webView.webViewClient = WebViewClient()
        // Expose the native bridge to JS as "AndroidExport"
        webView.addJavascriptInterface(exportBridge, "AndroidExport")
    }

    // ── Start / stop ───────────────────────────────────────────────────────────

    private fun pickModelAndStart() {
        lifecycleScope.launch {
            val models = withContext(Dispatchers.IO) {
                val state = ModelListManager.observeModelList()
                    .filterIsInstance<ModelListManager.ModelListState.Success>()
                    .first()
                state.models.filter { it.downloadedModelInfo != null || it.isLocal }
            }

            if (!isAdded) return@launch

            if (models.isEmpty()) {
                Toast.makeText(requireContext(), R.string.chat_server_no_models, Toast.LENGTH_LONG).show()
                return@launch
            }

            val names = models.map { it.displayName }.toTypedArray()
            MaterialAlertDialogBuilder(requireContext())
                .setTitle(R.string.chat_server_pick_model)
                .setItems(names) { _, idx ->
                    val wrapper = models[idx]
                    val modelId = wrapper.modelItem.modelId ?: ""

                    if (modelId.isEmpty()) {
                        Toast.makeText(requireContext(), requireContext().getString(R.string.model_not_found, ""), Toast.LENGTH_LONG).show()
                        return@setItems
                    }

                    val configPath = ModelUtils.getConfigPathForModel(modelId)
                    if (configPath != null) {
                        startServer(modelId, configPath)
                    } else {
                        Toast.makeText(requireContext(), R.string.config_file_not_found, Toast.LENGTH_LONG).show()
                    }
                }
                .setNegativeButton(android.R.string.cancel, null)
                .show()
        }
    }

    private fun startServer(modelId: String, configPath: String) {
        tvStatus.setText(R.string.chat_server_starting)
        btnStart.isEnabled = false
        setBottomNavVisible(false)

        val intent = Intent(requireContext(), ChatServerService::class.java).apply {
            action = ChatServerService.ACTION_START
            putExtra(ChatServerService.EXTRA_MODEL_ID, modelId)
            putExtra(ChatServerService.EXTRA_CONFIG_PATH, configPath)
        }
        androidx.core.content.ContextCompat.startForegroundService(requireContext(), intent)
        requireContext().bindService(intent, serviceConnection, android.content.Context.BIND_AUTO_CREATE)
        serviceBound = true

        hotspotManager = LocalHotspotManager(requireContext())
        hotspotJob = lifecycleScope.launch(Dispatchers.IO) {
            hotspotManager!!.hotspotInfoFlow()
                .catch { e ->
                    withContext(Dispatchers.Main) {
                        if (isAdded) {
                            tvStatus.text = getString(R.string.chat_server_hotspot_failed, e.message)
                            btnStart.isEnabled = true
                        }
                    }
                }
                .collect { result ->
                    result.onSuccess { info ->
                        val connInfo = HotspotConnectionInfo(
                            ssid = info.ssid,
                            password = info.passphrase,
                            gatewayIp = info.gatewayIp,
                            port = CHAT_SERVER_PORT,
                        )
                        val wifiQr = withContext(Dispatchers.Default) { QrCodeGenerator.generate(connInfo.wifiQrContent) }
                        val urlQr  = withContext(Dispatchers.Default) { QrCodeGenerator.generate(connInfo.urlQrContent) }
                        withContext(Dispatchers.Main) {
                            if (isAdded) showRunningState(connInfo, wifiQr, urlQr)
                        }
                    }
                    result.onFailure { e ->
                        withContext(Dispatchers.Main) {
                            if (isAdded) {
                                tvStatus.text = getString(R.string.chat_server_hotspot_failed, e.message)
                                btnStart.isEnabled = true
                            }
                        }
                    }
                }
        }
    }

    private fun stopServer() {
        hotspotJob?.cancel()
        hotspotJob = null
        hotspotManager = null
        if (serviceBound) {
            requireContext().unbindService(serviceConnection)
            serviceBound = false
        }
        requireContext().startService(
            Intent(requireContext(), ChatServerService::class.java).apply {
                action = ChatServerService.ACTION_STOP
            }
        )
        serverManager = null
        setBottomNavVisible(true)
        if (isAdded) showStoppedState()
    }

    /** Show or hide MainActivity's bottom navigation bar. */
    private fun setBottomNavVisible(visible: Boolean) { //TODO: don't use reflection for this, hahaha... but I guess it's okay for now because I'd rather not edit Alibaba's code directly if I can help it.
        val activity = activity ?: return
        try {
            val field = activity.javaClass.getDeclaredField("bottomNav")
            field.isAccessible = true
            val nav = field.get(activity) as? android.view.View ?: return
            nav.visibility = if (visible) android.view.View.VISIBLE else android.view.View.GONE
        } catch (e: Exception) {
            // bottomNav field not found or not accessible - ignore
        }
    }

    // ── UI helpers ─────────────────────────────────────────────────────────────

    private fun showRunningState(info: HotspotConnectionInfo, wifiQr: Bitmap?, urlQr: Bitmap?) {
        tvStatus.setText(R.string.chat_server_running)
        statusCard.visibility = View.GONE
        runningCard.visibility = View.VISIBLE
        ivWifiQr.setImageBitmap(wifiQr)
        ivUrlQr.setImageBitmap(urlQr)

        // Keep QR and displayed URL as-is (these point to the hotspot IP).
        tvUrl.text = info.urlQrContent
        tvQrWifiLabel.setEntries(HotspotLabelCycler.buildWifiEntries(requireContext()))
        tvQrUrlLabel.setEntries(HotspotLabelCycler.buildUrlEntries(requireContext()))

        // Use loopback for the WebView and the "open in browser" action.
        // Keep port from the hotspot connection info.
        localLoopbackUrl = "http://127.0.0.1:${info.port}/"
        webView.loadUrl(localLoopbackUrl!!)

        observeConnectedCount()
        observeInferenceDebug()
    }

    private fun showStoppedState() {
        statusCard.visibility = View.VISIBLE
        runningCard.visibility = View.GONE
        btnStart.isEnabled = true
        tvStatus.setText(R.string.chat_server_idle)
        ivWifiQr.setImageBitmap(null)
        ivUrlQr.setImageBitmap(null)
        tvUrl.text = ""
        tvQrWifiLabel.setEntries(emptyList())
        tvQrUrlLabel.setEntries(emptyList())
        // Clear loopback URL when stopped so the browser button falls back to the displayed QR URL (if any).
        localLoopbackUrl = null
        webView.loadUrl("about:blank")
        debugPanel.visibility = View.GONE
    }

    private fun updateUi() {
        if (serverManager?.isRunning() == true) {
            statusCard.visibility = View.GONE
            runningCard.visibility = View.VISIBLE
            observeConnectedCount()
        } else {
            showStoppedState()
        }
    }

    private fun observeConnectedCount() {
        val mgr = serverManager ?: return
        lifecycleScope.launch {
            mgr.connectedCountFlow.collect { count ->
                if (isAdded) {
                    tvUsers.text = resources.getQuantityString(
                        R.plurals.chat_server_user_count, count, count
                    )
                }
            }
        }
    }

    private fun updateUserCount() {
        val count = serverManager?.getConnectedUserCount() ?: 0
        tvUsers.text = resources.getQuantityString(R.plurals.chat_server_user_count, count, count)
    }

    private fun observeInferenceDebug() {
        if (inferenceDebugJob?.isActive == true) return   // already collecting
        val mgr = serverManager ?: return
        inferenceDebugJob = lifecycleScope.launch {
            mgr.inferenceDebugFlow.collect { state ->
                if (!isAdded) return@collect
                debugPanel.visibility = View.VISIBLE
                tvDebugPrompt.text = state.prompt
                tvDebugOutput.text = state.partialOutput
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        if (serviceBound) {
            requireContext().unbindService(serviceConnection)
            serviceBound = false
        }
        webView.destroy()
    }
}