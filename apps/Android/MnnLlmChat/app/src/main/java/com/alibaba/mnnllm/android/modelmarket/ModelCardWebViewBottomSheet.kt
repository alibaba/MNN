package com.alibaba.mnnllm.android.modelmarket

import android.annotation.SuppressLint
import android.app.Dialog
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.webkit.WebViewClient
import android.webkit.WebChromeClient
import android.widget.ImageButton
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.widget.PopupMenu
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.BaseBottomSheetDialogFragment
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog

class ModelCardWebViewBottomSheet : BaseBottomSheetDialogFragment() {
    private var url: String? = null
    private var title: String? = null
    private var webView: WebView? = null
    private var progressBar: ProgressBar? = null
    private var btnMenu: ImageButton? = null
    private var btnClose: ImageButton? = null
    private var bottomSheetBehavior: BottomSheetBehavior<View>? = null

    companion object {
        private const val ARG_URL = "arg_url"
        private const val ARG_TITLE = "arg_title"
        private const val TAG = "ModelCardWebView"

        fun newInstance(url: String, title: String): ModelCardWebViewBottomSheet {
            val fragment = ModelCardWebViewBottomSheet()
            val args = Bundle()
            args.putString(ARG_URL, url)
            args.putString(ARG_TITLE, title)
            fragment.arguments = args
            return fragment
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            url = it.getString(ARG_URL)
            title = it.getString(ARG_TITLE)
        }
    }

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val dialog = super.onCreateDialog(savedInstanceState) as BottomSheetDialog
        dialog.setOnShowListener { dialogInterface ->
            val bottomSheetDialog = dialogInterface as BottomSheetDialog
            val bottomSheet = bottomSheetDialog.findViewById<View>(com.google.android.material.R.id.design_bottom_sheet)
            bottomSheet?.let {
                bottomSheetBehavior = BottomSheetBehavior.from(it)
                bottomSheetBehavior?.apply {
                    state = BottomSheetBehavior.STATE_EXPANDED
                    isDraggable = true
                }
                
                // Set the height to 80% of screen height
                val displayMetrics = resources.displayMetrics
                val height = (displayMetrics.heightPixels * 0.8).toInt()
                it.layoutParams?.height = height
                it.requestLayout()
            }
        }
        return dialog
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.dialog_fragment_model_card_webview, container, false)
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        val tvTitle = view.findViewById<TextView>(R.id.tvModelCardTitle)
        webView = view.findViewById<WebView>(R.id.webViewModelCard)
        progressBar = view.findViewById<ProgressBar>(R.id.progressBar)
        btnMenu = view.findViewById<ImageButton>(R.id.btnMenu)
        btnClose = view.findViewById<ImageButton>(R.id.btnClose)
        
        tvTitle.text = title ?: getString(R.string.model_card)
        
        setupWebView()
        setupMenuButton()
        
        url?.let { 
            Log.d(TAG, "Loading URL: $it")
            webView?.loadUrl(it) 
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView?.let { webView ->
            // WebView settings
            webView.settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                loadWithOverviewMode = true
                useWideViewPort = true
                builtInZoomControls = true
                displayZoomControls = false
                setSupportZoom(true)
                defaultTextEncodingName = "utf-8"
                
                // Enable mixed content for HTTPS sites that load HTTP resources
                mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            }

            // Handle scroll to control BottomSheet behavior
            webView.setOnScrollChangeListener { _, _, scrollY, _, _ ->
                // Allow BottomSheet to be dragged down only when WebView is at the top
                bottomSheetBehavior?.isDraggable = scrollY == 0
            }

            // WebViewClient for handling page loading
            webView.webViewClient = object : WebViewClient() {
                override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                    super.onPageStarted(view, url, favicon)
                    Log.d(TAG, "Page started loading: $url")
                    progressBar?.visibility = View.VISIBLE
                }

                override fun onPageFinished(view: WebView?, url: String?) {
                    super.onPageFinished(view, url)
                    Log.d(TAG, "Page finished loading: $url")
                    progressBar?.visibility = View.GONE
                }

                override fun onReceivedError(
                    view: WebView?,
                    errorCode: Int,
                    description: String?,
                    failingUrl: String?
                ) {
                    super.onReceivedError(view, errorCode, description, failingUrl)
                    Log.e(TAG, "WebView error: $errorCode - $description for URL: $failingUrl")
                    progressBar?.visibility = View.GONE
                }

                override fun shouldOverrideUrlLoading(view: WebView?, url: String?): Boolean {
                    Log.d(TAG, "shouldOverrideUrlLoading: $url")
                    return false // Let WebView handle the URL
                }
            }

            // WebChromeClient for handling progress
            webView.webChromeClient = object : WebChromeClient() {
                override fun onProgressChanged(view: WebView?, newProgress: Int) {
                    super.onProgressChanged(view, newProgress)
                    Log.d(TAG, "Loading progress: $newProgress%")
                    progressBar?.progress = newProgress
                    if (newProgress == 100) {
                        progressBar?.visibility = View.GONE
                    }
                }
            }
        }
    }

    private fun setupMenuButton() {
        btnMenu?.setOnClickListener { view ->
            showPopupMenu(view)
        }
        btnClose?.setOnClickListener {
            dismiss()
        }
    }

    private fun showPopupMenu(anchorView: View) {
        val popupMenu = PopupMenu(requireContext(), anchorView)
        popupMenu.menuInflater.inflate(R.menu.model_card_menu, popupMenu.menu)
        
        popupMenu.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.menu_open_in_browser -> {
                    openInBrowser()
                    true
                }
                else -> false
            }
        }
        
        popupMenu.show()
    }

    private fun openInBrowser() {
        url?.let { url ->
            try {
                val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
                startActivity(intent)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open URL in browser: $url", e)
                Toast.makeText(requireContext(), getString(R.string.failed_to_open_url), Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroyView() {
        webView?.destroy()
        webView = null
        super.onDestroyView()
    }
} 