package com.alibaba.mnnllm.android.modelmarket

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.webkit.WebViewClient
import android.webkit.WebChromeClient
import android.widget.TextView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.BaseBottomSheetDialogFragment

class ModelCardWebViewBottomSheet : BaseBottomSheetDialogFragment() {
    private var url: String? = null
    private var title: String? = null

    companion object {
        private const val ARG_URL = "arg_url"
        private const val ARG_TITLE = "arg_title"

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

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.dialog_fragment_model_card_webview, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val tvTitle = view.findViewById<TextView>(R.id.tvModelCardTitle)
        val webView = view.findViewById<WebView>(R.id.webViewModelCard)
        tvTitle.text = title ?: getString(R.string.model_card)
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, url: String?): Boolean {
                url?.let { view?.loadUrl(it) }
                return true
            }
        }
        webView.webChromeClient = WebChromeClient()
        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        url?.let { webView.loadUrl(it) }
    }
} 