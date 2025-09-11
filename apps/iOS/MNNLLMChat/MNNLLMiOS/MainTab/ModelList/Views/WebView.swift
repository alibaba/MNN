//
//  WebView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/13.
//

import SwiftUI

import WebKit

struct WebView: UIViewRepresentable {
    let url: URL

    func makeUIView(context _: Context) -> WKWebView {
        let webView = WKWebView()
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context _: Context) {
        let request = URLRequest(url: url)
        uiView.load(request)
    }
}
