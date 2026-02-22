package com.alibaba.mnnllm.android.hotspot

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "HotspotViewModel"

/** Default port on which [LocalWebServer] listens. */
private const val WEB_SERVER_PORT = 8080

/**
 * Orchestrates the local hotspot, web server, and QR code generation.
 *
 * Exposes:
 *  - [wifiQrBitmap] — QR code for joining the hotspot Wi-Fi network
 *  - [urlQrBitmap]  — QR code for the landing-page URL
 *  - [error]        — non-null when a fatal error occurs
 */
class HotspotViewModel(application: Application) : AndroidViewModel(application) {

    private val hotspotManager = LocalHotspotManager(application)
    private val webServer = LocalWebServer(WEB_SERVER_PORT)

    private val _wifiQrBitmap = MutableLiveData<Bitmap?>()
    val wifiQrBitmap: LiveData<Bitmap?> = _wifiQrBitmap

    private val _urlQrBitmap = MutableLiveData<Bitmap?>()
    val urlQrBitmap: LiveData<Bitmap?> = _urlQrBitmap

    private val _connectionInfo = MutableLiveData<HotspotConnectionInfo?>()
    val connectionInfo: LiveData<HotspotConnectionInfo?> = _connectionInfo

    private val _error = MutableLiveData<String?>()
    val error: LiveData<String?> = _error

    /** Starts the local-only hotspot and web server. */
    fun startHotspot() {
        viewModelScope.launch(Dispatchers.IO) {
            hotspotManager.hotspotInfoFlow()
                .catch { e ->
                    Log.e(TAG, "Hotspot flow error", e)
                    _error.postValue(e.message ?: "Unknown hotspot error")
                }
                .collect { result ->
                    result.onSuccess { info ->
                        val connInfo = HotspotConnectionInfo(
                            ssid = info.ssid,
                            password = info.passphrase,
                            gatewayIp = info.gatewayIp,
                            port = WEB_SERVER_PORT,
                        )
                        _connectionInfo.postValue(connInfo)
                        webServer.start()

                        val wifiBitmap = withContext(Dispatchers.Default) {
                            QrCodeGenerator.generate(connInfo.wifiQrContent)
                        }
                        val urlBitmap = withContext(Dispatchers.Default) {
                            QrCodeGenerator.generate(connInfo.urlQrContent)
                        }
                        _wifiQrBitmap.postValue(wifiBitmap)
                        _urlQrBitmap.postValue(urlBitmap)
                    }
                    result.onFailure { e ->
                        Log.e(TAG, "Hotspot failed", e)
                        _error.postValue(e.message ?: "Failed to start hotspot")
                    }
                }
        }
    }

    override fun onCleared() {
        super.onCleared()
        webServer.stop()
    }
}
