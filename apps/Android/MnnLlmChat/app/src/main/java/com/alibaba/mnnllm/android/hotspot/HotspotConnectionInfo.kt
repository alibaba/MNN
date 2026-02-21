package com.alibaba.mnnllm.android.hotspot

data class HotspotConnectionInfo(
    val ssid: String,
    val password: String,
    val gatewayIp: String,
    val port: Int,
) {
    /** Standard Wi-Fi QR format recognized natively by Android and iOS cameras. */
    val wifiQrContent: String
        get() = "WIFI:S:${ssid.escapeWifiQr()};T:WPA;P:${password.escapeWifiQr()};;"

    /** URL the user should navigate to after joining the hotspot. */
    val urlQrContent: String
        get() = "http://$gatewayIp:$port"

    companion object {
        /**
         * Characters that must be escaped in the WIFI: QR format:
         * \ ; , " :
         */
        private fun String.escapeWifiQr(): String =
            this.replace("\\", "\\\\")
                .replace(";", "\\;")
                .replace(",", "\\,")
                .replace("\"", "\\\"")
                .replace(":", "\\:")
    }
}
