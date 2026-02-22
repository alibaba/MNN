package com.alibaba.mnnllm.android.hotspot

import android.graphics.Bitmap
import com.google.zxing.BarcodeFormat
import com.journeyapps.barcodescanner.BarcodeEncoder

/**
 * Generates QR code [Bitmap]s from string content using the ZXing library.
 */
object QrCodeGenerator {

    /**
     * Encodes [content] as a QR code [Bitmap] with the given [widthPx] and [heightPx].
     *
     * @return the generated [Bitmap], or `null` if encoding fails.
     */
    fun generate(content: String, widthPx: Int = 512, heightPx: Int = 512): Bitmap? {
        return try {
            BarcodeEncoder().encodeBitmap(content, BarcodeFormat.QR_CODE, widthPx, heightPx)
        } catch (e: Exception) {
            null
        }
    }
}
