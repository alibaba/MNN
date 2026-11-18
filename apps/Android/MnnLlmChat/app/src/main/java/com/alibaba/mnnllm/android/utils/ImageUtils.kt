package com.alibaba.mnnllm.android.utils

import android.content.ContentValues
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ColorSpace
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import androidx.annotation.CheckResult
import androidx.annotation.WorkerThread
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.findViewTreeLifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.google.android.material.bottomsheet.BottomSheetDialog
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

object ImageUtils {
    private const val TAG = "ImageUtils"

    fun showImageMenu(context: Context, imageUri: Uri, showShareOption: Boolean = true) {
        val dialog = BottomSheetDialog(context)
        val view = LayoutInflater.from(context).inflate(R.layout.bottom_sheet_image_menu, null)

        view.findViewById<View>(R.id.btn_save_image).setOnClickListener {
            dialog.dismiss()
            val success = saveImageToGallery(context, imageUri)
            if (success) {
                Toast.makeText(context, R.string.image_saved_to_gallery, Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(context, R.string.failed_to_save_image, Toast.LENGTH_SHORT).show()
            }
        }

        view.findViewById<View>(R.id.btn_share_image).setOnClickListener {
            dialog.dismiss()
            shareImage(context, imageUri)
        }
        view.findViewById<View>(R.id.btn_share_image).visibility =
            if (showShareOption) View.VISIBLE else View.GONE

        dialog.show()
        dialog.setContentView(view)
    }

    fun shareImage(context: Context, imageUri: Uri) {
        val shareUri = if (imageUri.scheme == "file") {
            androidx.core.content.FileProvider.getUriForFile(
                context,
                context.packageName + ".fileprovider",
                File(imageUri.path!!)
            )
        } else {
            imageUri
        }

        val shareIntent = Intent(Intent.ACTION_SEND).apply {
            type = "image/*"
            putExtra(Intent.EXTRA_STREAM, shareUri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(shareIntent, context.getString(R.string.share_image)))
    }

    fun saveImageToGallery(context: Context, imageUri: Uri): Boolean {
        val fileName = "MNN_Chat_${System.currentTimeMillis()}.jpg"
        val resolver = context.contentResolver

        // Create content values for the new image
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/MNN-Chat")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
        }

        // Insert the new image into MediaStore
        val galleryUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            ?: return false

        return try {
            // For file:// URI, use FileInputStream for better compatibility
            if (imageUri.scheme == "file") {
                val filePath = imageUri.path
                if (filePath != null) {
                    File(filePath).inputStream().use { inputStream ->
                        resolver.openOutputStream(galleryUri)?.use { outputStream ->
                            inputStream.copyTo(outputStream)
                        }
                    }
                } else {
                    return false
                }
            } else {
                // For content:// and other URIs, use ContentResolver
                resolver.openInputStream(imageUri)?.use { inputStream ->
                    resolver.openOutputStream(galleryUri)?.use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            }

            // Mark as not pending if on Android Q+
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                contentValues.clear()
                contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
                resolver.update(galleryUri, contentValues, null, null)
            }

            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save image to gallery", e)
            // Delete the failed entry
            try {
                resolver.delete(galleryUri, null, null)
            } catch (deleteException: Exception) {
                Log.e(TAG, "Failed to delete partial image", deleteException)
            }
            false
        }
    }

    const val ATTACHMENT_MAX_DIMENSION = 1024
    const val FULLSCREEN_MAX_DIMENSION = 2048
    const val PREVIEW_MAX_DIMENSION = 512
    private val imageWorkLock = Any()

    internal fun calculateInSampleSize(
        width: Int,
        height: Int,
        maxDimension: Int,
        maxPixels: Long
    ): Int? {
        if (width <= 0 || height <= 0 || maxDimension <= 0 || maxPixels <= 0) return null
        var sample = 1
        while (true) {
            val sampledWidth = (width.toLong() + sample - 1) / sample
            val sampledHeight = (height.toLong() + sample - 1) / sample
            if (sampledWidth <= maxDimension && sampledHeight <= maxDimension &&
                sampledWidth * sampledHeight <= maxPixels) return sample
            if (sample >= (1 shl 30)) return null
            sample *= 2
        }
    }

    @WorkerThread
    internal fun decodeBitmap(
        context: Context,
        uri: Uri,
        maxDimension: Int,
        maxPixels: Long = maxDimension.toLong() * maxDimension
    ): Bitmap? = synchronized(imageWorkLock) {
        decodeBitmap(maxDimension, maxPixels) {
            context.contentResolver.openInputStream(uri) ?: throw IOException("Cannot open image")
        }
    }

    private fun decodeBitmap(
        maxDimension: Int,
        maxPixels: Long,
        open: () -> InputStream
    ): Bitmap? {
        var decoded: Bitmap? = null
        var transformed: Bitmap? = null
        try {
            val limit = maxDimension.coerceAtMost(FULLSCREEN_MAX_DIMENSION)
            val pixels = maxPixels.coerceAtMost(limit.toLong() * limit)
            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            open().use { BitmapFactory.decodeStream(it, null, options) }
            val sample = calculateInSampleSize(options.outWidth, options.outHeight, limit, pixels)
                ?: return null
            val orientation = try {
                open().use {
                    ExifInterface(it).getAttributeInt(
                        ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
                    )
                }
            } catch (e: IOException) {
                ExifInterface.ORIENTATION_NORMAL
            }
            options.inJustDecodeBounds = false
            options.inSampleSize = sample
            options.inScaled = false
            options.inPreferredConfig = Bitmap.Config.ARGB_8888
            options.inPreferredColorSpace = ColorSpace.get(ColorSpace.Named.SRGB)
            open().use { decoded = BitmapFactory.decodeStream(it, null, options) }
            val bitmap = decoded ?: return null
            if (calculateInSampleSize(bitmap.width, bitmap.height, limit, pixels) != 1) return null
            val matrix = Matrix().apply {
                when (orientation) {
                    ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> setScale(-1f, 1f)
                    ExifInterface.ORIENTATION_ROTATE_180 -> setRotate(180f)
                    ExifInterface.ORIENTATION_FLIP_VERTICAL -> setScale(1f, -1f)
                    ExifInterface.ORIENTATION_TRANSPOSE -> {
                        setRotate(90f)
                        postScale(-1f, 1f)
                    }
                    ExifInterface.ORIENTATION_ROTATE_90 -> setRotate(90f)
                    ExifInterface.ORIENTATION_TRANSVERSE -> {
                        setRotate(270f)
                        postScale(-1f, 1f)
                    }
                    ExifInterface.ORIENTATION_ROTATE_270 -> setRotate(270f)
                }
            }
            val result = if (matrix.isIdentity) bitmap else {
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            }
            transformed = result
            if (result !== bitmap) bitmap.recycle()
            decoded = null
            transformed = null
            return result
        } catch (e: IOException) {
            Log.w(TAG, "Cannot read image", e)
        } catch (e: SecurityException) {
            Log.w(TAG, "Image access denied", e)
        } catch (e: IllegalArgumentException) {
            Log.w(TAG, "Invalid image", e)
        } catch (e: OutOfMemoryError) {
            Log.w(TAG, "Insufficient memory for bounded image decode")
        } finally {
            if (transformed !== decoded) transformed?.recycle()
            decoded?.recycle()
        }
        return null
    }

    @WorkerThread
    @CheckResult
    fun createImageAttachment(context: Context, uri: Uri, directory: File): File? =
        synchronized(imageWorkLock) {
            try {
                val path = directory.canonicalPath
                if (!path.startsWith(context.filesDir.canonicalPath + File.separator)) return null
                if (!directory.isDirectory && !directory.mkdirs()) return null
                val bitmap = decodeBitmap(context, uri, ATTACHMENT_MAX_DIMENSION) ?: return null
                try {
                    writeDerivedImage(bitmap, directory, 80)
                } finally {
                    bitmap.recycle()
                }
            } catch (e: IOException) {
                Log.w(TAG, "Cannot create image attachment", e)
                null
            } catch (e: SecurityException) {
                Log.w(TAG, "Cannot access image attachment directory", e)
                null
            }
        }

    private fun writeDerivedImage(bitmap: Bitmap, directory: File, quality: Int): File? {
        var output: File? = null
        var complete = false
        try {
            if (quality !in 0..100) return null
            val file = File.createTempFile("image_", ".jpg", directory)
            output = file
            FileOutputStream(file).use {
                if (!bitmap.compress(Bitmap.CompressFormat.JPEG, quality, it)) return null
            }
            if (file.length() == 0L) return null
            complete = true
            return file
        } catch (e: IOException) {
            Log.w(TAG, "Cannot write derived image", e)
        } catch (e: SecurityException) {
            Log.w(TAG, "Cannot access derived image", e)
        } catch (e: IllegalArgumentException) {
            Log.w(TAG, "Cannot encode derived image", e)
        } catch (e: OutOfMemoryError) {
            Log.w(TAG, "Insufficient memory for image encoding")
        } finally {
            if (!complete) output?.delete()
        }
        return null
    }

    @WorkerThread
    @CheckResult
    fun compressImageFile(
        file: File,
        maxDimension: Int = ATTACHMENT_MAX_DIMENSION,
        quality: Int = 80
    ): Boolean = synchronized(imageWorkLock) {
        var derived: File? = null
        try {
            val bitmap = decodeBitmap(maxDimension, maxDimension.toLong() * maxDimension) {
                file.inputStream()
            } ?: return false
            derived = try {
                writeDerivedImage(bitmap, file.absoluteFile.parentFile ?: return false, quality)
            } finally {
                bitmap.recycle()
            }
            val output = derived ?: return false
            Files.move(output.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            true
        } catch (e: IOException) {
            Log.w(TAG, "Cannot replace captured image", e)
            false
        } catch (e: SecurityException) {
            Log.w(TAG, "Cannot access captured image", e)
            false
        } finally {
            derived?.delete()
        }
    }

    internal class ImageRequest(
        private val imageView: ImageView,
        private val maxDimension: Int,
        private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
        private val decode: (Context, Uri, Int) -> Bitmap? = { context, uri, limit ->
            decodeBitmap(context, uri, limit)
        }
    ) : View.OnAttachStateChangeListener {
        private var uri: Uri? = null
        private var job: Job? = null
        private var generation = 0L

        init {
            imageView.addOnAttachStateChangeListener(this)
        }

        fun bind(uri: Uri) {
            clear()
            this.uri = uri
            if (imageView.isAttachedToWindow) load()
        }

        fun clear() {
            uri = null
            cancel()
        }

        private fun cancel() {
            generation++
            job?.cancel()
            job = null
            imageView.setImageDrawable(null)
        }

        private fun load() {
            val source = uri ?: return
            val owner = imageView.findViewTreeLifecycleOwner() ?: findLifecycleOwner(imageView.context)
                ?: return
            if (owner.lifecycle.currentState == Lifecycle.State.DESTROYED) return
            val token = generation
            val context = imageView.context.applicationContext
            job = owner.lifecycleScope.launch {
                var bitmap: Bitmap? = null
                try {
                    withContext(dispatcher) {
                        synchronized(imageWorkLock) {
                            ensureActive()
                            bitmap = decode(context, source, maxDimension)
                        }
                    }
                    ensureActive()
                    if (token == generation && imageView.isAttachedToWindow) {
                        imageView.setImageBitmap(bitmap)
                        bitmap = null
                    }
                } finally {
                    bitmap?.recycle()
                }
            }
        }

        override fun onViewAttachedToWindow(view: View) {
            load()
        }

        override fun onViewDetachedFromWindow(view: View) {
            cancel()
        }

        private fun findLifecycleOwner(context: Context): LifecycleOwner? {
            var current = context
            while (true) {
                if (current is LifecycleOwner) return current
                if (current !is ContextWrapper || current.baseContext === current) return null
                current = current.baseContext
            }
        }
    }
}
