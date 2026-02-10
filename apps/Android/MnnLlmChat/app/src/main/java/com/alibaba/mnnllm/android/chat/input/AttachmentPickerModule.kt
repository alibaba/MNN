// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.input

import android.Manifest
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException

class AttachmentPickerModule(private val activity: ChatActivity) {
    private val takePhotoView: View
    private val chooseImageView: View
    private val chooseVideoView: View

    private val attachmentPreview: ImageView
    private val imagePreviewLayout: View
    private val imagePreviewDelete: ImageView
    private val selectAttachmentLayoutParent: View

    private var imageUri: Uri? = null
    private var photoFile: File? = null
    private var callback: ImagePickCallback? = null

    private val imagePreviewRecycler: androidx.recyclerview.widget.RecyclerView
    private val imagePreviewAdapter: ImagePreviewAdapter

    init {
        takePhotoView = activity.findViewById(R.id.more_item_camera)
        chooseImageView = activity.findViewById(R.id.more_item_photo)
        chooseVideoView = activity.findViewById(R.id.more_item_video)
        if (ModelTypeUtils.isVisualModel(activity.modelId!!)) {
            takePhotoView.setOnClickListener { v: View? -> takePhoto() }
            chooseImageView.setOnClickListener { v: View? -> chooseImageView() }
        } else {
            takePhotoView.visibility = View.GONE
            chooseImageView.visibility = View.GONE
        }
        if (ModelTypeUtils.isVideoModel(activity.modelId!!)) {
            chooseVideoView.setOnClickListener { v: View? -> chooseVideo() }
        } else {
            chooseVideoView.visibility = View.GONE
        }
        val chooseAudioView = activity.findViewById<View>(R.id.more_item_audio)
        if (ModelTypeUtils.isAudioModel(activity.modelId!!)) {
            chooseAudioView.setOnClickListener { v: View? -> chooseAudio() }
        } else {
            chooseAudioView.visibility = View.GONE
        }
        val voiceChatView = activity.findViewById<View>(R.id.more_item_voice_chat)
        //disable temporary
        voiceChatView.visibility = View.GONE
        attachmentPreview = activity.findViewById(R.id.image_preview)
        imagePreviewLayout = activity.findViewById(R.id.image_preview_layout)
        imagePreviewRecycler = activity.findViewById(R.id.image_preview_recycler)
        imagePreviewDelete = activity.findViewById(R.id.image_preview_delete)
        selectAttachmentLayoutParent = activity.findViewById(R.id.layout_more_menu)
        imagePreviewDelete.setOnClickListener { v: View? -> deletePreviewImage() }

        imagePreviewAdapter = ImagePreviewAdapter { uri ->
            imagePreviewAdapter.removeImage(uri)
            if (imagePreviewAdapter.itemCount == 0) {
                hidePreview()
            } else {
                if (callback != null) {
                    callback!!.onAttachmentPicked(imagePreviewAdapter.getImages(), AttachmentType.Image)
                }
            }
        }
        imagePreviewRecycler.adapter = imagePreviewAdapter
    }

    private fun deletePreviewImage() {
        if (imageUri != null) {
            if (photoFile != null) {
                photoFile!!.delete()
                photoFile = null
            }
            imageUri = null
        }
        showAttachmentLayout()
        hidePreview()
    }

    private fun hidePreview() {
        imagePreviewLayout.visibility = View.GONE
        imagePreviewRecycler.visibility = View.GONE
        imagePreviewDelete.visibility = View.GONE
        if (callback != null) {
            callback!!.onAttachmentRemoved()
        }
    }

    private fun chooseAudio() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.setType("audio/x-wav")
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        try {
            activity.startActivityForResult(
                Intent.createChooser(intent, activity.getString(R.string.select_wav_file)),
                REQUEST_CODE_SELECT_WAV
            )
        } catch (ex: ActivityNotFoundException) {
            Toast.makeText(this.activity, R.string.file_manager_required, Toast.LENGTH_SHORT)
                .show()
        }
    }

    private fun startVoiceChat() {
        // Hide the attachment menu
        hideAttachmentLayout()
        
        // Start the voice chat fragment
        activity.startVoiceChat()
    }

    private fun chooseImageView() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.setType("image/*")
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
        activity.startActivityForResult(
            Intent.createChooser(intent, activity.getString(R.string.select_picture)),
            REQUEST_CODE_SELECT_IMAGE,
            null
        )
    }

    private fun chooseVideo() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.setType("video/*")
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        try {
            activity.startActivityForResult(
                Intent.createChooser(intent, activity.getString(R.string.select_video)),
                REQUEST_CODE_SELECT_VIDEO
            )
        } catch (ex: ActivityNotFoundException) {
            Toast.makeText(this.activity, R.string.file_manager_required, Toast.LENGTH_SHORT)
                .show()
        }
    }

    private fun takePhoto() {
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) 
            != PackageManager.PERMISSION_GRANTED) {
            // Permission is not granted, request it
            if (ActivityCompat.shouldShowRequestPermissionRationale(activity, Manifest.permission.CAMERA)) {
                // Show rationale to user
                Toast.makeText(activity, R.string.camera_permission_rationale, Toast.LENGTH_LONG).show()
            }
            ActivityCompat.requestPermissions(
                activity,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_CODE_CAMERA_PERMISSION
            )
            return
        }
        
        // Permission is granted, proceed with camera
        startCameraIntent()
    }
    
    private fun startCameraIntent() {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        photoFile = File(
            FileUtils.generateDestPhotoFilePath(
                this.activity,
                activity.sessionId!!
            )
        )
        imageUri = Uri.fromFile(photoFile)
        val fileProviderUri = FileProvider.getUriForFile(
            this.activity,
            activity.packageName + ".fileprovider",
            photoFile!!
        )
        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProviderUri)
        try {
            activity.startActivityForResult(cameraIntent, REQUEST_CODE_CAPTURE_IMAGE)
        } catch (e: SecurityException) {
            Log.e(TAG, "Camera permission denied", e)
            Toast.makeText(activity, R.string.camera_permission_denied, Toast.LENGTH_SHORT).show()
        } catch (e: ActivityNotFoundException) {
            Log.e(TAG, "No camera app found", e)
            Toast.makeText(activity, R.string.no_camera_app_found, Toast.LENGTH_SHORT).show()
        }
    }

    fun setOnImagePickCallback(callback: ImagePickCallback?) {
        this.callback = callback
    }

    fun canHandleResult(requestCode: Int): Boolean {
        return requestCode >= REQUEST_CODE_SELECT_WAV && requestCode <= REQUEST_CODE_CAPTURE_IMAGE ||
               requestCode == REQUEST_CODE_CAMERA_PERMISSION ||
               requestCode == REQUEST_CODE_SELECT_VIDEO
    }

    fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == REQUEST_CODE_CAPTURE_IMAGE) {
            if (resultCode == Activity.RESULT_OK) {
                if (imageUri != null) {
                    val imagePath = imageUri?.path
                    Log.d("ImagePath", "Image saved to: $imagePath")
                    imagePreviewAdapter.addImage(imageUri!!)
                    showImagePreview()
                }
            }
            imageUri = null
        } else if (requestCode == REQUEST_CODE_SELECT_IMAGE) {
            if (resultCode == Activity.RESULT_OK && data != null) {
                val uris = mutableListOf<Uri>()
                if (data.clipData != null) {
                    val count = data.clipData!!.itemCount
                    for (i in 0 until count) {
                        uris.add(data.clipData!!.getItemAt(i).uri)
                    }
                } else if (data.data != null) {
                    uris.add(data.data!!)
                }

                if (uris.isNotEmpty()) {
                    val processedUris = mutableListOf<Uri>()
                    for (uri in uris) {
                        try {
                            val destImageFile = FileUtils.generateDestImageFilePath(
                                this.activity,
                                activity.sessionId!!
                            )
                            FileUtils.copyFileUriToPath(
                                this.activity,
                                uri,
                                destImageFile
                            )
                            processedUris.add(Uri.fromFile(File(destImageFile)))
                        } catch (e: IOException) {
                            Log.e(TAG, "get file failed ", e)
                        }
                    }
                    if (processedUris.isNotEmpty()) {
                        imagePreviewAdapter.addImages(processedUris)
                        showImagePreview()
                    }
                }
            }
        } else if (requestCode == REQUEST_CODE_SELECT_VIDEO) {
            if (resultCode == Activity.RESULT_OK) {
                val videoUri = data!!.data
                try {
                    val destVideoPath = FileUtils.generateDestVideoFilePath(
                        this.activity,
                        activity.sessionId!!
                    )
                    val destFile =
                        FileUtils.copyFileUriToPath(this.activity, videoUri!!, destVideoPath)
                    showVideoPreview(Uri.fromFile(destFile))
                } catch (e: IOException) {
                    Log.e(TAG, "get video file failed", e)
                    Toast.makeText(this.activity, R.string.video_file_failed, Toast.LENGTH_SHORT)
                        .show()
                }
            }
        } else if (requestCode == REQUEST_CODE_SELECT_WAV) {
            if (resultCode == Activity.RESULT_OK) {
                val audioUri = data!!.data
                try {
                    val destAudioPath = FileUtils.generateDestAudioFilePath(
                        this.activity,
                        activity.sessionId!!
                    )
                    val destFile =
                        FileUtils.copyFileUriToPath(this.activity, audioUri!!, destAudioPath)
                    showAudioPreview(Uri.fromFile(destFile))
                } catch (e: IOException) {
                    Log.e(TAG, "get audio file failed", e)
                    Toast.makeText(this.activity, R.string.audio_file_failed, Toast.LENGTH_SHORT)
                        .show()
                }
            }
        }
    }
    
    fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Camera permission granted, proceed with camera
                startCameraIntent()
            } else {
                // Camera permission denied
                Toast.makeText(activity, R.string.camera_permission_denied, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun showAudioPreview(audioUri: Uri) {
        attachmentPreview.setImageResource(R.drawable.ic_audio_attachment)
        imagePreviewLayout.visibility = View.VISIBLE
        imagePreviewDelete.visibility = View.VISIBLE
        imagePreviewRecycler.visibility = View.GONE
        hideAttachmentLayout()
        if (callback != null) {
            callback!!.onAttachmentPicked(listOf(audioUri), AttachmentType.Audio)
        }
    }

    private fun showVideoPreview(videoUri: Uri) {
        activity.lifecycleScope.launch(Dispatchers.IO) {
            try {
                val thumbnail = com.alibaba.mnnllm.android.utils.VideoThumbnailUtils.generateVideoThumbnail(
                    activity, videoUri, 200, 200
                )
                withContext(Dispatchers.Main) {
                    if (thumbnail != null) {
                        attachmentPreview.setImageBitmap(thumbnail)
                    } else {
                        attachmentPreview.setImageResource(R.drawable.ic_video)
                    }
                    imagePreviewLayout.visibility = View.VISIBLE
                    imagePreviewDelete.visibility = View.VISIBLE
                    imagePreviewRecycler.visibility = View.GONE
                    hideAttachmentLayout()
                    if (callback != null) {
                        callback!!.onAttachmentPicked(listOf(videoUri), AttachmentType.Video)
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    attachmentPreview.setImageResource(R.drawable.ic_video)
                    imagePreviewLayout.visibility = View.VISIBLE
                    imagePreviewDelete.visibility = View.VISIBLE
                    imagePreviewRecycler.visibility = View.GONE
                    hideAttachmentLayout()
                    if (callback != null) {
                        callback!!.onAttachmentPicked(listOf(videoUri), AttachmentType.Video)
                    }
                }
            }
        }
    }

    private fun showImagePreview() {
        imagePreviewRecycler.visibility = View.VISIBLE
        imagePreviewLayout.visibility = View.GONE
        imagePreviewDelete.visibility = View.GONE
        hideAttachmentLayout()
        if (callback != null) {
            callback!!.onAttachmentPicked(imagePreviewAdapter.getImages(), AttachmentType.Image)
        }
        imageUri = null
    }

    fun toggleAttachmentVisibility() {
        if (isShowing) {
            hideAttachmentLayout()
        } else {
            showAttachmentLayout()
        }
    }

    fun hideAttachmentLayout() {
        selectAttachmentLayoutParent.visibility = View.GONE
        if (callback != null) {
            callback!!.onAttachmentLayoutHide()
        }
    }

    private fun showAttachmentLayout() {
        selectAttachmentLayoutParent.visibility = View.VISIBLE
        if (callback != null) {
            callback!!.onAttachmentLayoutShow()
        }
    }

    val isShowing: Boolean
        get() = selectAttachmentLayoutParent.visibility == View.VISIBLE

    fun clearInput() {
        photoFile = null
        imageUri = null
        imagePreviewAdapter.clear()
        hidePreview()
    }

    interface ImagePickCallback {
        fun onAttachmentPicked(imageUris: List<Uri>?, audio: AttachmentType?)
        fun onAttachmentRemoved()

        fun onAttachmentLayoutShow()

        fun onAttachmentLayoutHide()
    }

    enum class AttachmentType {
        Image, Audio, Video
    }

    companion object {
        const val TAG: String = "ImagePickerModule"
        var REQUEST_CODE_CAPTURE_IMAGE: Int = 100
        const val REQUEST_CODE_CAMERA_PERMISSION: Int = 101

        var REQUEST_CODE_SELECT_IMAGE: Int = 99
        var REQUEST_CODE_SELECT_VIDEO: Int = 97
        var REQUEST_CODE_SELECT_WAV: Int = 98
    }
}
