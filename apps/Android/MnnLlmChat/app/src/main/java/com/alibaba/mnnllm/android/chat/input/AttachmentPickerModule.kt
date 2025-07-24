// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.input

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import androidx.core.content.FileProvider
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import java.io.File
import java.io.IOException

class AttachmentPickerModule(private val activity: ChatActivity) {
    private val takePhotoView: View
    private val chooseImageView: View

    private val attachmentPreview: ImageView
    private val imagePreviewLayout: View
    private val imagePreviewDelete: ImageView
    private val selectAttachmentLayoutParent: View

    private var imageUri: Uri? = null
    private var photoFile: File? = null
    private var callback: ImagePickCallback? = null

    init {
        val modelName = activity.modelName
        takePhotoView = activity.findViewById(R.id.more_item_camera)
        chooseImageView = activity.findViewById(R.id.more_item_photo)
        if (ModelUtils.isVisualModel(modelName)) {
            takePhotoView.setOnClickListener { v: View? -> takePhoto() }
            chooseImageView.setOnClickListener { v: View? -> chooseImageView() }
        } else {
            takePhotoView.visibility = View.GONE
            chooseImageView.visibility = View.GONE
        }
        val chooseAudioView = activity.findViewById<View>(R.id.more_item_audio)
        if (ModelUtils.isAudioModel(modelName)) {
            chooseAudioView.setOnClickListener { v: View? -> chooseAudio() }
        } else {
            chooseAudioView.visibility = View.GONE
        }

        // Voice chat menu item - show for all models except diffusion
        val voiceChatView = activity.findViewById<View>(R.id.more_item_voice_chat)
//        if (!ModelUtils.isDiffusionModel(modelName)) {
//            voiceChatView.setOnClickListener { v: View? -> startVoiceChat() }
//        } else {
//            voiceChatView.visibility = View.GONE
//        }
        //disable temporary
        voiceChatView.visibility = View.GONE
        attachmentPreview = activity.findViewById(R.id.image_preview)
        imagePreviewLayout = activity.findViewById(R.id.image_preview_layout)
        imagePreviewDelete = activity.findViewById(R.id.image_preview_delete)
        selectAttachmentLayoutParent = activity.findViewById(R.id.layout_more_menu)
        imagePreviewDelete.setOnClickListener { v: View? -> deletePreviewImage() }
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
                Intent.createChooser(intent, "Select a WAV file"),
                REQUEST_CODE_SELECT_WAV
            )
        } catch (ex: ActivityNotFoundException) {
            Toast.makeText(this.activity, "Please install a File Manager.", Toast.LENGTH_SHORT)
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
        activity.startActivityForResult(
            Intent.createChooser(intent, "Select Picture"),
            REQUEST_CODE_SELECT_IMAGE,
            null
        )
    }

    private fun takePhoto() {
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
        activity.startActivityForResult(cameraIntent, REQUEST_CODE_CAPTURE_IMAGE)
    }

    fun setOnImagePickCallback(callback: ImagePickCallback?) {
        this.callback = callback
    }

    fun canHandleResult(requestCode: Int): Boolean {
        return requestCode >= REQUEST_CODE_SELECT_WAV && requestCode <= REQUEST_CODE_CAPTURE_IMAGE
    }

    fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == REQUEST_CODE_CAPTURE_IMAGE) {
            if (resultCode == Activity.RESULT_OK) {
                if (imageUri != null) {
                    showImagePreview()
                    val imagePath = imageUri?.path
                    Log.d("ImagePath", "Image saved to: $imagePath")
                    showImagePreview()
                }
            }
            imageUri = null
        } else if (requestCode == REQUEST_CODE_SELECT_IMAGE) {
            if (resultCode == Activity.RESULT_OK) {
                val uri = data!!.data
                try {
                    val destImageFile = FileUtils.generateDestImageFilePath(
                        this.activity,
                        activity.sessionId!!
                    )
                    FileUtils.copyFileUriToPath(
                        this.activity,
                        uri!!,
                        destImageFile
                    )
                    imageUri = Uri.fromFile(File(destImageFile))
                    showImagePreview()
                } catch (e: IOException) {
                    Log.e(TAG, "get file failed ", e)
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
                    Toast.makeText(this.activity, "get audio file failed", Toast.LENGTH_SHORT)
                        .show()
                }
            }
        }
    }

    private fun showAudioPreview(audioUri: Uri) {
        attachmentPreview.setImageResource(R.drawable.ic_audio_attachment)
        imagePreviewLayout.visibility = View.VISIBLE
        imagePreviewDelete.visibility = View.VISIBLE
        hideAttachmentLayout()
        if (callback != null) {
            callback!!.onAttachmentPicked(audioUri, AttachmentType.Audio)
        }
    }

    private fun showImagePreview() {
        attachmentPreview.setImageURI(imageUri)
        imagePreviewLayout.visibility = View.VISIBLE
        imagePreviewDelete.visibility = View.VISIBLE
        hideAttachmentLayout()
        if (callback != null) {
            callback!!.onAttachmentPicked(imageUri, AttachmentType.Image)
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
        hidePreview()
    }

    interface ImagePickCallback {
        fun onAttachmentPicked(imageUri: Uri?, audio: AttachmentType?)
        fun onAttachmentRemoved()

        fun onAttachmentLayoutShow()

        fun onAttachmentLayoutHide()
    }

    enum class AttachmentType {
        Image, Audio
    }

    companion object {
        const val TAG: String = "ImagePickerModule"
        var REQUEST_CODE_CAPTURE_IMAGE: Int = 100

        var REQUEST_CODE_SELECT_IMAGE: Int = 99
        var REQUEST_CODE_SELECT_WAV: Int = 98
    }
}
