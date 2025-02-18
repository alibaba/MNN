// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import static android.app.Activity.RESULT_OK;

import android.content.Intent;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.core.content.FileProvider;

import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.utils.FileUtils;
import com.alibaba.mnnllm.android.utils.ModelUtils;

import java.io.File;
import java.io.IOException;

public class AttachmentPickerModule {

    private final ChatActivity activity;
    private final String modelName;
    private View takePhotoView;
    private View chooseImageView;

    private final ImageView attachmentPreview;
    private final View imagePreivewLayout;
    private final ImageView imagePreviewDelete;
    private final View selectAttachmentLayoutParent;

    public static final String TAG = "ImagePickerModule";
    public static int REQUEST_CODE_CAPTURE_IMAGE = 100;

    public static int REQUEST_CODE_SELECT_IMAGE = 99;
    public static int REQUEST_CODE_SELECT_WAV = 98;

    private Uri imageUri;
    private File photoFile;
    private ImagePickCallback callback;

    public AttachmentPickerModule(ChatActivity activity) {
        this.activity = activity;
        this.modelName = activity.getModelName();
        takePhotoView = this.activity.findViewById(R.id.more_item_camera);
        chooseImageView = this.activity.findViewById(R.id.more_item_photo);
        if (ModelUtils.isVisualModel(this.modelName)) {
            takePhotoView.setOnClickListener(v -> takePhoto());
            chooseImageView.setOnClickListener(v -> chooseImageView());
        } else {
            takePhotoView.setVisibility(View.GONE);
            chooseImageView.setVisibility(View.GONE);
        }
        View chooseAudioView = this.activity.findViewById(R.id.more_item_audio);
        if (ModelUtils.isAudioModel(this.modelName)) {
            chooseAudioView.setOnClickListener(v -> chooseAudio());
        } else {
            chooseAudioView.setVisibility(View.GONE);
        }
        attachmentPreview = this.activity.findViewById(R.id.image_preview);
        imagePreivewLayout = this.activity.findViewById(R.id.image_preview_layout);
        imagePreviewDelete = this.activity.findViewById(R.id.image_preview_delete);
        selectAttachmentLayoutParent = this.activity.findViewById(R.id.layout_more_menu);
        imagePreviewDelete.setOnClickListener(v -> deletePreviewImage());
    }

    private void deletePreviewImage() {
        if (imageUri != null) {
            if (photoFile != null) {
                photoFile.delete();
                photoFile = null;
            }
            imageUri = null;
        }
        showAttachmentLayout();
        hidePreview();
    }

    private void hidePreview() {
        imagePreivewLayout.setVisibility(View.GONE);
        imagePreviewDelete.setVisibility(View.GONE);
        if (callback != null) {
            callback.onAttachmentRemoved();
        }
    }

    private void chooseAudio() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("audio/x-wav");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        try {
            this.activity.startActivityForResult(Intent.createChooser(intent, "Select a WAV file"), REQUEST_CODE_SELECT_WAV);
        } catch (android.content.ActivityNotFoundException ex) {
            Toast.makeText(this.activity, "Please install a File Manager.", Toast.LENGTH_SHORT).show();
        }
    }

    private void chooseImageView() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        this.activity.startActivityForResult(Intent.createChooser(intent, "Select Picture"), REQUEST_CODE_SELECT_IMAGE, null);
    }

    private void takePhoto() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        photoFile = new File(FileUtils.generateDestPhotoFilePath(this.activity, this.activity.getSessionId()));
        imageUri = Uri.fromFile(photoFile);
        Uri fileProviderUri = FileProvider.getUriForFile(this.activity, this.activity.getPackageName() + ".fileprovider", photoFile);
        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProviderUri);
        this.activity.startActivityForResult(cameraIntent, REQUEST_CODE_CAPTURE_IMAGE);
    }

    public void setOnImagePickCallback(ImagePickCallback callback) {
        this.callback = callback;
    }

    public boolean canHandleResult(int requestCode) {
        return requestCode >= REQUEST_CODE_SELECT_WAV && requestCode <= REQUEST_CODE_CAPTURE_IMAGE;
    }
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_CODE_CAPTURE_IMAGE) {
            if (resultCode == RESULT_OK) {
                String imagePath = imageUri.getPath(); // This is the path to the saved image
                Log.d("ImagePath", "Image saved to: " + imagePath);
                showImagePreview();
            }
            imageUri = null;
        } else if (requestCode == REQUEST_CODE_SELECT_IMAGE) {
            if (resultCode == RESULT_OK) {
                Uri uri = data.getData();
                try {
                    String destImageFile = FileUtils.generateDestImageFilePath(this.activity, this.activity.getSessionId());
                    FileUtils.copyFileUriToPath(this.activity,
                            uri,
                            destImageFile
                            );
                    imageUri = Uri.fromFile(new File(destImageFile));
                    showImagePreview();
                } catch (IOException e) {
                    Log.e(TAG, "get file failed ", e);
                }
            }
        } else if (requestCode == REQUEST_CODE_SELECT_WAV) {
            if (resultCode == RESULT_OK) {
                Uri audioUri = data.getData();
                try {
                    String destAudioPath = FileUtils.generateDestAudioFilePath(this.activity, this.activity.getSessionId());
                    File destFile = FileUtils.copyFileUriToPath(this.activity, audioUri, destAudioPath);
                    showAudioPreview(Uri.fromFile(destFile));
                } catch (IOException e) {
                    Log.e(TAG, "get audio file failed", e);
                    Toast.makeText(this.activity, "get audio file failed" , Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    private void showAudioPreview(Uri audioUri) {
        attachmentPreview.setImageResource(R.drawable.ic_audio_attachment);
        imagePreivewLayout.setVisibility(View.VISIBLE);
        imagePreviewDelete.setVisibility(View.VISIBLE);
        hideAttachmentLayout();
        if (callback != null) {
            callback.onAttachmentPicked(audioUri, AttachmentType.Audio);
        }
    }

    private void showImagePreview() {
        attachmentPreview.setImageURI(imageUri);
        imagePreivewLayout.setVisibility(View.VISIBLE);
        imagePreviewDelete.setVisibility(View.VISIBLE);
        hideAttachmentLayout();
        if (callback != null) {
            callback.onAttachmentPicked(imageUri, AttachmentType.Image);
        }
        imageUri = null;
    }

    public void toggleAttachmentVisibility() {
        if (isShowing()) {
            hideAttachmentLayout();
        } else {
            showAttachmentLayout();
        }
    }
    public void hideAttachmentLayout() {
        selectAttachmentLayoutParent.setVisibility(View.GONE);
        if (callback != null) {
            callback.onAttachmentLayoutHide();
        }
    }
    private void showAttachmentLayout() {
        selectAttachmentLayoutParent.setVisibility(View.VISIBLE);
        if (callback != null) {
            callback.onAttachmentLayoutShow();
        }
    }

    public boolean isShowing() {
        return selectAttachmentLayoutParent.getVisibility() == View.VISIBLE;
    }

    public String getPathForUri(Uri uri) {
        if ("file".equals(uri.getScheme())) {
            return uri.getPath();
        }
        return null;
    }

    public void clearInput() {
        photoFile = null;
        imageUri = null;
        hidePreview();
    }

    public interface ImagePickCallback {
        void onAttachmentPicked(Uri imageUri, AttachmentType audio);
        void onAttachmentRemoved();

        void onAttachmentLayoutShow();

        void onAttachmentLayoutHide();
    }

    public enum AttachmentType {
        Image,Audio
    }
}
