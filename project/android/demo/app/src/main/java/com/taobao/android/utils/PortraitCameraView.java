package com.taobao.android.utils;

import android.app.Activity;
import android.content.Context;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.util.AttributeSet;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class PortraitCameraView extends SurfaceView implements SurfaceHolder.Callback, Camera.PreviewCallback {

    public interface PreviewCallback {
        void onPreviewFrame(byte[] data, int imageWidth, int imageHeight, int angle, int degree, boolean needFlipX);
    }

    class Config {
        float rate; //宽高比
        int minPreviewWidth;
        int minPictureWidth;
    }

    class CameraSizeComparator implements Comparator<Camera.Size> {
        public int compare(Camera.Size lhs, Camera.Size rhs) {
            if (lhs.height == rhs.height) {
                return 0;
            } else if (lhs.height > rhs.height) {
                return 1;
            } else {
                return -1;
            }
        }
    }

    private static final int IMAGE_FORMAT = ImageFormat.NV21;
    private final Config mConfig;
    private Camera mCamera;
    private int mCameraId;
    private Camera.Size mPreviewSize;
    private Camera.Size mPictureSize;
    private RotateType mPreviewRotateType;
    private int mDegree;// 屏幕翻转的角度，每次屏幕翻转重新创建时更新
    private Camera.Parameters mParams;
    private final CameraSizeComparator sizeComparator;
    private PreviewCallback mPreviewCallback;

    public PortraitCameraView(Context context) {
        this(context, null);
    }

    public PortraitCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
        mConfig = new Config();
        mConfig.minPreviewWidth = 720;
        mConfig.minPictureWidth = 720;
        mConfig.rate = 1.334f;

        sizeComparator = new CameraSizeComparator();
        mPreviewRotateType = RotateType.Rotate0;
        mCameraId = Camera.CameraInfo.CAMERA_FACING_FRONT;

        SurfaceHolder holder = getHolder();
        holder.addCallback(this);
        holder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    private void openCamera(SurfaceHolder holder, int cameraId) {
        releaseCamera(); // release Camera, if not release camera before call camera, it will be locked

        mCameraId = cameraId;
        mCamera = Camera.open(mCameraId);
        setCameraDisplayOrientation((Activity) getContext(), mCameraId, mCamera);

        mParams = mCamera.getParameters();
        mPictureSize = getPropPictureSize(mParams.getSupportedPictureSizes(), mConfig.rate, mConfig.minPictureWidth);
        mPreviewSize = getPropPreviewSize(mParams.getSupportedPreviewSizes(), mConfig.rate, mConfig.minPreviewWidth);

        mParams.setPictureSize(mPictureSize.width, mPictureSize.height);
        mParams.setPreviewSize(mPreviewSize.width, mPreviewSize.height);
        mParams.setPreviewFormat(IMAGE_FORMAT);
        if (mCameraId == Camera.CameraInfo.CAMERA_FACING_BACK) {
            mParams.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
        }
        mCamera.setParameters(mParams); // setting camera parameters
        holder.setFixedSize(mPreviewSize.width, mPreviewSize.height);
        try {
            mCamera.setPreviewDisplay(holder);
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        mCamera.setPreviewCallback(this);
        mCamera.startPreview();
        //KLog.i(TAG, "Camera id=%d preview=%dx%d picture=%dx%d", mCameraId, mPreviewSize.width, mPreviewSize.height, mPictureSize.width, mPictureSize.height);
    }

    private synchronized void releaseCamera() {
        if (mCamera != null) {
            try {
                mCamera.setPreviewCallback(null);
            } catch (Exception e) {
                e.printStackTrace();
            }
            try {
                mCamera.stopPreview();
            } catch (Exception e) {
                e.printStackTrace();
            }
            try {
                mCamera.release();
            } catch (Exception e) {
                e.printStackTrace();
            }
            mCamera = null;
        }
    }

    private void setCameraDisplayOrientation(Activity activity, int cameraId, Camera camera) {
        Camera.CameraInfo info = new Camera.CameraInfo();
        Camera.getCameraInfo(cameraId, info);
        int rotation = activity.getWindowManager().getDefaultDisplay()
                .getRotation();

        // 屏幕翻转角度，和前后摄像头无关
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
            default:
                degrees = 0;
                break;
        }

        // 固定不变：某设备前置270，后置90
        switch (info.orientation) {
            case 90:
                mPreviewRotateType = RotateType.Rotate90;// 90
                break;
            case 180:
                mPreviewRotateType = RotateType.Rotate180;
                break;
            case 270:
                mPreviewRotateType = RotateType.Rotate270;
                break;
            default:
                mPreviewRotateType = RotateType.Rotate0;
                break;
        }

        int displayDegree;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            displayDegree = (info.orientation + degrees) % 360;
            displayDegree = (360 - displayDegree) % 360;  // compensate the mirror
        } else {
            displayDegree = (info.orientation - degrees + 360) % 360;
        }
        camera.setDisplayOrientation(displayDegree);

        mDegree = degrees;
    }

    @Override
    public void onPreviewFrame(byte[] data, Camera camera) {
        mPreviewCallback.onPreviewFrame(data, mPreviewSize.width, mPreviewSize.height, mPreviewRotateType.type, mDegree, mCameraId == Camera.CameraInfo.CAMERA_FACING_FRONT);
    }

    public void setPreviewCallback(PortraitCameraView.PreviewCallback previewCallback) {
        mPreviewCallback = previewCallback;
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        openCamera(getHolder(), mCameraId);
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    public void onResume() {
        openCamera(getHolder(), mCameraId);
    }

    public void onPause() {
        releaseCamera();
    }

    public void switchCamera() {
        openCamera(getHolder(), mCameraId == Camera.CameraInfo.CAMERA_FACING_FRONT ? Camera.CameraInfo.CAMERA_FACING_BACK : Camera.CameraInfo.CAMERA_FACING_FRONT);
    }

    private Camera.Size getPropPreviewSize(List<Camera.Size> list, float th, int minWidth) {
        Collections.sort(list, sizeComparator);

        int i = 0;
        for (Camera.Size s : list) {
            if ((s.height >= minWidth) && equalRate(s, th)) {
                break;
            }
            i++;
        }
        if (i == list.size()) {
            i = 0;
        }
        return list.get(i);
    }

    private Camera.Size getPropPictureSize(List<Camera.Size> list, float th, int minWidth) {
        Collections.sort(list, sizeComparator);

        int i = 0;
        for (Camera.Size s : list) {
            if ((s.height >= minWidth) && equalRate(s, th)) {
                break;
            }
            i++;
        }
        if (i == list.size()) {
            i = 0;
        }
        return list.get(i);
    }

    private boolean equalRate(Camera.Size s, float rate) {
        float r = (float) (s.width) / (float) (s.height);
        if (Math.abs(r - rate) <= 0.03) {
            return true;
        } else {
            return false;
        }
    }

    public Camera.Size getPreviewSize() {
        return mPreviewSize;
    }
}
