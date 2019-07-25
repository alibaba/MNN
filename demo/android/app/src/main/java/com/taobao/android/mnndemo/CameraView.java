package com.taobao.android.mnndemo;

import android.app.Activity;
import android.content.Context;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class CameraView extends SurfaceView implements SurfaceHolder.Callback, Camera.PreviewCallback {
    private static final String TAG = "AiNNDemo";

    private static final int MINIMUM_PREVIEW_SIZE = 320;
    private static final int PREVIEW_CALLBACK_FREQUENCE = 5;
    private Camera mCamera;
    private Camera.Parameters mParams;
    private Camera.Size mPreviewSize;
    private PreviewCallback mPreviewCallback;
    private int mOrientationAngle;
    private int previewCallbackCount;

    public CameraView(Context context) {
        this(context, null);
    }

    public CameraView(Context context, AttributeSet attrs) {
        super(context, attrs);

        SurfaceHolder holder = getHolder();
        holder.addCallback(this);
    }

    public void setPreviewCallback(CameraView.PreviewCallback previewCallback) {
        mPreviewCallback = previewCallback;
    }

    private void openCamera(SurfaceHolder holder) {
        // release Camera, if not release camera before call camera, it will be locked
        releaseCamera();

        mCamera = Camera.open();

        setCameraDisplayOrientation((Activity) getContext(), Camera.CameraInfo.CAMERA_FACING_BACK, mCamera);

        mParams = mCamera.getParameters();
        mPreviewSize = getPropPreviewSize(mParams.getSupportedPreviewSizes(), 800, 800);
        mParams.setPreviewSize(mPreviewSize.width, mPreviewSize.height);

        mParams.setPreviewFormat(ImageFormat.NV21);
        mParams.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);

        mCamera.setParameters(mParams);

        if (mPreviewCallback != null) {
            mPreviewCallback.onGetPreviewOptimalSize(mPreviewSize.width, mPreviewSize.height);
        }

        try {
            mCamera.setPreviewDisplay(holder);
        } catch (IOException e) {
            e.printStackTrace();
        }
        mCamera.setPreviewCallback(this);
        mCamera.startPreview();
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

    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {
        Log.i("AiNNDemo", "surfaceCreated");

        openCamera(surfaceHolder);
    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i1, int i2) {
        Log.i(TAG, "surfaceChanged");
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
        Log.i("AiNNDemo", "surfaceDestroyed");
    }

    public void setCameraDisplayOrientation(Activity activity,
                                            int cameraId, android.hardware.Camera camera) {
        android.hardware.Camera.CameraInfo info =
                new android.hardware.Camera.CameraInfo();
        android.hardware.Camera.getCameraInfo(cameraId, info);
        int rotation = activity.getWindowManager().getDefaultDisplay()
                .getRotation();
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

        mOrientationAngle = info.orientation;

        int result;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (info.orientation + degrees) % 360;
            result = (360 - result) % 360;  // compensate the mirror
        } else {  // back-facing
            result = (info.orientation - degrees + 360) % 360;
        }
        camera.setDisplayOrientation(result);
    }

    public void onResume() {
        openCamera(getHolder());
    }

    public void onPause() {
        releaseCamera();
    }

    @Override
    public void onPreviewFrame(byte[] bytes, Camera camera) {

        previewCallbackCount++;
        if (previewCallbackCount % PREVIEW_CALLBACK_FREQUENCE != 0) {
            return;
        }
        previewCallbackCount = 0;

        mPreviewCallback.onPreviewFrame(bytes, mPreviewSize.width, mPreviewSize.height, mOrientationAngle);
    }

    /**
     * Given choices supported by a camera, chooses the smallest one whose
     * width and height are at least as large as the minimum of both, or an exact match if possible.
     *
     * @param choices   The list of sizes that the camera supports for the intended output class
     * @param minWidth  The minimum desired width
     * @param minHeight The minimum desired height
     * @return The optimal size, or an arbitrary one if none were big enough
     */
    private Camera.Size getPropPreviewSize(List<Camera.Size> choices, int minWidth, int minHeight) {
        final int minSize = Math.max(Math.min(minWidth, minHeight), MINIMUM_PREVIEW_SIZE);

        final List<Camera.Size> bigEnough = new ArrayList<Camera.Size>();
        final List<Camera.Size> tooSmall = new ArrayList<Camera.Size>();

        for (Camera.Size option : choices) {
            if (option.width == minWidth && option.height == minHeight) {
                return option;
            }

            if (option.height >= minSize && option.width >= minSize) {
                bigEnough.add(option);
            } else {
                tooSmall.add(option);
            }
        }

        if (bigEnough.size() > 0) {
            Camera.Size chosenSize = Collections.min(bigEnough, new CompareSizesByArea());
            return chosenSize;
        } else {
            return choices.get(0);
        }
    }


    public interface PreviewCallback {
        void onGetPreviewOptimalSize(int optimalWidth, int optimalHeight);

        void onPreviewFrame(byte[] data, int imageWidth, int imageHeight, int angle);
    }

    // Compares two size based on their areas.
    static class CompareSizesByArea implements Comparator<Camera.Size> {
        @Override
        public int compare(final Camera.Size lhs, final Camera.Size rhs) {
            return Long.signum(
                    (long) lhs.width * lhs.height - (long) rhs.width * rhs.height);
        }
    }

}
