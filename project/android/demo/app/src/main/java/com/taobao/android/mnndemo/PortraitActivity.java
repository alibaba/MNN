package com.taobao.android.mnndemo;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.OrientationEventListener;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.ViewStub;
import android.view.Window;
import android.view.WindowManager;
import android.widget.CompoundButton;
import android.widget.Switch;
import android.widget.Toast;

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNImageProcess;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.Common;
import com.taobao.android.utils.PermissionUtils;
import com.taobao.android.utils.PortraitCameraView;

import java.io.IOException;

import static com.taobao.android.mnn.MNNPortraitNative.nativeConvertMaskToPixelsMultiChannels;

public class PortraitActivity extends AppCompatActivity {

    private String mMobileModelFileName;

    private String mMobileModelPath;

    private MNNNetInstance.Config mConfig;// session config

    private int mRotateDegree;// 0/90/180/360

    private int netInputWidth = 257;
    private int netInputHeight = 257;

    private int netOutputWidth = 257;
    private int netOutputHeight = 257;

    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;
    private MNNNetInstance.Session.Tensor mOutputTensor;

    Paint keyPaint = new Paint();

    public void prepareModels() {

        mMobileModelFileName = "Portrait/Portrait.tflite.mnn";
        mMobileModelPath = getCacheDir() + "/Portrait.tflite.mnn";

        try {
            Common.copyAssetResource2File(getBaseContext(), mMobileModelFileName, mMobileModelPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public void prepareNet() {
        if (mNetInstance != null) {
            mNetInstance.release();
        }

        String modelPath = mMobileModelPath;

        // create net instance
        mNetInstance = MNNNetInstance.createFromFile(modelPath);

        // create session
        mConfig = new MNNNetInstance.Config();
        mConfig.numThread = 4;
        mConfig.forwardType = MNNForwardType.FORWARD_AUTO.type;

        // mConfig.outputTensors = new String[1];
        // mConfig.outputTensors[0] = "ResizeBilinear_3";

        mSession = mNetInstance.createSession(mConfig);

        // get input tensor
        mInputTensor = mSession.getInput(null);
        mOutputTensor = mSession.getOutput(null);

        netInputHeight = mInputTensor.getDimensions()[2];
        netInputWidth = mInputTensor.getDimensions()[3];

        netOutputHeight = mOutputTensor.getDimensions()[2];
        netOutputWidth = mOutputTensor.getDimensions()[3];

    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main_stub);
        detectScreenRotate();
        prepareModels();
        prepareNet();
        cameraDetect();

        keyPaint.setColor((Color.GREEN));
        keyPaint.setStyle(Paint.Style.FILL);
        keyPaint.setStrokeWidth(8);
        keyPaint.setTextSize(80);

    }

    void cameraDetect() {
        PermissionUtils.askPermission(this, new String[]{Manifest.permission.CAMERA, Manifest
                .permission.WRITE_EXTERNAL_STORAGE}, 10, initViewRunnable);
        mCameraView.switchCamera();
        mCameraView.switchCamera();
    }

    private SurfaceHolder mDrawSurfaceHolder;
    private PortraitCameraView mCameraView;

    private Runnable initViewRunnable = new Runnable() {
        @Override
        public void run() {

            ViewStub stub = (ViewStub) findViewById(R.id.PortraitStub);
            stub.setLayoutResource(R.layout.activity_portrait);
            stub.inflate();

            SurfaceView drawView = (SurfaceView) findViewById(R.id.lines_view);
            drawView.setZOrderOnTop(true);
            drawView.getHolder().setFormat(PixelFormat.TRANSPARENT);
            mDrawSurfaceHolder = drawView.getHolder();

            Switch cameraSwitch = (Switch) findViewById(R.id.cameraSwitch);
            cameraSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
                @Override
                public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                    mCameraView.switchCamera();
                }
            });

            mCameraView = ((PortraitCameraView) findViewById(R.id.camera_view));
            mCameraView.setPreviewCallback(new PortraitCameraView.PreviewCallback() {

                @Override
                public void onPreviewFrame(byte[] data, int imageWidth, int imageHeight, int angle, int degree, boolean needFlipX) {

                    MNNImageProcess.Config config = new MNNImageProcess.Config();
                    config.mean = new float[]{127.5f, 127.5f, 127.5f};
                    config.normal = new float[]{2.0f / 255.0f, 2.0f / 255.0f, 2.0f / 255.0f};
                    config.source = MNNImageProcess.Format.YUV_NV21;// input source format
                    config.dest = MNNImageProcess.Format.RGB; // input data format

                    int needRotateAngle = (angle + mRotateDegree) % 360;

                    Matrix matrix = new Matrix();
                    matrix.setScale(1.0f / imageWidth, 1.0f / imageHeight);
                    matrix.postRotate(needRotateAngle, 0.5f, 0.5f);
                    matrix.postScale(netInputWidth, netInputHeight);
                    matrix.invert(matrix);

                    MNNImageProcess.convertBuffer(data, imageWidth, imageHeight, mInputTensor, config, matrix);

                    long start = System.currentTimeMillis();
                    mSession.run();
                    float[] mask = mOutputTensor.getFloatData();// get float results
                    long detectTime = System.currentTimeMillis() - start;

                    int[] pixels = nativeConvertMaskToPixelsMultiChannels(mask, netOutputWidth * netOutputHeight);

                    final Bitmap maskBitmap = Bitmap.createBitmap(pixels, netOutputWidth, netOutputHeight, Bitmap.Config.ARGB_8888);

                    drawBitmap(maskBitmap, needFlipX, detectTime);
                }
            });

        }
    };

    void drawBitmap(Bitmap maskBitmap, boolean needFlipX, long detectTime) {

        Canvas canvas = null;
        try {
            canvas = mDrawSurfaceHolder.lockCanvas();

            if (canvas == null) {
                return;
            }
            float scaleWidth = (float) canvas.getWidth() / (float) netOutputWidth;
            float scaleHeight = (float) canvas.getHeight() / (float) netOutputHeight;
            Matrix bitmapMatrix = new Matrix();
            if (needFlipX) {
                bitmapMatrix.postScale(-scaleWidth, scaleHeight);
            } else {
                bitmapMatrix.postScale(scaleWidth, scaleHeight);
            }
            Bitmap newBM = Bitmap.createBitmap(maskBitmap, 0, 0, netOutputWidth, netOutputHeight, bitmapMatrix, true);

            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
            canvas.drawBitmap(newBM, 0, 0, keyPaint);
            String costTime = "cost time : " + detectTime + " ms ";
            canvas.drawText(costTime, 200, 100, keyPaint);
        } catch (Throwable t) {
            Log.e(Common.TAG, "Draw result error:" + t);
        } finally {
            if (canvas != null) {
                mDrawSurfaceHolder.unlockCanvasAndPost(canvas);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (10 == requestCode) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

            } else {
                Toast.makeText(this, "没有获得必要的权限", Toast.LENGTH_SHORT).show();
            }
        }
    }

    /**
     * 监听屏幕旋转
     */
    void detectScreenRotate() {
        OrientationEventListener orientationListener = new OrientationEventListener(this,
                SensorManager.SENSOR_DELAY_NORMAL) {
            @Override
            public void onOrientationChanged(int orientation) {

                if (orientation == OrientationEventListener.ORIENTATION_UNKNOWN) {
                    return;  //手机平放时，检测不到有效的角度
                }

                //可以根据不同角度检测处理，这里只检测四个角度的改变
                orientation = (orientation + 45) / 90 * 90;
                mRotateDegree = orientation % 360;
            }
        };


        if (orientationListener.canDetectOrientation()) {
            orientationListener.enable();
        } else {
            orientationListener.disable();
        }
    }

    @Override
    protected void onPause() {
        mCameraView.onPause();
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mCameraView.onResume();
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

}
