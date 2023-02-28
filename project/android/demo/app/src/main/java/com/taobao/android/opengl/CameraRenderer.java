package com.taobao.android.opengl;

import android.content.Context;
import android.content.res.Configuration;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.widget.TextView;

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.mnndemo.R;
import com.taobao.android.utils.Common;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class CameraRenderer extends GLSurfaceView implements
        GLSurfaceView.Renderer,
        SurfaceTexture.OnFrameAvailableListener{

    private Context mContext;

    private Camera mCamera = null;
    private SurfaceTexture mSurfaceTexture;

    private final OESTexture mCameraTexture = new OESTexture();
    private final Shader mOffscreenShader = new Shader();
    private int mWidth, mHeight;
    private boolean updateTexture = false;

    private ByteBuffer mVertices;
    private float[] mTransformM = new float[16];
    private float[] mOrientationM = new float[16];
    private float[] mRatio = new float[2];

    private TextView mTextView = null;
    private ByteBuffer mPreviewImageBuf = null;

    private int mCameraID = Camera.CameraInfo.CAMERA_FACING_FRONT;
    private int mCameraWidth = 0;
    private int mCameraHeight = 0;

    private byte[] mPreviewData = null;
    private MnnThread mnnThread = new MnnThread();
    private static volatile boolean mOutputDateReady = false;
    private static boolean mThreadReady;

    private MNNNetInstance.Config mConfig;// session config


    private int netInputWidth = 257;
    private int netInputHeight = 257;

    private int netOutputWidth = 257;
    private int netOutputHeight = 257;

    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;
    private MNNNetInstance.Session.Tensor mOutputTensor;

    private String mMobileModelFileName;
    private String mMobileModelPath;

    public void prepareModels() {
        mMobileModelFileName = "Portrait/Portrait.tflite.mnn";
        mMobileModelPath = mContext.getCacheDir() + "/Portrait.tflite.mnn";

        try {
            Common.copyAssetResource2File(mContext, mMobileModelFileName, mMobileModelPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    public void prepareNet() {
        if (mNetInstance != null) {
            return;
            // mNetInstance.release();
        }
        prepareModels();
        // create net instance
        mNetInstance = MNNNetInstance.createFromFile(mMobileModelPath);

        // create session
        mConfig = new MNNNetInstance.Config();
        mConfig.numThread = 4;
        mConfig.forwardType = MNNForwardType.FORWARD_OPENGL.type;

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

    private final Camera.PreviewCallback mPreviewCB = new Camera.PreviewCallback() {
        @Override
        public void onPreviewFrame(byte[] data, Camera camera) {

            //thread init
            if (!mThreadReady) {
                mnnThread.start();
                mOutputDateReady = true;
                mThreadReady = true;
            }
            if (mOutputDateReady) {
                Log.d(Common.TAG, "set input data !!!");
                mOutputDateReady = false;
                while(!mOutputDateReady){
                    Log.d(Common.TAG, "wait mnn finish !!!");
                }
                Log.d(Common.TAG, "get output data !!!");
            }
            camera.addCallbackBuffer(data);
        }
    };

    public class MnnThread extends Thread {
        public void run(){
            prepareNet();
            while (true) {
                if (mOutputDateReady == false){
                    Log.d(Common.TAG, "start session run !!!");
                    mSession.run();
                    mOutputDateReady = true;
                    Log.d(Common.TAG, "end session run !!!");
                }
            }
        }
    }

    public CameraRenderer(Context context) {
        super(context);
        mContext = context;
        init();
    }

    public CameraRenderer(Context context, AttributeSet attrs){
        super(context, attrs);
        mContext = context;
        init();
    }

    public void setContext(Context context) {
        mContext = context;
    }

    public void setTextView(TextView tv) {
        mTextView = tv;
    }

    public void init(){
        final byte VERTICES_COORDS[] = {-1, 1, -1, -1, 1, 1, 1, -1};
        mVertices = ByteBuffer.allocateDirect(4 * 2);
        mVertices.put(VERTICES_COORDS).position(0);

        setPreserveEGLContextOnPause(true);
        setEGLContextClientVersion(2);
        setRenderer(this);
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
    }

    @Override
    public synchronized void onFrameAvailable(SurfaceTexture surfaceTexture){
        updateTexture = true;
        requestRender();
    }

    @Override
    public synchronized void onSurfaceCreated(GL10 gl, EGLConfig config) {
        try {
            mOffscreenShader.setProgram(R.raw.vshader, R.raw.fshader, mContext);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    public synchronized void onSurfaceChanged(GL10 gl, int width, int height) {

        if(mCamera != null){
            updateTexture = false;
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;

        }

        Log.e(Common.TAG, "onSurfaceChanged: width " + width + ", height " + height);
        mWidth = width;
        mHeight= height;

        mCameraTexture.init();

        SurfaceTexture oldSurfaceTexture = mSurfaceTexture;
        mSurfaceTexture = new SurfaceTexture(mCameraTexture.getTextureId());

        mSurfaceTexture.setOnFrameAvailableListener(this);
        if(oldSurfaceTexture != null){
            oldSurfaceTexture.release();
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            try {
                mCamera = Camera.open(findFrontCamera());
            } catch (RuntimeException e) {
                Log.e(Common.TAG, "failed to open: " + mCameraID + e.getLocalizedMessage());
            }
        }

        try {
            mCamera.setPreviewTexture(mSurfaceTexture);
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        Camera.Parameters param = mCamera.getParameters();
        List<Size> psize = param.getSupportedPreviewSizes();

        if (psize.size() > 0) {
            mCameraWidth = 1920;
            mCameraHeight = 1080;
            param.setPreviewSize(mCameraWidth, mCameraHeight);
        }

        if (mContext.getResources().getConfiguration().orientation == Configuration.ORIENTATION_PORTRAIT) {
            Matrix.setRotateM(mOrientationM, 0, 90.0f, 0f, 0f, 1f);
            mRatio[1] = 1;//mCameraWidth *1.0f/height;
            mRatio[0] = 1;//mCameraHeight *1.0f/width;
        } else {
            Matrix.setRotateM(mOrientationM, 0, 0.0f, 0f, 0f, 1f);
            mRatio[1] = 1;//mCameraHeight *1.0f/height;
            mRatio[0] = 1;//mCameraWidth *1.0f/width;
        }

        mPreviewData = new byte[mCameraWidth * mCameraHeight * 3 / 2 + 4084];
        mPreviewImageBuf = ByteBuffer.allocateDirect(mCameraWidth * mCameraHeight * 3 / 2 + 4084);

        mCamera.setParameters(param);
        mCamera.setPreviewCallbackWithBuffer(mPreviewCB);
        mCamera.addCallbackBuffer(mPreviewData);
        mCamera.startPreview();

        requestRender();
    }

    @Override
    public synchronized void onDrawFrame(GL10 gl) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glClearColor(0, 0, 0, 1);
        GLES20.glClearDepthf(1);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT | GLES20.GL_STENCIL_BUFFER_BIT);
        GLES20.glDisable(GLES20.GL_DEPTH_TEST);

        if (updateTexture) {
            mSurfaceTexture.updateTexImage();
            updateTexture = false;
            GLES20.glViewport(0, 0, mWidth, mHeight);
            mOffscreenShader.useProgram();

            mSurfaceTexture.getTransformMatrix(mTransformM);
            int uTransformM = mOffscreenShader.getHandle("uTransformM");
            int uOrientationM = mOffscreenShader.getHandle("uOrientationM");
            int uRatioV = mOffscreenShader.getHandle("ratios");
            int uTexID = mOffscreenShader.getHandle("sTexture");

            GLES20.glUniformMatrix4fv(uTransformM, 1, false, mTransformM, 0);
            GLES20.glUniformMatrix4fv(uOrientationM, 1, false, mOrientationM, 0);
            GLES20.glUniform2fv(uRatioV, 1, mRatio, 0);
            GLES20.glUniform1i(uTexID, 0);

            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, mCameraTexture.getTextureId());
            renderVertices(mOffscreenShader.getHandle("aPosition"));

            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0);
            GLES20.glUseProgram(0);
        }
    }

    private void renderVertices(int aPosition){
        GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_BYTE, false, 0, mVertices);
        GLES20.glEnableVertexAttribArray(aPosition);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(aPosition);
    }

    public void onDestroy(){
        updateTexture = false;
        mSurfaceTexture.release();
        if(mCamera != null){
            mCamera.stopPreview();
            mCamera.setPreviewCallback(null);
            mCamera.release();
        }
        mCamera = null;
    }

    public static int findFrontCamera(){
        int cameraCount = Camera.getNumberOfCameras();
        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
        for(int camIdx = 0; camIdx < cameraCount; camIdx++){
            Camera.getCameraInfo(camIdx, cameraInfo);
            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT){
                return camIdx;
            }
        }
        return -1;
    }
}
