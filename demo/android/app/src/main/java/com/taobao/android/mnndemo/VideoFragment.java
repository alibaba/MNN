package com.taobao.android.mnnapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Matrix;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.OrientationEventListener;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewStub;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.FrameLayout;
import android.widget.RelativeLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNImageProcess;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.Common;
import com.taobao.android.utils.TxtFileReader;

import java.text.DecimalFormat;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

public class VideoFragment extends Fragment implements AdapterView.OnItemSelectedListener {

    private final String TAG = "VideoFragment";
    private final int MAX_CLZ_SIZE = 1000;
    private final int CAMERA_PERMISSION = 800;

    private String[] mModelFileName;

    private String mWordsFileName_1000;
    private String mWordsFileName_1001;

    private String[] mMNNFileName;
    private String[] mModelPath;

    private List<String> mWords;
    private List<String> mWords_1000;
    private List<String> mWords_1001;

    // current using model
    private int mSelectedModelIndex;
    // session config
    private final MNNNetInstance.Config mConfig = new MNNNetInstance.Config();

    private CameraView mCameraView;
    private Spinner mForwardTypeSpinner;
    private Spinner mThreadNumSpinner;
    private Spinner mModelSpinner;

    private TextView mFirstResult;
    private TextView mSecondResult;
    private TextView mThirdResult;
    private TextView mTimeTextView;

    private int[] mInputWidth;
    private int[] mInputHeigth;

    HandlerThread mThread;
    Handler mHandle;

    private AtomicBoolean mLockUIRender = new AtomicBoolean(false);
    private AtomicBoolean mDrop = new AtomicBoolean(false);

    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;

    private OrientationEventListener orientationListener;

    // 0/90/180/360
    private int mRotateDegree;

    // View加载完成与否
    private boolean isViewInited;

    private void prepareModels() {
        for (int i = 0; i < mModelPath.length; ++i) {
            mModelPath[i] = getActivity().getCacheDir() + "/" + mMNNFileName[i];
        }
        try {
            for (int i = 0; i < mModelFileName.length; ++i) {
                Common.copyAssetResource2File(getActivity().getBaseContext(), mModelFileName[i], mModelPath[i]);
            }
            mWords_1000 = TxtFileReader.getUniqueUrls(getActivity().getBaseContext(), mWordsFileName_1000, Integer.MAX_VALUE);
            mWords_1001 = TxtFileReader.getUniqueUrls(getActivity().getBaseContext(), mWordsFileName_1001, Integer.MAX_VALUE);
            mWords = mWords_1000;
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    private void prepareNet() {
        if (null != mSession) {
            mSession.release();
            mSession = null;
        }
        if (mNetInstance != null) {
            mNetInstance.release();
            mNetInstance = null;
        }

        Log.d(TAG, "Use model with index " + mSelectedModelIndex);

        // create net instance
        mNetInstance = MNNNetInstance.createFromFile(mModelPath[mSelectedModelIndex]);

        // mConfig.saveTensors;
        mSession = mNetInstance.createSession(mConfig);

        // get input tensor
        mInputTensor = mSession.getInput(null);

        int[] dimensions = mInputTensor.getDimensions();
        // force batch = 1  NCHW  [batch, channels, height, width]
        dimensions[0] = 1;
        mInputTensor.reshape(dimensions);
        mSession.reshape();

        // 只有在fragment可见时才运行网络进行预测
        mLockUIRender.set(!getUserVisibleHint());
    }

    /**
     * 在openOrCloseCamera()方法中进行双重标记判断,通过后即可打开相机
    **/
    private void openOrCloseCamera() {
        if (mCameraView == null) {
            return;
        }
        Log.d(TAG, "In openOrCloseCamera : getUserVisibleHint : " + getUserVisibleHint());
        if (getUserVisibleHint() && isViewInited) {
            mCameraView.onResume();
            mCameraView.setVisibility(View.VISIBLE);
            mLockUIRender.set(false);
            Log.d(TAG, "Open camera");
        } else {
            mCameraView.onPause();
            mCameraView.setVisibility(View.GONE);
            mLockUIRender.set(true);
            Log.d(TAG, "Close camera");
        }
    }

    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        getActivity().getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getActivity().getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);

        return inflater.inflate(R.layout.activity_video, null);
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);

        // View已创建
        isViewInited = true;

        mModelFileName = getActivity().getResources().getStringArray(R.array.model_filename_list);
        mWordsFileName_1000 = getString(R.string.words_1000);
        mWordsFileName_1001 = getString(R.string.words_1001);
        mMNNFileName = getActivity().getResources().getStringArray(R.array.model_mnn_list);
        mModelPath = new String[mMNNFileName.length];
        mInputWidth = getActivity().getResources().getIntArray(R.array.input_width);
        mInputHeigth = getActivity().getResources().getIntArray(R.array.input_height);

        orientationListener = new OrientationEventListener(getActivity(),
                SensorManager.SENSOR_DELAY_NORMAL) {
            @Override
            public void onOrientationChanged(int orientation) {
                if (orientation == OrientationEventListener.ORIENTATION_UNKNOWN) {
                    // 手机平放时，检测不到有效的角度
                    return;
                }
                // 可以根据不同角度检测处理，这里只检测四个角度的改变
                orientation = (orientation + 45) / 90 * 90;
                mRotateDegree = orientation % 360;
                Log.d(TAG, "Rotate degree set to " + mRotateDegree);
            }
        };
        if (orientationListener.canDetectOrientation()) {
            orientationListener.enable();
        } else {
            orientationListener.disable();
        }

        mSelectedModelIndex = 0;
        mConfig.numThread = 4;
        mConfig.forwardType = MNNForwardType.FORWARD_CPU.type;

        // prepare mnn net models
        prepareModels();

        mForwardTypeSpinner = getActivity().findViewById(R.id.forwardTypeSpinner);
        mThreadNumSpinner = getActivity().findViewById(R.id.threadsSpinner);
        mThreadNumSpinner.setSelection(2);
        mModelSpinner = getActivity().findViewById(R.id.modelTypeSpinner);

        mFirstResult = getActivity().findViewById(R.id.firstResult);
        mSecondResult = getActivity().findViewById(R.id.secondResult);
        mThirdResult = getActivity().findViewById(R.id.thirdResult);
        mTimeTextView = getActivity().findViewById(R.id.timeTextView);

        mForwardTypeSpinner.setOnItemSelectedListener(this);
        mThreadNumSpinner.setOnItemSelectedListener(this);
        mModelSpinner.setOnItemSelectedListener(this);

        // init sub thread handle
        mLockUIRender.set(true);
        clearUIForPrepareNet();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (getActivity().checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
            } else {
                handlePreViewCallBack();
            }
        } else {
            handlePreViewCallBack();
        }

        mThread = new HandlerThread("MNNNet");
        mThread.start();
        mHandle = new Handler(mThread.getLooper());

        mHandle.post(new Runnable() {
            @Override
            public void run() {
                prepareNet();
            }
        });
        openOrCloseCamera();
    }

    /**
     * 此方法会在onCreateView(）之前执行
     * 当viewPager中fragment改变可见状态时也会调用
     * 当fragment从可见到不见，或者从不可见切换到可见，都会调用此方法
     * true表示当前页面可见，false表示不可见
     */
    @Override
    public void setUserVisibleHint(boolean isVisibleToUser) {
        super.setUserVisibleHint(isVisibleToUser);
        Log.d(TAG, "setUserVisibleHint---"+isVisibleToUser);
        openOrCloseCamera();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (CAMERA_PERMISSION == requestCode) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                handlePreViewCallBack();
            } else {
                Toast.makeText(getActivity(), getString(R.string.not_enough_permission), Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void handlePreViewCallBack() {

        ViewStub stub = getActivity().findViewById(R.id.stub);
        stub.inflate();

        mCameraView = getActivity().findViewById(R.id.camera_view);

        mCameraView.setPreviewCallback(new CameraView.PreviewCallback() {
            @Override
            public void onGetPreviewOptimalSize(int optimalWidth, int optimalHeight) {
                // adjust video preview size according to screen size
                DisplayMetrics metric = new DisplayMetrics();
                getActivity().getWindowManager().getDefaultDisplay().getMetrics(metric);
                int fixedVideoHeight = metric.widthPixels * optimalWidth / optimalHeight;

                FrameLayout layoutVideo = getActivity().findViewById(R.id.videoLayout);
                RelativeLayout.LayoutParams params = (RelativeLayout.LayoutParams) layoutVideo.getLayoutParams();
                params.height = fixedVideoHeight;
                layoutVideo.setLayoutParams(params);
            }

            @Override
            public void onPreviewFrame(final byte[] data, final int imageWidth, final int imageHeight, final int angle) {
                if (mLockUIRender.get()) {
                    return;
                }

                if (mDrop.get()) {
                    Log.w(TAG, "drop frame , net running too slow !!");
                }
                else {
                    mDrop.set(true);
                    mHandle.post(new Runnable() {
                        @Override
                        public void run() {
                            mDrop.set(false);
                            if (mLockUIRender.get()) {
                                Log.d(TAG, "return in OnPreviewFrame");
                                return;
                            }

                            // calculate corrected angle based on camera orientation and mobile rotate degree. (back camrea)
                            int needRotateAngle = (angle + mRotateDegree) % 360;

                            /*
                             *  convert data to input tensor
                             */
                            final MNNImageProcess.Config config = new MNNImageProcess.Config();
                            //if (mSelectedModelIndex != 3) {
                            if (!mModelFileName[mSelectedModelIndex].toLowerCase().contains("squeezenet")) {
                                // not squeezenet
                                // normalization params
                                config.mean = new float[]{103.94f, 116.78f, 123.68f};
                                config.normal = new float[]{0.017f, 0.017f, 0.017f};
                            }
                            // input source format
                            config.source = MNNImageProcess.Format.YUV_NV21;
                            // input data format
                            config.dest = MNNImageProcess.Format.BGR;
                            // matrix transform: dst to src
                            final Matrix matrix = new Matrix();
                            matrix.postScale(mInputWidth[mSelectedModelIndex] / (float) imageWidth, mInputHeigth[mSelectedModelIndex] / (float) imageHeight);
                            matrix.postRotate(needRotateAngle, mInputWidth[mSelectedModelIndex] / 2, mInputHeigth[mSelectedModelIndex] / 2);
                            matrix.invert(matrix);

                            MNNImageProcess.convertBuffer(data, imageWidth, imageHeight, mInputTensor, config, matrix);

                            final long startTimestamp = System.nanoTime();
                            /**
                             * inference
                             */
                            mSession.run();

                            /**
                             * get output tensor
                             */
                            MNNNetInstance.Session.Tensor output = mSession.getOutput(null);

                            float[] result = output.getFloatData();// get float results
                            final long endTimestamp = System.nanoTime();
                            final float inferenceTimeCost = (endTimestamp - startTimestamp) / 1000000.0f;

                            if (result.length > MAX_CLZ_SIZE) {
                                Log.w(TAG, "session result too big (" + result.length + "), model incorrect ?");
                            }

                            final List<Map.Entry<Integer, Float>> maybes = new ArrayList<>();
                            for (int i = 0; i < result.length; i++) {
                                float confidence = result[i];
                                if (confidence > 0.01) {
                                    maybes.add(new AbstractMap.SimpleEntry<Integer, Float>(i, confidence));
                                }
                            }

                            Collections.sort(maybes, new Comparator<Map.Entry<Integer, Float>>() {
                                @Override
                                public int compare(Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2) {
                                    if (Math.abs(o1.getValue() - o2.getValue()) <= Float.MIN_NORMAL) {
                                        return 0;
                                    }
                                    return o1.getValue() > o2.getValue() ? -1 : 1;
                                }
                            });

                            // show results on ui
                            getActivity().runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    if (maybes.size() == 0) {
                                        mFirstResult.setText(R.string.no_data);
                                        mSecondResult.setText("");
                                        mThirdResult.setText("");
                                    }
                                    if (maybes.size() > 0) {
                                        mFirstResult.setTextColor(maybes.get(0).getValue() > 0.2 ? Color.BLACK : Color.parseColor("#a4a4a4"));
                                        final Integer iKey = maybes.get(0).getKey();
                                        final Float fValue = maybes.get(0).getValue();
                                        String strWord = "unknown";
                                        if (iKey < mWords.size()) {
                                            strWord = mWords.get(iKey);
                                        }
                                        final String resKey = strWord;//.length() >= 10 ? strWord.substring(10) : strWord;
                                        //final String resKey = strWord;//mSelectedModelIndex == 2 ? strWord.length() >= 10 ? strWord.substring(10) : strWord : strWord;
                                        mFirstResult.setText(String.format("%-10s", resKey) + "：" + new DecimalFormat("0.00").format(fValue));
                                    }
                                    if (maybes.size() > 1) {
                                        final Integer iKey = maybes.get(1).getKey();
                                        final Float fValue = maybes.get(1).getValue();
                                        String strWord = "unknown";
                                        if (iKey < mWords.size()) {
                                            strWord = mWords.get(iKey);
                                        }
                                        final String resKey = strWord;//.length() >= 10 ? strWord.substring(10) : strWord;
                                        //final String resKey = strWord;//mSelectedModelIndex == 2 ? strWord.length() >= 10 ? strWord.substring(10) : strWord : strWord;
                                        mSecondResult.setText(String.format("%-10s", resKey) + "：" + new DecimalFormat("0.00").format(fValue));

                                    }
                                    if (maybes.size() > 2) {
                                        final Integer iKey = maybes.get(2).getKey();
                                        final Float fValue = maybes.get(2).getValue();
                                        String strWord = "unknown";
                                        if (iKey < mWords.size()) {
                                            strWord = mWords.get(iKey);
                                        }
                                        final String resKey = strWord;//.length() >= 10 ? strWord.substring(10) : strWord;
                                        //final String resKey = strWord;//mSelectedModelIndex == 1 ? strWord.length() >= 10 ? strWord.substring(10) : strWord : strWord;
                                        mThirdResult.setText(String.format("%-10s", resKey) + "：" + new DecimalFormat("0.00").format(fValue));
                                    }
                                    mTimeTextView.setText(String.format(getString(R.string.speed) + "：%.2f ms, %.2f FPS", inferenceTimeCost, 1000 / inferenceTimeCost));
                                    Log.d(TAG, "Finish one predict");
                                }
                            });
                        }
                    });
                }
            }
        });
    }

    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {

        // forward type
        if (mForwardTypeSpinner.getId() == adapterView.getId()) {
            if (i == 0) {
                mConfig.forwardType = MNNForwardType.FORWARD_CPU.type;
            }
            else if (i == 1) {
                mConfig.forwardType = MNNForwardType.FORWARD_OPENCL.type;
            }
            else if (i == 2) {
                mConfig.forwardType = MNNForwardType.FORWARD_OPENGL.type;
            }
            else if (i == 3) {
                mConfig.forwardType = MNNForwardType.FORWARD_VULKAN.type;
            }
        }
        // threads num
        else if (mThreadNumSpinner.getId() == adapterView.getId()) {
            String[] threadList = getResources().getStringArray(R.array.thread_list);
            mConfig.numThread = Integer.parseInt(threadList[i].split(" ")[1]);
        }
        // model index
        else if (mModelSpinner.getId() == adapterView.getId()) {
            mSelectedModelIndex = i;
            if (mMNNFileName[mSelectedModelIndex].toLowerCase().contains("tf")) {
                mWords = mWords_1001;
            }
            else {
                mWords = mWords_1000;
            }
        }

        mLockUIRender.set(true);
        clearUIForPrepareNet();

        mHandle.post(new Runnable() {
            @Override
            public void run() {
                prepareNet();
            }
        });
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {}

    private void clearUIForPrepareNet() {
        mFirstResult.setText(R.string.prepare_net);
        mSecondResult.setText("");
        mThirdResult.setText("");
        mTimeTextView.setText("");
    }

    @Override
    public void onPause() {
        mCameraView.onPause();
        super.onPause();
    }

    @Override
    public void onResume() {
        super.onResume();
        mCameraView.onResume();
    }

    @Override
    public void onStop() {
        super.onStop();
        mCameraView.onPause();
    }

    @Override
    public void onDestroy(){
        orientationListener.disable();
        mThread.interrupt();
        /**
         * instance release
         */
        mHandle.post(new Runnable() {
            @Override
            public void run() {
                if (mNetInstance != null) {
                    mNetInstance.release();
                }
            }
        });
        super.onDestroy();
    }
}
