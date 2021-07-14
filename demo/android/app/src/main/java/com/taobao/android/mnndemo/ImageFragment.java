package com.taobao.android.mnnapp;

import android.Manifest;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.support.v4.content.FileProvider;
import android.support.v4.os.EnvironmentCompat;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.PopupWindow;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNImageProcess;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.Common;
import com.taobao.android.utils.TxtFileReader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static android.app.Activity.RESULT_OK;

public class ImageFragment extends Fragment implements View.OnClickListener, AdapterView.OnItemSelectedListener {
    private final String TAG = "ImageFragment";
    private final int CAMERA_PERMISSION = 800;
    private final int WRITE_EXTERNAL_PERMISSION = 801;

    private String mDemoImage;
    private String[] mModelFileName;
    private String mWordsFileName_1000;
    private String mWordsFileName_1001;

    private String[] mMNNFileName;
    private String[] mMNNModelPath;

    private List<String> mWords;
    private List<String> mWords_1000;
    private List<String> mWords_1001;

    private View rootView;

    private ImageView mImageView;
    private TextView mSelectModelView;
    // private TextView mPredict;
    private TextView mRepeatPredict;
    private EditText mRepeatTimes;
    private TextView mResultText;
    private TextView mTimeText;
    private Bitmap mBitmap;


    private Spinner mModelSelectSpinner;
    private PopupWindow popupWindow;
    private TextView popupAlbum;
    private TextView popupCamera;
    private TextView popupCancel;

    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;

    private int[] mInputWidth;
    private int[] mInputHeight;

    // current using model
    private int mSelectedModelIndex = 0;
    private int repeat_times = 1;

    private final int POPUP_WINDOW_YSHIFT = 200;

    private final int VIA_ALBUM = 802;
    private final int VIA_CAMERA = 803;
    private final int USE_CROP = 804;
    private Uri mCutUri;
    private Uri mCameraUri;

    private void prepareModels() {

        for (int i = 0; i < mMNNModelPath.length; ++i) {
            mMNNModelPath[i] = getActivity().getCacheDir() + "/" + mMNNFileName[i];
        }

        try {
            for (int i = 0; i < mModelFileName.length; ++i) {
                Common.copyAssetResource2File(getActivity().getBaseContext(), mModelFileName[i], mMNNModelPath[i]);
            }
            if (mWords == null) {
                mWords_1000 = TxtFileReader.getUniqueUrls(getActivity().getBaseContext(), mWordsFileName_1000, Integer.MAX_VALUE);
                mWords_1001 = TxtFileReader.getUniqueUrls(getActivity().getBaseContext(), mWordsFileName_1001, Integer.MAX_VALUE);
                mWords = mWords_1000;
            }
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    private void prepareMNNNet() {
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
        mNetInstance = MNNNetInstance.createFromFile(mMNNModelPath[mSelectedModelIndex]);

        // create session with config
        MNNNetInstance.Config config = new MNNNetInstance.Config();
        // set threads
        config.numThread = 4;
        // set CPU/GPU
        config.forwardType = MNNForwardType.FORWARD_CPU.type;

        // mConfig.saveTensors;
        mSession = mNetInstance.createSession(config);

        // get input tensor
        mInputTensor = mSession.getInput(null);
    }

    private class NetPrepareTask extends AsyncTask<String, Void, String> {
        protected String doInBackground(String... tasks) {
            prepareModels();
            prepareMNNNet();
            return "success";
        }

        protected void onPostExecute(String result) {

//            mPredict.setText(R.string.inference);
//            mPredict.setClickable(true);
            mRepeatPredict.setText(R.string.inference);
            mRepeatPredict.setClickable(true);
        }
    }

    private class ImageProcessResult {
        public String result;
        public double avgTimeCost;
        public double timeCostStd;
    }

    private class ImageProcessTask extends AsyncTask<String, Void, ImageProcessResult> {

        protected ImageProcessResult doInBackground(String... tasks) {
            /*
             *  convert data to input tensor
             */
            final MNNImageProcess.Config config = new MNNImageProcess.Config();
            // normalization params
            //if (mSelectedModelIndex != 3) {
            if (!mModelFileName[mSelectedModelIndex].toLowerCase().contains("squeezenet")) {
                // not SqueezeNet
                // normalization params
                config.mean = new float[]{103.94f, 116.78f, 123.68f};
                config.normal = new float[]{0.017f, 0.017f, 0.017f};
            }
            // input data format
            config.dest = MNNImageProcess.Format.BGR;
            // bitmap transform
            Matrix matrix = new Matrix();
            matrix.postScale(mInputWidth[mSelectedModelIndex] / (float) mBitmap.getWidth(), mInputHeight[mSelectedModelIndex] / (float) mBitmap.getHeight());
            matrix.invert(matrix);

//            MNNImageProcess.convertBitmap(mBitmap, mInputTensor, config, matrix);
            final double[] timeCostArray = new double[repeat_times];
            double totalTime = 0.0f;

            for (int i = 0; i < repeat_times; ++i) {
                MNNImageProcess.convertBitmap(mBitmap, mInputTensor, config, matrix);
                final long startTimestamp = System.nanoTime();
                /**
                 * inference
                 */
                mSession.run();

                final long endTimestamp = System.nanoTime();
                timeCostArray[i] = (endTimestamp - startTimestamp) / 1000000.0;
                totalTime += timeCostArray[i];
            }

            double avgTimeCost = totalTime / repeat_times;
            double timeStd = 0.0;
            double totalSquare = 0.0;
            for (int i = 0; i < repeat_times; ++i) {
                Log.d(TAG, "Time " + i + " : " + timeCostArray[i]);
                totalSquare += (timeCostArray[i] - avgTimeCost) * (timeCostArray[i] - avgTimeCost);
            }
            if (repeat_times > 1) {
                timeStd = Math.sqrt(totalSquare / (repeat_times - 1));
            }

            /**
             * also you can use runWithCallback if you concern about some outputs of the middle layers,
             * this method execute inference and also return middle Tensor outputs synchronously.
             */
            // MNNNetInstance.Session.Tensor[] tensors =  mSession.runWithCallback(new String[]{"conv1"});
            /**
             * get output tensor
             */
            MNNNetInstance.Session.Tensor output = mSession.getOutput(null);
            // get float results
            float[] result = output.getFloatData();

            // 显示结果
            List<Map.Entry<Integer, Float>> maybes = new ArrayList<>();
            Log.d(TAG, "Size of result: " + result.length);
            for (int i = 0; i < result.length; i++) {
                float confidence = result[i];
                if (confidence > 0.01) {
                    maybes.add(new AbstractMap.SimpleEntry<Integer, Float>(i, confidence));
                }
            }

            Log.i(Common.TAG, "Inference result size=" + result.length + ", maybe=" + maybes.size());

            Collections.sort(maybes, new Comparator<Map.Entry<Integer, Float>>() {
                @Override
                public int compare(Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2) {
                    if (Math.abs(o1.getValue() - o2.getValue()) <= Float.MIN_NORMAL) {
                        return 0;
                    }
                    return o1.getValue() > o2.getValue() ? -1 : 1;
                }
            });

            final StringBuilder sb = new StringBuilder();
            for (Map.Entry<Integer, Float> entry : maybes) {
                sb.append(getString(R.string.category) + ":");
                String category = mWords.get(entry.getKey());
                // final String res = category.length() >= 10 ? category.substring(10) : category;
                sb.append(String.format("%-10s", category));
                sb.append(" " + getString(R.string.confidence) + ":");
                sb.append(new DecimalFormat("0.00").format(entry.getValue()));
                sb.append("\n");
            }

            final ImageProcessResult processResult = new ImageProcessResult();
            processResult.result = sb.toString();
            processResult.avgTimeCost = avgTimeCost;
            processResult.timeCostStd = timeStd;
            return processResult;
        }

        protected void onPostExecute(ImageProcessResult result) {
            mResultText.setText(result.result);
            mTimeText.setText(getString(R.string.avg_cost_time) + ": " +
                    new DecimalFormat("0.00").format(result.avgTimeCost)
                    + "±" + new DecimalFormat("0.00").format(result.timeCostStd) + "ms");
        }
    }

    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        rootView = inflater.inflate(R.layout.activity_image, null);
        return rootView;
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);

        mDemoImage = getString(R.string.demo_image_path);
        mModelFileName = getActivity().getResources().getStringArray(R.array.model_filename_list);
        mWordsFileName_1000 = getString(R.string.words_1000);
        mWordsFileName_1001 = getString(R.string.words_1001);
        mMNNFileName = getActivity().getResources().getStringArray(R.array.model_mnn_list);
        mMNNModelPath = new String[mMNNFileName.length];
        mInputWidth = getActivity().getResources().getIntArray(R.array.input_width);
        mInputHeight = getActivity().getResources().getIntArray(R.array.input_height);

        mImageView = getActivity().findViewById(R.id.imageView);
        mSelectModelView = getActivity().findViewById(R.id.select_model);
//        mPredict = getActivity().findViewById(R.id.predict);
        mRepeatPredict = getActivity().findViewById(R.id.repeat_predict);
        mRepeatTimes = getActivity().findViewById(R.id.repeat_times);
        mResultText = getActivity().findViewById(R.id.editText);
        mTimeText = getActivity().findViewById(R.id.timeText);

        mModelSelectSpinner = getActivity().findViewById(R.id.modelSelectSpinner);
        mModelSelectSpinner.setOnItemSelectedListener(this);

        mImageView.setOnClickListener(this);
//        mPredict.setOnClickListener(this);
        mRepeatPredict.setOnClickListener(this);

        // show image
        AssetManager am = getActivity().getAssets();
        try {
            final InputStream picStream = am.open(mDemoImage);
            mBitmap = BitmapFactory.decodeStream(picStream);
            picStream.close();
            mImageView.setImageBitmap(mBitmap);
        } catch (Throwable t) {
            t.printStackTrace();
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (getActivity().checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
            }
            if (getActivity().checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, WRITE_EXTERNAL_PERMISSION);
            }
        }

        mSelectModelView.setText(R.string.select_model);
//        mPredict.setText(R.string.prepare_net);
//        mPredict.setClickable(false);
        mRepeatPredict.setText(R.string.prepare_net);
        mRepeatPredict.setClickable(false);
        final NetPrepareTask prepareTask = new NetPrepareTask();
        prepareTask.execute("");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            return;
        }
        else {
            Toast.makeText(getActivity(), getString(R.string.not_enough_permission), Toast.LENGTH_SHORT).show();
        }
    }

    private void showPopupWindow() {
        View view = LayoutInflater.from(getContext()).inflate(R.layout.popup_window, null);
        popupWindow = new PopupWindow(view, ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, true);
        popupWindow.setBackgroundDrawable(new BitmapDrawable());
        popupWindow.setOutsideTouchable(true);
        // View rootView = LayoutInflater.from(getContext()).inflate(R.layout.activity_image, null);
        popupWindow.showAtLocation(rootView, Gravity.BOTTOM, 0, POPUP_WINDOW_YSHIFT);

        popupAlbum = view.findViewById(R.id.pop_album);
        popupCamera = view.findViewById(R.id.pop_camera);
        popupCancel = view.findViewById(R.id.pop_cancel);
        popupAlbum.setOnClickListener(this);
        popupCamera.setOnClickListener(this);
        popupCancel.setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.imageView:
                if (popupWindow == null) {
                    showPopupWindow();
                }
                else {
                    popupWindow.showAtLocation(rootView, Gravity.BOTTOM, 0, POPUP_WINDOW_YSHIFT);
                }
                break;
            case R.id.pop_album:
                popupWindow.dismiss();
                openAlbum();
                break;
            case R.id.pop_camera:
                popupWindow.dismiss();
                openCamera();
                break;
            case R.id.pop_cancel:
                popupWindow.dismiss();
                break;
            case R.id.repeat_predict:
                if (mBitmap == null) {
                    return;
                }
                repeat_times = Integer.parseInt(mRepeatTimes.getText().toString());
                mResultText.setText(R.string.inferencing);
                mTimeText.setText("");
                ImageProcessTask imageProcessTask = new ImageProcessTask();
                imageProcessTask.execute("");
                break;
        }
    }

    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
        if (adapterView.getId() == mModelSelectSpinner.getId()) {
            mSelectedModelIndex = i;
            // tensorflow官方的模型都有1001个类，多了第一类的背景类
            if (mMNNFileName[mSelectedModelIndex].toLowerCase().contains("tf")) {
                mWords = mWords_1001;
            }
            else {
                mWords = mWords_1000;
            }
        }
//        mPredict.setText(R.string.prepare_net);
//        mPredict.setClickable(false);
        mRepeatPredict.setText(R.string.prepare_net);
        mRepeatPredict.setClickable(false);
        final NetPrepareTask prepareTask = new NetPrepareTask();
        prepareTask.execute("");
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {}

        @Override
    public void onDestroy() {

        /**
         * instance release
         */
        if (mNetInstance != null) {
            mNetInstance.release();
            mNetInstance = null;
        }

        super.onDestroy();
    }

    private void openAlbum() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        // 如果第二个参数大于或等于0，那么当用户操作完成后会返回到本程序的onActivityResult方法
        startActivityForResult(intent, VIA_ALBUM);
    }

    private void openCamera() {
        Uri outputUri = createImageUri();
        mCameraUri = outputUri;
        // 启动相机程序
        Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
        intent.putExtra(MediaStore.EXTRA_OUTPUT, outputUri);
        startActivityForResult(intent, VIA_CAMERA);
    }

    private Intent crop(Uri uri) {
        Log.d(TAG, "Uri in cropFromAlbum: " + uri);
        Intent intent = new Intent("com.android.camera.action.CROP");
        // Uri outputUri = createImageUri();
        // 为了避免相册中看到大量裁剪图片，还是放在缓存文件夹
        File imagePath = new File(getActivity().getExternalCacheDir(), "cut.jpeg");
        if (imagePath.exists()) {
            imagePath.delete();
        }
        try {
            imagePath.createNewFile();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        Uri outputUri = Uri.fromFile(imagePath);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        }

        mCutUri = outputUri;
        Log.d(TAG, "mCutUri: " + mCutUri);
        // crop为true是设置在开启的intent中设置显示的view可以剪裁
        intent.putExtra("crop", true);
        // 支持缩放
        intent.putExtra("scale", true);
        // aspectX,aspectY 是宽高的比例，这里设置正方形
        intent.putExtra("aspectX", 1);
        intent.putExtra("aspectY", 1);
        //intent.putExtra("outputX", 224);
        //intent.putExtra("outputY", 224);

        // 如果图片过大，会导致oom，这里设置为false
        intent.putExtra("return-data", false);
        if (uri != null) {
            intent.setDataAndType(uri, "image/*");
        }
        if (outputUri != null) {
            intent.putExtra(MediaStore.EXTRA_OUTPUT, outputUri);
        }
        intent.putExtra("noFaceDetection", true);
        // 输出格式
        intent.putExtra("outputFormat", Bitmap.CompressFormat.JPEG.toString());

        return intent;
    }

    /**
     * 创建图片地址uri,用于保存拍照后的照片
     */
    private Uri createImageUri() {
        String status = Environment.getExternalStorageState();
        // 判断是否有SD卡,优先使用SD卡存储,当没有SD卡时使用手机存储
        if (status.equals(Environment.MEDIA_MOUNTED)) {
            return getActivity().getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, new ContentValues());
        }
        else {
            return getActivity().getContentResolver().insert(MediaStore.Images.Media.INTERNAL_CONTENT_URI, new ContentValues());
        }
    }

    /**
     * 创建图片地址uri,用于保存裁剪后的照片
     */
    private Uri createCropUri() {
        String status = Environment.getExternalStorageState();
        // 判断是否有SD卡,优先使用SD卡存储,当没有SD卡时使用手机存储
        if (status.equals(Environment.MEDIA_MOUNTED)) {
            return getActivity().getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, new ContentValues());
        }
        else {
            return getActivity().getContentResolver().insert(MediaStore.Images.Media.INTERNAL_CONTENT_URI, new ContentValues());
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        Log.d(TAG, "resultCode: " + resultCode);
        // 用户操作完成，结果码返回是-1，即RESULT_OK
        if (resultCode == RESULT_OK) {
            switch (requestCode) {
                case VIA_ALBUM:
                    startActivityForResult(crop(data.getData()), USE_CROP);
                    break;
                case VIA_CAMERA:
                    startActivityForResult(crop(mCameraUri), USE_CROP);
                    break;
                case USE_CROP:
                    Log.d(TAG, "CROP--mCutUri: " + mCutUri);
                    ContentResolver cr = getActivity().getContentResolver();
                    try {
                        // 获取图片
                        mBitmap = BitmapFactory.decodeStream(cr.openInputStream(mCutUri));
                        mImageView.setImageBitmap(mBitmap);
                    } catch (FileNotFoundException e) {
                        Log.e(TAG, e.getMessage(), e);
                    }
                    mTimeText.setText("");
                    mResultText.setText("");
                    break;
            }
        }
        else {
            // 操作错误或没有选择图片
            Log.i(TAG, "operation error or cancel");
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
