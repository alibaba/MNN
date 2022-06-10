package com.taobao.android.mnndemo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNImageProcess;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.Common;
import com.taobao.android.utils.TxtFileReader;

import java.io.InputStream;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class ImageActivity extends AppCompatActivity implements View.OnClickListener {

    private final String TargetPic = "MobileNet/testcat.jpg";
    private final String MobileWordsFileName = "MobileNet/synset_words.txt";

    private List<String> mMobileTaiWords;

    private ImageView mImageView;
    private TextView mTextView;
    private TextView mResultText;
    private TextView mTimeText;
    private Bitmap mBitmap;

    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;

    private final int InputWidth = 224;
    private final int InputHeight = 224;

    private class NetPrepareTask extends AsyncTask<String, Void, String> {
        protected String doInBackground(String... tasks) {
            prepareMobileNet();
            return "success";
        }

        protected void onPostExecute(String result) {
            mTextView.setText("Start MobileNet Inference");
            mTextView.setClickable(true);
        }
    }

    private class ImageProcessResult {
        public String result;
        public float inferenceTimeCost;
    }

    private class ImageProcessTask extends AsyncTask<String, Void, ImageProcessResult> {

        protected ImageProcessResult doInBackground(String... tasks) {
            /*
             *  convert data to input tensor
             */
            final MNNImageProcess.Config config = new MNNImageProcess.Config();
            // normalization params
            config.mean = new float[]{103.94f, 116.78f, 123.68f};
            config.normal = new float[]{0.017f, 0.017f, 0.017f};
            // input data format
            config.dest = MNNImageProcess.Format.BGR;
            // bitmap transform
            Matrix matrix = new Matrix();
            matrix.postScale(InputWidth / (float) mBitmap.getWidth(), InputHeight / (float) mBitmap.getHeight());
            matrix.invert(matrix);

            MNNImageProcess.convertBitmap(mBitmap, mInputTensor, config, matrix);

            final long startTimestamp = System.nanoTime();
            /**
             * inference
             */
            mSession.run();

            final long endTimestamp = System.nanoTime();
            final float inferenceTimeCost = (endTimestamp - startTimestamp) / 1000000.0f;

            /**
             * also you can use runWithCallback if you concern about some outputs of the middle layers,
             * this method execute inference and also return middle Tensor outputs synchronously.
             */
//                MNNNetInstance.Session.Tensor[] tensors =  mSession.runWithCallback(new String[]{"conv1"});

            /**
             * get output tensor
             */
            MNNNetInstance.Session.Tensor output = mSession.getOutput(null);
            float[] result = output.getFloatData();// get float results


            // 显示结果
            List<Map.Entry<Integer, Float>> maybes = new ArrayList<>();
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
                sb.append("物体:");
                sb.append(mMobileTaiWords.get(entry.getKey()));
                sb.append(" 自信度:");
                sb.append(entry.getValue());
                sb.append("\n");
            }

            final ImageProcessResult processResult = new ImageProcessResult();
            processResult.result = sb.toString();
            processResult.inferenceTimeCost = inferenceTimeCost;
            return processResult;
        }

        protected void onPostExecute(ImageProcessResult result) {
            mResultText.setText(result.result);
            mTimeText.setText("cost time：" + result.inferenceTimeCost + "ms");
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image);

        mImageView = findViewById(R.id.imageView);
        mTextView = findViewById(R.id.textView);
        mResultText = findViewById(R.id.editText);
        mTimeText = findViewById(R.id.timeText);

        mTextView.setOnClickListener(this);

        // show image
        AssetManager am = getAssets();
        try {
            final InputStream picStream = am.open(TargetPic);
            mBitmap = BitmapFactory.decodeStream(picStream);
            picStream.close();
            mImageView.setImageBitmap(mBitmap);
        } catch (Throwable t) {
            t.printStackTrace();
        }


        mTextView.setText("prepare Mobile Net ...");
        mTextView.setClickable(false);
        final NetPrepareTask prepareTask = new NetPrepareTask();
        prepareTask.execute("");
    }


    private void prepareMobileNet() {

        String modelPath = getCacheDir() + "mobilenet_v1.caffe.mnn";

        try {
            mMobileTaiWords = TxtFileReader.getUniqueUrls(getBaseContext(), MobileWordsFileName, Integer.MAX_VALUE);
        } catch (Throwable t) {
            t.printStackTrace();
        }

        // create net instance
        mNetInstance = MNNNetInstance.createFromFile(modelPath);

        // create session with config
        MNNNetInstance.Config config = new MNNNetInstance.Config();
        config.numThread = 4;// set threads
        config.forwardType = MNNForwardType.FORWARD_CPU.type;// set CPU/GPU
        /**
         * config middle layer names, if you concern about the output of the middle layer.
         * use session.getOutput("layer name") to get the output of the middle layer.
         */
//        config.saveTensors = new String[]{"conv1"};
        mSession = mNetInstance.createSession(config);

        // get input tensor
        mInputTensor = mSession.getInput(null);
    }

    @Override
    public void onClick(View view) {
        if (mBitmap == null) {
            return;
        }

        mResultText.setText("inference result ...");
        ImageProcessTask imageProcessTask = new ImageProcessTask();
        imageProcessTask.execute("");
    }

    @Override
    protected void onDestroy() {

        /**
         * instance release
         */
        if (mNetInstance != null) {
            mNetInstance.release();
            mNetInstance = null;
        }

        super.onDestroy();
    }
}
