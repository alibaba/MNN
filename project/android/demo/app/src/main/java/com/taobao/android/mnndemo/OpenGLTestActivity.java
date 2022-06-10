package com.taobao.android.mnndemo;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;

import com.taobao.android.opengl.CameraRenderer;

public class OpenGLTestActivity extends AppCompatActivity {

    private TextView mTextView;
    private CameraRenderer mRenderer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);

        View decorView = getWindow().getDecorView();
        int uiOptions = View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_FULLSCREEN;
        decorView.setSystemUiVisibility(uiOptions);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN );
        setContentView(R.layout.opengl_test);

        mTextView = (TextView) findViewById(R.id.cost_time_text);
        mRenderer = (CameraRenderer)findViewById(R.id.renderer_view);
        mRenderer.setContext(this);
        mRenderer.setTextView(mTextView);

    }

    @Override
    protected void onResume() {
        super.onResume();
        mRenderer.onResume();
    }

    @Override
    protected void onPause() {
        mRenderer.onPause();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mRenderer.onDestroy();
    }
}


