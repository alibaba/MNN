package com.mnn.llm;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_CODE = 100;
    private Chat mChat;
    private Intent mIntent;
    private Button mLoadButton;
    private TextView mModelInfo;
    private String mSearchPath ;
    private String mModelName = "qwen-1.8b-int4";
    private String mModelDir = mSearchPath + mModelName;
    private ProgressBar mProgressBar;

    @SuppressLint("HandlerLeak")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //copy images
        try {
            String assetDir = "image";
            String targetDir = new File(getFilesDir(), "image").getAbsolutePath();
            FileUtils.copyAssetsRecursively(this, assetDir, targetDir);
        } catch (IOException e) {
            e.printStackTrace();
        }
        mProgressBar = findViewById(R.id.loading_spinner);
        mIntent = new Intent(this, Conversation.class);
        mModelInfo = findViewById(R.id.model_info);
        File configFile = new File("/data/local/tmp/models/Qwen-VL-2B-convert-4bit-per_channel/config.json");
        if(configFile.exists()){
            mModelDir = "/data/local/tmp/models/Qwen-VL-2B-convert-4bit-per_channel/";
        }
        else{
            //there is no config file in above dir, try to find in other folders
            try{
                mModelDir = FileUtils.findConfigDir(this, "/data/local/tmp/models/");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        onCheckModels();
        new Thread(() -> {
            if(mModelDir == ""){
                mModelInfo.setText("Model Not Found!");
                mModelInfo.setVisibility(View.VISIBLE);
            }
            else{
                mProgressBar.setIndeterminate(true);
                mModelInfo.setText("Loading Qwen ViT model...");
                mModelInfo.setVisibility(View.VISIBLE);
                mChat = new Chat();
                mChat.Init(mModelDir);
                runOnUiThread(() -> {
                    mIntent.putExtra("chat", mChat);
                    startActivityForResult(mIntent, REQUEST_CODE);
                });
            }
        }).start();
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            finish();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    private void onCheckModels() {
        boolean modelReady = checkModelsReady();
        if (!modelReady) {
            try {
                mModelDir = Common.copyAssetResource2File(this, mModelName);
                modelReady = checkModelsReady();
            } catch (IOException | InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        if (!modelReady) {
            mModelInfo.setVisibility(View.VISIBLE);
            mModelInfo.setText(String.format("%sModels are already，model loading", mModelName));
            mLoadButton.setText("Load Model");
        }
    }

    private boolean checkModelsReady() {
        File dir = new File(mModelDir);
        return dir.exists();
    }


}