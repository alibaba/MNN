package com.mnn.llm;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {
    private Chat mChat;
    private Intent mIntent;
    private Button mLoadButton;
    private TextView mModelInfo;
    private Spinner mSpinnerModels;
    private final String mSearchPath = "/data/local/tmp/mnn-llm/";
    private String mModelName = "qwen-1.8b-int4";
    private String mModelDir = mSearchPath + mModelName;

    @SuppressLint("HandlerLeak")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mIntent = new Intent(this, Conversation.class);
        mModelInfo = findViewById(R.id.model_info);
        mLoadButton = findViewById(R.id.load_button);
        mSpinnerModels = findViewById(R.id.spinner_models);

        mModelDir = this.getCacheDir() + "/" + mModelName;
        populateFoldersSpinner();
        mSpinnerModels.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (position > 0) {
                    mModelName = (String) parent.getItemAtPosition(position);
                    mModelInfo.setText(String.format("选择模型：%s", mModelName));
                    mModelInfo.setVisibility(View.VISIBLE);
                    mModelDir = mSearchPath + mModelName + "/config.json";
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });
    }

    public void loadModel(View view) {
        onCheckModels();
        mLoadButton.setClickable(false);
        mLoadButton.setBackgroundColor(Color.parseColor("#2454e4"));
        mLoadButton.setText("模型加载中 ...");

        new Thread(() -> {
            mChat = new Chat();
            mChat.Init(mModelDir);
            runOnUiThread(() -> {
                mIntent.putExtra("chat", mChat);
                startActivity(mIntent);
            });
        }).start();
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
            mModelInfo.setText(String.format("%s模型文件就绪，模型加载中", mModelName));
            mLoadButton.setText("加载模型");
        }
    }

    private boolean checkModelsReady() {
        File dir = new File(mModelDir);
        return dir.exists();
    }

    private ArrayList<String> getFoldersList(String path) {
        File directory = new File(path);
        File[] files = directory.listFiles();
        ArrayList<String> foldersList = new ArrayList<>();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    foldersList.add(file.getName());
                }
            }
        }
        return foldersList;
    }

    private void populateFoldersSpinner() {
        ArrayList<String> folders = getFoldersList("/data/local/tmp/mnn-llm");
        folders.add(0, getString(R.string.spinner_prompt));
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, folders);
        mSpinnerModels.setAdapter(adapter);
    }
}