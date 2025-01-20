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
    private String mSearchPath ;
    private String mModelName = "qwen-1.8b-int4";
    private String mModelDir = mSearchPath + mModelName;

    @SuppressLint("HandlerLeak")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            String assetDir = "models";
            String targetDir = new File(getFilesDir(), "models").getAbsolutePath();
            FileUtils.copyAssetsRecursively(this, assetDir, targetDir);
            mSearchPath = targetDir + "/";
        } catch (IOException e) {
            e.printStackTrace();
        }
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
                    mModelInfo.setText(String.format("Select Model：%s", mModelName));
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
        mLoadButton.setText("Model loading ...");

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
            mModelInfo.setText(String.format("%sModels are already，model loading", mModelName));
            mLoadButton.setText("Load Model");
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
        ArrayList<String> folders = getFoldersList(mSearchPath);
        folders.add(0, getString(R.string.spinner_prompt));
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, R.layout.spinner_dropdown_item, folders);
        mSpinnerModels.setAdapter(adapter);
    }
}