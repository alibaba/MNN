// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.View;
import android.widget.Toast;

import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.view.GravityCompat;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.fragment.app.Fragment;

import com.alibaba.mls.api.download.ModelDownloadManager;
import com.alibaba.mnnllm.android.chat.ChatActivity;
import com.alibaba.mnnllm.android.history.ChatHistoryFragment;
import com.alibaba.mnnllm.android.modelist.ModelListFragment;
import com.alibaba.mnnllm.android.settings.MainSettings;
import com.alibaba.mnnllm.android.update.UpdateChecker;
import com.alibaba.mnnllm.android.utils.GithubUtils;
import com.alibaba.mnnllm.android.utils.ModelUtils;
import com.google.android.material.navigation.NavigationView;
import com.techiness.progressdialoglibrary.ProgressDialog;
import java.io.File;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {

    public static final String TAG = "MainActivity";
    private ProgressDialog progressDialog;
    private DrawerLayout drawerLayout;
    private ActionBarDrawerToggle toggle;
    private ModelListFragment modelListFragment;
    private ChatHistoryFragment chatHistoryFragment;
    private UpdateChecker updateChecker;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_test);
        Toolbar toolbar = findViewById(R.id.toolbar);
        drawerLayout = findViewById(R.id.drawer_layout);
        updateChecker = new UpdateChecker(this);
        updateChecker.checkForUpdates(this, false);

        // Set up ActionBar toggle
        toggle = new ActionBarDrawerToggle(
                this, drawerLayout,
                toolbar,
                R.string.nav_open,
                R.string.nav_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();
        getSupportFragmentManager().beginTransaction()
                .replace(R.id.main_fragment_container,
                        getModelListFragment())
                .commit();
        getSupportFragmentManager().beginTransaction()
                .replace(R.id.history_fragment_container,
                        getChatHistoryFragment())
                .commit();
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                if (drawerLayout.isDrawerOpen(GravityCompat.START)) {
                    drawerLayout.closeDrawer(GravityCompat.START);
                } else {
                    finish();
                }
            }
        });
        setSupportActionBar(toolbar);
        Objects.requireNonNull(getSupportActionBar()).setDisplayHomeAsUpEnabled(true);
    }

    private Fragment getModelListFragment() {
        if (modelListFragment == null) {
            modelListFragment = new ModelListFragment();
        }
        return modelListFragment;
    }

    private Fragment getChatHistoryFragment() {
        if (chatHistoryFragment == null) {
            chatHistoryFragment = new ChatHistoryFragment();
        }
        return chatHistoryFragment;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (toggle.onOptionsItemSelected(item)) {
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    public void runModel(String destModelDir, String modelId, String sessionId) {
        Log.d(TAG, "runModel destModelDir: " + destModelDir);
        if (MainSettings.INSTANCE.isStopDownloadOnChatEnabled(this)) {
            ModelDownloadManager.getInstance(this).pauseAllDownloads();
        }
        drawerLayout.close();
        progressDialog = new ProgressDialog(this);
        progressDialog.setMessage(getResources().getString(R.string.model_loading));
        progressDialog.show();
        if (destModelDir == null) {
            destModelDir = ModelDownloadManager.getInstance(this).getDownloadedFile(modelId).getAbsolutePath();
        }
        boolean isDiffusion = ModelUtils.isDiffusionModel(modelId);
        String configFilePath = null;
        if (!isDiffusion) {
            String configFileName = "config.json";
            configFilePath = destModelDir + "/" + configFileName;
            boolean configFileExists = new File(configFilePath).exists();
            if (!configFileExists) {
                Toast.makeText(this, getString(R.string.config_file_not_found, configFilePath), Toast.LENGTH_LONG).show();
                progressDialog.dismiss();
                return;
            }
        }
        progressDialog.dismiss();
        Intent intent = new Intent(this, ChatActivity.class);
        intent.putExtra("chatSessionId", sessionId);
        if (isDiffusion) {
            intent.putExtra("diffusionDir", destModelDir);
        } else {
            intent.putExtra("configFilePath", configFilePath);
        }
        intent.putExtra("modelId", modelId);
        intent.putExtra("modelName", ModelUtils.getModelName(modelId));
        startActivity(intent);
    }

    public void onStarProject(View view) {
        GithubUtils.starProject(this);
    }

    public void onReportIssue(View view) {
        GithubUtils.reportIssue(this);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == ModelDownloadManager.REQUEST_CODE_POST_NOTIFICATIONS) {
            ModelDownloadManager.getInstance(this).startForegroundService();
        }
    }

    public void checkForUpdate() {
        updateChecker.checkForUpdates(this, true);
    }
}