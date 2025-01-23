// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
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
import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.history.ChatHistoryFragment;
import com.alibaba.mnnllm.android.modelist.ModelListFragment;
import com.alibaba.mnnllm.android.utils.ModelUtils;
import com.google.android.material.navigation.NavigationView;
import com.techiness.progressdialoglibrary.ProgressDialog;
import java.io.File;

public class MainActivity extends AppCompatActivity {

    public static final String TAG = "MainActivity";
    private ProgressDialog progressDialog;
    private final String repoGithubUrl = "https://github.com/alibaba/MNN";
    private DrawerLayout drawerLayout;
    private ActionBarDrawerToggle toggle;
    private ModelListFragment modelListFragment;
    private ChatHistoryFragment chatHistoryFragment;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_test);
        Toolbar toolbar = findViewById(R.id.toolbar);
        drawerLayout = findViewById(R.id.drawer_layout);
        NavigationView navigationView = findViewById(R.id.nav_view);

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
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
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

    private void openInBrowser(String url) {
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
        startActivity(intent);
    }

    public void runModel(String destModelDir, String modelName, String sessionId) {
        ModelDownloadManager.getInstance(this).pauseAllDownloads();
        drawerLayout.close();
        progressDialog = new ProgressDialog(this);
        progressDialog.setMessage(getResources().getString(R.string.model_loading));
        progressDialog.show();
        if (destModelDir == null) {
            destModelDir = ModelDownloadManager.getInstance(this).getDownloadPath(modelName).getAbsolutePath();
        }
        boolean isDiffusion = ModelUtils.isDiffusionModel(modelName);
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
        intent.putExtra("modelName", modelName);
        startActivity(intent);
    }

    public void onStarProject(View view) {
        openInBrowser(this.repoGithubUrl);
    }

    public void onReportIssue(View view) {
        openInBrowser(this.repoGithubUrl + "/issues");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == ModelDownloadManager.REQUEST_CODE_POST_NOTIFICATIONS) {
            ModelDownloadManager.getInstance(this).startForegroundService();
        }
    }
}