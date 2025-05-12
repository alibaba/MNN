// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.history.ChatHistoryFragment
import com.alibaba.mnnllm.android.mainsettings.MainSettings.isStopDownloadOnChatEnabled
import com.alibaba.mnnllm.android.modelist.ModelListFragment
import com.alibaba.mnnllm.android.update.UpdateChecker
import com.alibaba.mnnllm.android.utils.GithubUtils
import com.alibaba.mnnllm.android.utils.ModelUtils
import com.techiness.progressdialoglibrary.ProgressDialog
import java.io.File

class MainActivity : AppCompatActivity() {
    private var progressDialog: ProgressDialog? = null
    private lateinit var drawerLayout: DrawerLayout
    private var toggle: ActionBarDrawerToggle? = null
    private var modelListFragment: ModelListFragment? = null
        get() {
            if (field == null) {
                field = ModelListFragment()
            }
            return field
        }
    private var chatHistoryFragment: ChatHistoryFragment? = null
        get() {
            if (field == null) {
                field = ChatHistoryFragment()
            }
            return field
        }
    private var updateChecker: UpdateChecker? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        drawerLayout = findViewById(R.id.drawer_layout)
        updateChecker = UpdateChecker(this)
        updateChecker!!.checkForUpdates(this, false)
        toggle = ActionBarDrawerToggle(
            this, drawerLayout,
            toolbar,
            R.string.nav_open,
            R.string.nav_close
        )
        drawerLayout.addDrawerListener(toggle!!)
        toggle!!.syncState()
        supportFragmentManager.beginTransaction()
            .replace(
                R.id.main_fragment_container,
                modelListFragment!!
            )
            .commit()
        supportFragmentManager.beginTransaction()
            .replace(
                R.id.history_fragment_container,
                chatHistoryFragment!!
            )
            .commit()
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (drawerLayout.isDrawerOpen(GravityCompat.START)) {
                    drawerLayout.closeDrawer(GravityCompat.START)
                } else {
                    finish()
                }
            }
        })
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (toggle!!.onOptionsItemSelected(item)) {
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    fun runModel(destModelDir: String?, modelId: String?, sessionId: String?) {
        var destPath = destModelDir
        Log.d(TAG, "runModel destModelDir: $destPath")
        if (isStopDownloadOnChatEnabled(this)) {
            ModelDownloadManager.getInstance(this).pauseAllDownloads()
        }
        drawerLayout.close()
        progressDialog = ProgressDialog(this)
        progressDialog!!.setMessage(resources.getString(R.string.model_loading))
        progressDialog!!.show()
        if (destPath == null) {
            destPath =
                ModelDownloadManager.getInstance(this).getDownloadedFile(modelId!!)?.absolutePath
            if (destPath == null) {
                Toast.makeText(
                    this,
                    getString(R.string.model_not_found, modelId),
                    Toast.LENGTH_LONG
                ).show()
                progressDialog?.dismiss()
                return
            }
        }
        val isDiffusion = ModelUtils.isDiffusionModel(modelId!!)
        var configFilePath: String? = null
        if (!isDiffusion) {
            val configFileName = "config.json"
            configFilePath = "$destPath/$configFileName"
            val configFileExists = File(configFilePath).exists()
            if (!configFileExists) {
                Toast.makeText(
                    this,
                    getString(R.string.config_file_not_found, configFilePath),
                    Toast.LENGTH_LONG
                ).show()
                progressDialog!!.dismiss()
                return
            }
        }
        progressDialog!!.dismiss()
        val intent = Intent(this, ChatActivity::class.java)
        intent.putExtra("chatSessionId", sessionId)
        if (isDiffusion) {
            intent.putExtra("diffusionDir", destPath)
        } else {
            intent.putExtra("configFilePath", configFilePath)
        }
        intent.putExtra("modelId", modelId)
        intent.putExtra("modelName", ModelUtils.getModelName(modelId))
        startActivity(intent)
    }

    fun onStarProject(view: View?) {
        GithubUtils.starProject(this)
    }

    fun onReportIssue(view: View?) {
        GithubUtils.reportIssue(this)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == ModelDownloadManager.REQUEST_CODE_POST_NOTIFICATIONS) {
            ModelDownloadManager.getInstance(this).tryStartForegroundService()
        }
    }

    fun checkForUpdate() {
        updateChecker!!.checkForUpdates(this, true)
    }

    companion object {
        const val TAG: String = "MainActivity"
    }
}