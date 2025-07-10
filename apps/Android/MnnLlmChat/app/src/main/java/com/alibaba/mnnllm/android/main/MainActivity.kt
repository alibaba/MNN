// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.main

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.util.Log
import android.util.TypedValue
import android.view.Menu
import android.view.MenuInflater
import android.view.MenuItem
import android.view.View
import android.view.ViewTreeObserver
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SearchView
import androidx.appcompat.widget.Toolbar
import androidx.core.view.GravityCompat
import androidx.core.view.MenuHost
import androidx.core.view.MenuProvider
import androidx.drawerlayout.widget.DrawerLayout
import androidx.fragment.app.Fragment
import androidx.lifecycle.Lifecycle
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.benchmark.BenchmarkFragment
import com.alibaba.mnnllm.android.chat.ChatRouter
import com.alibaba.mnnllm.android.history.ChatHistoryFragment
import com.alibaba.mnnllm.android.mainsettings.MainSettingsActivity
import com.alibaba.mnnllm.android.modelist.ModelListFragment
import com.alibaba.mnnllm.android.modelmarket.ModelMarketFragment
import com.alibaba.mnnllm.android.update.UpdateChecker
import com.alibaba.mnnllm.android.utils.CrashUtil
import com.alibaba.mnnllm.android.utils.GithubUtils
import com.alibaba.mnnllm.android.utils.RouterUtils.startActivity
import com.alibaba.mnnllm.android.utils.Searchable
import com.alibaba.mnnllm.android.widgets.BottomTabBar
import com.google.android.material.appbar.AppBarLayout
import com.google.android.material.appbar.MaterialToolbar
import com.techiness.progressdialoglibrary.ProgressDialog
import android.view.ViewGroup

class MainActivity : AppCompatActivity() {
    private var progressDialog: ProgressDialog? = null
    private lateinit var drawerLayout: DrawerLayout
    private var toggle: ActionBarDrawerToggle? = null
    private lateinit var appBarLayout: AppBarLayout
    private lateinit var materialToolbar: MaterialToolbar
    private var toolbarHeightPx: Int = 0
    private var offsetChangedListener: AppBarLayout.OnOffsetChangedListener? = null
    private var modelListFragment: ModelListFragment? = null
    private var modelMarketFragment: ModelMarketFragment? = null
    private var benchmarkFragment: BenchmarkFragment? = null
    private var chatHistoryFragment: ChatHistoryFragment? = null
    private var currentFragment: Fragment? = null

    private var filterComponent: FilterComponent? = null
    private var updateChecker: UpdateChecker? = null
    private lateinit var expandableFabLayout: View
    
    // Add field to track current search view
    private var currentSearchView: SearchView? = null

    private lateinit var bottomNav: BottomTabBar

    private val menuProvider: MenuProvider = object : MenuProvider {
        override fun onCreateMenu(menu: Menu, menuInflater: MenuInflater) {
            menuInflater.inflate(R.menu.menu_main, menu)
            setupSearchView(menu)
            setupOtherMenuItems(menu)
        }

        override fun onMenuItemSelected(menuItem: MenuItem): Boolean {
            return true
        }

        override fun onPrepareMenu(menu: Menu) {
            super.onPrepareMenu(menu)
            val searchItem = menu.findItem(R.id.action_search)
            val reportCrashMenu = menu.findItem(R.id.action_report_crash)
            reportCrashMenu.isVisible = CrashUtil.hasCrash()
            
            // Show/hide search based on current fragment
            searchItem.isVisible = when (currentFragment) {
                modelListFragment, modelMarketFragment -> true
                else -> false
            }
        }
    }

    private fun setupSearchView(menu: Menu) {
        val searchItem = menu.findItem(R.id.action_search)
        val searchView = searchItem.actionView as SearchView?
        if (searchView != null) {
            currentSearchView = searchView
            searchView.setOnQueryTextListener(object : SearchView.OnQueryTextListener {
                override fun onQueryTextSubmit(query: String): Boolean {
                    handleSearch(query)
                    return false
                }

                override fun onQueryTextChange(query: String): Boolean {
                    handleSearch(query)
                    return true
                }
            })
            searchItem.setOnActionExpandListener(object : MenuItem.OnActionExpandListener {
                override fun onMenuItemActionExpand(item: MenuItem): Boolean {
                    Log.d(TAG, "SearchView expanded")
                    return true
                }

                override fun onMenuItemActionCollapse(item: MenuItem): Boolean {
                    Log.d(TAG, "SearchView collapsed")
                    handleSearchCleared()
                    return true
                }
            })
        }
    }

    private fun setupOtherMenuItems(menu: Menu) {
        val issueMenu = menu.findItem(R.id.action_github_issue)
        issueMenu.setOnMenuItemClickListener { 
            onReportIssue(null)
            true
        }
        
        val settingsMenu = menu.findItem(R.id.action_settings)
        settingsMenu.setOnMenuItemClickListener {
            startActivity(this@MainActivity, MainSettingsActivity::class.java)
            true
        }

        val starGithub = menu.findItem(R.id.action_star_project)
        starGithub.setOnMenuItemClickListener { 
            onStarProject(null)
            true
        }
        
        val reportCrashMenu = menu.findItem(R.id.action_report_crash)
        reportCrashMenu.setOnMenuItemClickListener {
            if (CrashUtil.hasCrash()) {
                CrashUtil.shareLatestCrash(this@MainActivity)
            }
            true
        }
    }

    private fun handleSearch(query: String) {
        val searchableFragment = currentFragment as? Searchable
        searchableFragment?.onSearchQuery(query)
    }

    private fun handleSearchCleared() {
        val searchableFragment = currentFragment as? Searchable
        searchableFragment?.onSearchCleared()
    }

    /**
     * Set the SearchView query and expand it if needed
     */
    fun setSearchQuery(query: String) {
        if (query.isEmpty()) return
        
        val menu = materialToolbar.menu
        val searchItem = menu?.findItem(R.id.action_search)
        
        if (searchItem != null && searchItem.isVisible) {
            try {
                // Expand the search view first
                searchItem.expandActionView()
                
                // Set the query after expansion
                currentSearchView?.let { searchView ->
                    searchView.setQuery(query, false)
                    searchView.clearFocus() // Prevent automatic keyboard popup
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to set search query: $query", e)
            }
        }
    }
    
    /**
     * Get the current search query
     */
    fun getCurrentSearchQuery(): String {
        return currentSearchView?.query?.toString() ?: ""
    }
    
    /**
     * Clear the search query and collapse the SearchView
     */
    fun clearSearch() {
        val menu = materialToolbar.menu
        val searchItem = menu?.findItem(R.id.action_search)
        searchItem?.collapseActionView()
    }

    private fun setupAppBar() {
        appBarLayout = findViewById(R.id.app_bar)
        materialToolbar = findViewById(R.id.toolbar)

        toolbarHeightPx = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            48f, // Toolbar height in DP from your XML
            resources.displayMetrics
        ).toInt()

        materialToolbar.viewTreeObserver.addOnGlobalLayoutListener(object : ViewTreeObserver.OnGlobalLayoutListener {
            override fun onGlobalLayout() {
                materialToolbar.viewTreeObserver.removeOnGlobalLayoutListener(this)
                val measuredHeight = materialToolbar.height
                if (measuredHeight > 0) {
                    toolbarHeightPx = measuredHeight
                }
            }
        })

        offsetChangedListener = AppBarLayout.OnOffsetChangedListener { appBarLayout, verticalOffset ->
            if (toolbarHeightPx <= 0) {
                val currentToolbarHeight = materialToolbar.height
                if (currentToolbarHeight > 0) {
                    toolbarHeightPx = currentToolbarHeight
                } else {
                    toolbarHeightPx = TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, 48f, resources.displayMetrics).toInt()
                    if (toolbarHeightPx == 0) return@OnOffsetChangedListener // Still zero, cannot proceed
                }
            }
            val absVerticalOffset = Math.abs(verticalOffset)
            var alpha = 1.0f - (absVerticalOffset.toFloat() / toolbarHeightPx.toFloat())
            alpha = alpha.coerceIn(0.0f, 1.0f)
            materialToolbar.alpha = alpha
        }
        //appBarLayout.addOnOffsetChangedListener(offsetChangedListener)
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setupAppBar()
        drawerLayout = findViewById(R.id.drawer_layout)
        expandableFabLayout = findViewById(R.id.expandable_fab_layout)
        updateChecker = UpdateChecker(this)
        updateChecker!!.checkForUpdates(this, false)
        
        bottomNav = findViewById(R.id.bottom_navigation)
        if (savedInstanceState == null) {
            modelListFragment = ModelListFragment()
            modelMarketFragment = ModelMarketFragment()
            benchmarkFragment = BenchmarkFragment()
            supportFragmentManager.beginTransaction()
                .add(R.id.main_fragment_container, modelMarketFragment!!, "market").hide(modelMarketFragment!!)
                .add(R.id.main_fragment_container, benchmarkFragment!!, "benchmark").hide(benchmarkFragment!!)
                .add(R.id.main_fragment_container, modelListFragment!!, "list")
                .commit()
            currentFragment = modelListFragment
        } else {
            Log.d(TAG, "onCreate: Configuration change detected, restoring fragments")
            benchmarkFragment = supportFragmentManager.findFragmentByTag("benchmark") as? BenchmarkFragment
            modelListFragment = supportFragmentManager.findFragmentByTag("list") as? ModelListFragment
            modelMarketFragment = supportFragmentManager.findFragmentByTag("market") as? ModelMarketFragment
            
            Log.d(TAG, "onCreate: Found fragments - list: ${modelListFragment != null}, market: ${modelMarketFragment != null}, benchmark: ${benchmarkFragment != null}")
            
            val allFragments = listOfNotNull(modelListFragment, modelMarketFragment, benchmarkFragment)
            Log.d(TAG, "onCreate: All fragments count: ${allFragments.size}")
            
            allFragments.forEachIndexed { index, fragment ->
                Log.d(TAG, "onCreate: Fragment $index (${fragment.javaClass.simpleName}) - isVisible: ${fragment.isVisible}, isAdded: ${fragment.isAdded}, isHidden: ${fragment.isHidden}")
            }
            
            val visibleFragment = allFragments.find { !it.isHidden }
            Log.d(TAG, "onCreate: Found visible fragment (by !isHidden): ${visibleFragment?.javaClass?.simpleName}")
            
            Log.d(TAG, "onCreate: Current fragment set to: ${currentFragment?.javaClass?.simpleName}")
            
            val transaction = supportFragmentManager.beginTransaction()
            allFragments.forEach { 
                Log.d(TAG, "onCreate: Hiding fragment: ${it.javaClass.simpleName}")
                transaction.hide(it) 
            }
            
            if (currentFragment != null) {
                Log.d(TAG, "onCreate: Showing current fragment: ${currentFragment!!.javaClass.simpleName}")
                transaction.show(currentFragment!!)
            }
            transaction.commit()
            
            // 如果某个fragment为null，说明恢复失败，需要重新创建
            if (modelListFragment == null) {
                Log.d(TAG, "onCreate: ModelListFragment is null, creating new one")
                modelListFragment = ModelListFragment()
                supportFragmentManager.beginTransaction()
                    .add(R.id.main_fragment_container, modelListFragment!!, "list")
                    .commit()
                if (currentFragment == null) {
                    currentFragment = modelListFragment
                    Log.d(TAG, "onCreate: Set currentFragment to newly created ModelListFragment")
                }
            }
            if (modelMarketFragment == null) {
                Log.d(TAG, "onCreate: ModelMarketFragment is null, creating new one")
                modelMarketFragment = ModelMarketFragment()
                supportFragmentManager.beginTransaction()
                    .add(R.id.main_fragment_container, modelMarketFragment!!, "market")
                    .hide(modelMarketFragment!!)
                    .commit()
            }
            if (benchmarkFragment == null) {
                Log.d(TAG, "onCreate: BenchmarkFragment is null, creating new one")
                benchmarkFragment = BenchmarkFragment()
                supportFragmentManager.beginTransaction()
                    .add(R.id.main_fragment_container, benchmarkFragment!!, "benchmark")
                    .hide(benchmarkFragment!!)
                    .commit()
            }
        }

        bottomNav.setOnTabSelectedListener { tab ->
            Log.d(TAG, "bottomNav.setOnTabSelectedListener: Tab selected: $tab")
            val targetFragment = when (tab) {
                BottomTabBar.Tab.LOCAL_MODELS -> modelListFragment
                BottomTabBar.Tab.MODEL_MARKET -> modelMarketFragment
                BottomTabBar.Tab.BENCHMARK -> benchmarkFragment
            }
            
            Log.d(TAG, "bottomNav.setOnTabSelectedListener: Target fragment: ${targetFragment?.javaClass?.simpleName}, Current fragment: ${currentFragment?.javaClass?.simpleName}")
            
            if (targetFragment != null && currentFragment != targetFragment) {
                Log.d(TAG, "bottomNav.setOnTabSelectedListener: Switching from ${currentFragment?.javaClass?.simpleName} to ${targetFragment.javaClass.simpleName}")
                
                if (currentFragment is ModelMarketFragment) {
                    Log.d(TAG, "bottomNav.setOnTabSelectedListener: Current fragment is ModelMarketFragment, ensuring toolbar is cleaned")
                    val appBarContent = findViewById<ViewGroup>(R.id.app_bar_content)
                    val filterContainerView = appBarContent?.findViewById<View>(R.id.filter_download_state)?.parent as? ViewGroup
                    if (filterContainerView != null && appBarContent.indexOfChild(filterContainerView) != -1) {
                        Log.d(TAG, "bottomNav.setOnTabSelectedListener: Removing filter container from appBarContent")
                        appBarContent.removeView(filterContainerView)
                    }
                }
                
                supportFragmentManager.beginTransaction()
                    .hide(currentFragment!!)
                    .show(targetFragment)
                    .commitNow()
                currentFragment = targetFragment
                invalidateOptionsMenu()
            }
            val titleRes = when (tab) {
                BottomTabBar.Tab.LOCAL_MODELS -> R.string.nav_name_chats
                BottomTabBar.Tab.MODEL_MARKET -> R.string.models_market
                BottomTabBar.Tab.BENCHMARK -> R.string.benchmark
            }
            supportActionBar?.setTitle(titleRes)
            
            expandableFabLayout.visibility = if (tab == BottomTabBar.Tab.LOCAL_MODELS) {
                View.VISIBLE
            } else {
                View.GONE
            }
        }
        
        Log.d(TAG, "onCreate: Before bottomNav.select, currentFragment: ${currentFragment?.javaClass?.simpleName}")
        val initialTab = when (currentFragment) {
            modelMarketFragment -> BottomTabBar.Tab.MODEL_MARKET
            benchmarkFragment -> BottomTabBar.Tab.BENCHMARK
            else -> BottomTabBar.Tab.LOCAL_MODELS
        }
        Log.d(TAG, "onCreate: Setting initial tab to: $initialTab")
        bottomNav.select(initialTab)
        
        // 设置对应的toolbar标题
        val titleRes = when (initialTab) {
            BottomTabBar.Tab.LOCAL_MODELS -> R.string.nav_name_chats
            BottomTabBar.Tab.MODEL_MARKET -> R.string.models_market
            BottomTabBar.Tab.BENCHMARK -> R.string.benchmark
        }
        supportActionBar?.setTitle(titleRes)
        
        // 设置对应的fab可见性
        expandableFabLayout.visibility = if (initialTab == BottomTabBar.Tab.LOCAL_MODELS) {
            View.VISIBLE
        } else {
            View.GONE
        }

        toggle = ActionBarDrawerToggle(
            this, drawerLayout,
            toolbar,
            R.string.nav_open,
            R.string.nav_close
        )
        drawerLayout.addDrawerListener(toggle!!)
        toggle!!.syncState()
        if (chatHistoryFragment == null) {
            chatHistoryFragment = ChatHistoryFragment()
        }
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
        
        val menuHost: MenuHost = this
        menuHost.addMenuProvider(menuProvider, this, Lifecycle.State.RESUMED)
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (toggle!!.onOptionsItemSelected(item)) {
            return true
        }
        return super.onOptionsItemSelected(item)
    }
    
    fun runModel(destModelDir: String?, modelIdParam: String?, sessionId: String?) {
        ChatRouter.startRun(this, modelIdParam!!, destModelDir, sessionId)
        drawerLayout.close()
    }

    fun onStarProject(view: View?) {
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle(R.string.star_project_confirm_title)
            .setMessage(R.string.star_project_confirm_message)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                GithubUtils.starProject(this)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .setCancelable(false)
            .show()
    }

    fun onReportIssue(view: View?) {
        GithubUtils.reportIssue(this)
    }

    fun addLocalModels(view: View?) {
        val adbCommand = "adb shell mkdir -p /data/local/tmp/mnn_models && adb push \${model_path} /data/local/tmp/mnn_models/"
        val message = getResources().getString(R.string.add_local_models_message, adbCommand)
        val dialog = androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle(R.string.add_local_models_title)
            .setMessage(message)
            .setPositiveButton(android.R.string.ok) { dialog, _ -> dialog.dismiss() }
            .setNeutralButton(R.string.copy_command) { _, _ ->
                val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                val clip = ClipData.newPlainText("ADB Command", adbCommand)
                clipboard.setPrimaryClip(clip)
                Toast.makeText(this, R.string.copied_to_clipboard, Toast.LENGTH_SHORT).show()
            }
            .create()
        dialog.show()
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

    override fun onDestroy() {
        super.onDestroy()
        /*
        offsetChangedListener?.let {
            appBarLayout.removeOnOffsetChangedListener(it)
        }
        */
        
    }

    fun onAddModelButtonClick(view: View) {
        bottomNav.select(BottomTabBar.Tab.MODEL_MARKET)
    }

    companion object {
        const val TAG: String = "MainActivity"
    }
}