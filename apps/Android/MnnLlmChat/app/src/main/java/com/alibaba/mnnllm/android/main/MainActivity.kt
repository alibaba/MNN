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
import com.alibaba.mls.api.source.ModelSources
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
import com.alibaba.mnnllm.android.widgets.ModelSwitcherView
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import com.google.android.material.appbar.AppBarLayout
import com.google.android.material.appbar.MaterialToolbar
import com.alibaba.mnnllm.android.chat.SelectSourceFragment
import android.content.Intent
import com.alibaba.mnnllm.android.qnn.QnnModule
import com.alibaba.mnnllm.android.privacy.PrivacyPolicyManager
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import com.alibaba.mnnllm.android.privacy.PrivacyPolicyDialogFragment

class MainActivity : AppCompatActivity(), MainFragmentManager.FragmentLifecycleListener {
    private lateinit var drawerLayout: DrawerLayout
    private var toggle: ActionBarDrawerToggle? = null
    private lateinit var appBarLayout: AppBarLayout
    private lateinit var materialToolbar: MaterialToolbar
    private lateinit var mainTitleSwitcher: ModelSwitcherView
    private var toolbarHeightPx: Int = 0
    private var offsetChangedListener: AppBarLayout.OnOffsetChangedListener? = null
    private var chatHistoryFragment: ChatHistoryFragment? = null
    private var updateChecker: UpdateChecker? = null
    private lateinit var expandableFabLayout: View
    
    // Add field to track current search view
    private var currentSearchView: SearchView? = null

    private lateinit var bottomNav: BottomTabBar
    private lateinit var mainFragmentManager: MainFragmentManager

    private val currentFragment: Fragment?
        get() {
            return mainFragmentManager.activeFragment
        }

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
            Log.d(TAG, "onPrepareMenu")
            super.onPrepareMenu(menu)
            val searchItem = menu.findItem(R.id.action_search)
            val reportCrashMenu = menu.findItem(R.id.action_report_crash)
            reportCrashMenu.isVisible = CrashUtil.hasCrash()
            
            // Show/hide search based on current fragment
            searchItem.isVisible = when (bottomNav.getSelectedTab()) {
                BottomTabBar.Tab.LOCAL_MODELS, BottomTabBar.Tab.MODEL_MARKET -> true
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
        mainTitleSwitcher = findViewById(R.id.main_title_switcher)

        // Initially hide the dropdown arrow and make it non-clickable
        updateMainTitleSwitcherMode(false)

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

    /**
     * Update the mode of the main title switcher
     * @param isSourceSwitcherMode Whether it is in source switcher mode (shows dropdown arrow and is clickable)
     */
    private fun updateMainTitleSwitcherMode(isSourceSwitcherMode: Boolean) {
        val dropdownArrow = mainTitleSwitcher.findViewById<View>(R.id.iv_dropdown_arrow)
        if (isSourceSwitcherMode) {
            // Source switcher mode: show dropdown arrow, clickable
            dropdownArrow?.visibility = View.VISIBLE
            mainTitleSwitcher.isClickable = true
            mainTitleSwitcher.isFocusable = true
            mainTitleSwitcher.setOnClickListener {
                // Show source selection dialog
                showSourceSelectionDialog()
            }
        } else {
            // Title display mode: hide dropdown arrow, not clickable
            dropdownArrow?.visibility = View.GONE
            mainTitleSwitcher.isClickable = false
            mainTitleSwitcher.isFocusable = false
            mainTitleSwitcher.setOnClickListener(null)
        }
    }

    /**
     * Show source selection dialog
     */
    private fun showSourceSelectionDialog() {
        val availableSources = ModelSources.sourceList
        val displayNames = ModelSources.sourceDisPlayList
        val currentProvider = MainSettings.getDownloadProviderString(this)
        
        // Use SelectSourceFragment from ModelMarketFragment
        val fragment = SelectSourceFragment.newInstance(availableSources, displayNames, currentProvider)
        fragment.setOnSourceSelectedListener { selectedSource ->
            MainSettings.setDownloadProvider(this, selectedSource)
            // Set title to display name
            val idx = ModelSources.sourceList.indexOf(selectedSource)
            val displayName = if (idx != -1) getString(ModelSources.sourceDisPlayList[idx]) else selectedSource
            mainTitleSwitcher.text = displayName
            // Notify ModelMarketFragment to update
            if (currentFragment is ModelMarketFragment) {
                (currentFragment as ModelMarketFragment).onSourceChanged()
            }
        }
        fragment.show(supportFragmentManager, "SourceSelectionDialog")
    }

    private fun updateExpandableFabLayout(newTab: BottomTabBar.Tab) {
        expandableFabLayout.visibility = if (newTab == BottomTabBar.Tab.LOCAL_MODELS) {
            View.VISIBLE
        } else {
            View.GONE
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        mainFragmentManager.onSaveInstanceState(outState)
    }

    override fun onTabChanged(newTab: BottomTabBar.Tab) {
        Log.d(TAG, "Tab changed to $newTab, updating UI accordingly.")

        when (newTab) {
            BottomTabBar.Tab.LOCAL_MODELS -> {
                updateMainTitleSwitcherMode(false)
                mainTitleSwitcher.text = getString(R.string.nav_name_chats)
            }
            BottomTabBar.Tab.MODEL_MARKET -> {
                updateMainTitleSwitcherMode(true)
                val currentProvider = MainSettings.getDownloadProviderString(this)
                val idx = ModelSources.sourceList.indexOf(currentProvider)
                val displayName = if (idx != -1) getString(ModelSources.sourceDisPlayList[idx]) else currentProvider
                mainTitleSwitcher.text = displayName
            }
            BottomTabBar.Tab.BENCHMARK -> {
                updateMainTitleSwitcherMode(false)
                mainTitleSwitcher.text = getString(R.string.benchmark)
            }
        }
        updateExpandableFabLayout(newTab)
        invalidateOptionsMenu()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Check privacy policy agreement first
        checkPrivacyPolicyAgreement()
        
        setContentView(R.layout.activity_main)
        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setupAppBar()
        bottomNav = findViewById(R.id.bottom_navigation)
        drawerLayout = findViewById(R.id.drawer_layout)
        expandableFabLayout = findViewById(R.id.expandable_fab_layout)
        updateChecker = UpdateChecker(this)
        updateChecker!!.checkForUpdates(this, false)
        mainFragmentManager = MainFragmentManager(this, R.id.main_fragment_container, bottomNav, this)
        mainFragmentManager.initialize(savedInstanceState)
        Log.d(TAG, "onCreate: Before bottomNav.select, currentFragment: ${currentFragment?.javaClass?.simpleName}")
        toggle = ActionBarDrawerToggle(
            this, drawerLayout,
            toolbar,
            R.string.nav_open,
            R.string.nav_close
        )
        drawerLayout.addDrawerListener(toggle!!)
        toggle!!.syncState()
        // Remove eager creation of chatHistoryFragment here
        // Lazy load chatHistoryFragment when drawer is first opened
        drawerLayout.addDrawerListener(object : DrawerLayout.DrawerListener {
            override fun onDrawerSlide(drawerView: View, slideOffset: Float) {}
            override fun onDrawerOpened(drawerView: View) {
                if (chatHistoryFragment == null) {
                    chatHistoryFragment = ChatHistoryFragment()
                    supportFragmentManager.beginTransaction()
                        .replace(R.id.history_fragment_container, chatHistoryFragment!!)
                        .commit()
                }
            }
            override fun onDrawerClosed(drawerView: View) {}
            override fun onDrawerStateChanged(newState: Int) {}
        })
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
        supportActionBar?.setDisplayShowTitleEnabled(false)  // Disable default title display
        
        val menuHost: MenuHost = this
        menuHost.addMenuProvider(menuProvider, this, Lifecycle.State.RESUMED)
        
        // Handle intent extras for navigation from notification
        handleIntentExtras(intent)
    }
    
    private fun handleIntentExtras(intent: Intent?) {
        intent?.let {
            val selectTab = it.getStringExtra(EXTRA_SELECT_TAB)
            if (selectTab == TAB_MODEL_MARKET) {
                // Post to ensure the UI is ready
                bottomNav.post {
                    bottomNav.select(BottomTabBar.Tab.MODEL_MARKET)
                }
            }
        }
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
    }

    fun onAddModelButtonClick(view: View) {
        bottomNav.select(BottomTabBar.Tab.MODEL_MARKET)
    }
    
    /**
     * Check if user has agreed to privacy policy
     * If not, show privacy policy dialog
     */
    private fun checkPrivacyPolicyAgreement() {
        if (!ENABLE_PRIVACY_POLICY_CHECK) {
            return
        }
        
        val privacyManager = PrivacyPolicyManager.getInstance(this)
        
        if (!privacyManager.hasUserAgreed()) {
            showPrivacyPolicyDialog()
        }
    }
    
    /**
     * Show privacy policy dialog
     */
    private fun showPrivacyPolicyDialog() {
        val dialog = PrivacyPolicyDialogFragment.newInstance(
            onAgree = {
                // User agreed to privacy policy
                val privacyManager = PrivacyPolicyManager.getInstance(this)
                privacyManager.setUserAgreed(true)
                Log.d(TAG, "User agreed to privacy policy")
            },
            onDisagree = {
                // User disagreed to privacy policy
                Toast.makeText(this, getString(R.string.privacy_policy_exit_message), Toast.LENGTH_LONG).show()
                Log.d(TAG, "User disagreed to privacy policy")
                // Exit the application
                finishAffinity()
            }
        )
        
        dialog.show(supportFragmentManager, PrivacyPolicyDialogFragment.TAG)
    }

    companion object {
        const val TAG: String = "MainActivity"
        const val EXTRA_SELECT_TAB = "com.alibaba.mnnllm.android.select_tab"
        const val TAB_MODEL_MARKET = "model_market"
        
        // Control whether to show privacy policy agreement dialog
        const val ENABLE_PRIVACY_POLICY_CHECK = false
    }
}