// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.modelist

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.Menu
import android.view.MenuInflater
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.appcompat.widget.SearchView
import androidx.core.view.MenuHost
import androidx.core.view.MenuProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.Lifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.main.MainActivity
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.mainsettings.MainSettingsActivity
import com.alibaba.mnnllm.android.utils.CrashUtil
import com.alibaba.mnnllm.android.utils.PreferenceUtils.isFilterDownloaded
import com.alibaba.mnnllm.android.utils.PreferenceUtils.setFilterDownloaded
import com.alibaba.mnnllm.android.utils.RouterUtils.startActivity

class ModelListFragment : Fragment(), ModelListContract.View {
    private lateinit var modelListRecyclerView: RecyclerView
    private lateinit var modelListLoadingView: View
    private lateinit var modelListErrorView: View
    private lateinit var modelListEmptyView: View

    override var adapter: ModelListAdapter? = null
        private set
    private var modelListPresenter: ModelListPresenter? = null
    private val hfModelItemList: MutableList<ModelItem> = mutableListOf()

    private var modelListErrorText: TextView? = null

    private var filterDownloaded = false
    private var filterQuery = ""

    private fun setupSearchView(menu: Menu) {
        val searchItem = menu.findItem(R.id.action_search)
        val searchView = searchItem.actionView as SearchView?
        if (searchView != null) {
            searchView.setOnQueryTextListener(object : SearchView.OnQueryTextListener {
                override fun onQueryTextSubmit(query: String): Boolean {
                    filterQuery = query
                    adapter!!.setFilter(query)
                    return false
                }

                override fun onQueryTextChange(query: String): Boolean {
                    filterQuery = query
                    adapter!!.setFilter(query)
                    return true
                }
            })
            searchItem.setOnActionExpandListener(object : MenuItem.OnActionExpandListener {
                override fun onMenuItemActionExpand(item: MenuItem): Boolean {
                    // SearchView is expanded
                    Log.d("SearchView", "SearchView expanded")
                    return true
                }

                override fun onMenuItemActionCollapse(item: MenuItem): Boolean {
                    // SearchView is collapsed
                    Log.d("SearchView", "SearchView collapsed")
                    adapter!!.unfilter()

                    return true
                }
            })
        }
    }

    private val menuProvider: MenuProvider = object : MenuProvider {
        override fun onCreateMenu(menu: Menu, menuInflater: MenuInflater) {
            // Inflate your menu resource here
            menuInflater.inflate(R.menu.menu_main, menu)
            setupSearchView(menu)
            val issueMenu = menu.findItem(R.id.action_github_issue)
            issueMenu.setOnMenuItemClickListener { item: MenuItem? ->
                if (activity != null) {
                    (activity as MainActivity).onReportIssue(null)
                }
                true
            }
            val settingsMenu = menu.findItem(R.id.action_settings)
            settingsMenu.setOnMenuItemClickListener {
                if (activity != null) {
                    startActivity(activity!!, MainSettingsActivity::class.java)
                }
                true
            }

            val starGithub = menu.findItem(R.id.action_star_project)
            starGithub.setOnMenuItemClickListener { item: MenuItem? ->
                if (activity != null) {
                    (activity as MainActivity).onStarProject(null)
                }
                true
            }
            val reportCrashMenu = menu.findItem(R.id.action_report_crash)
            reportCrashMenu.setOnMenuItemClickListener {
                if (CrashUtil.hasCrash()) {
                    CrashUtil.shareLatestCrash(context!!)
                }
                true
            }
        }

        override fun onMenuItemSelected(menuItem: MenuItem): Boolean {
            return true
        }

        override fun onPrepareMenu(menu: Menu) {
            super<MenuProvider>.onPrepareMenu(menu)
            val menuResumeAllDownlods = menu.findItem(R.id.action_resume_all_downloads)
            menuResumeAllDownlods.setVisible(modelListPresenter!!.unfinishedDownloadCount > 0)
            menuResumeAllDownlods.setOnMenuItemClickListener { item: MenuItem? ->
                modelListPresenter!!.resumeAllDownloads()
                true
            }
            val reportCrashMenu = menu.findItem(R.id.action_report_crash)
            reportCrashMenu.isVisible = CrashUtil.hasCrash()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_modellist, container, false)
        modelListRecyclerView = view.findViewById(R.id.model_list_recycler_view)
        modelListLoadingView = view.findViewById(R.id.model_list_loading_view)
        modelListErrorView = view.findViewById(R.id.model_list_failed_view)
        modelListEmptyView = view.findViewById(R.id.model_list_empty_view)
        modelListErrorText = modelListErrorView.findViewById(R.id.tv_error_text)
        modelListErrorView.setOnClickListener {
            modelListPresenter!!.load()
        }
        modelListRecyclerView.setLayoutManager(
            LinearLayoutManager(
                context,
                LinearLayoutManager.VERTICAL,
                false
            )
        )
        adapter = ModelListAdapter(hfModelItemList)
        adapter!!.setEmptyView(modelListEmptyView)

        modelListRecyclerView.setAdapter(adapter)
        modelListPresenter = ModelListPresenter(requireContext(), this)
        adapter!!.setModelListListener(modelListPresenter)
        filterDownloaded = isFilterDownloaded(context)
        adapter!!.setFilter(filterQuery)
        adapter!!.filterDownloadState(filterDownloaded.toString())
        modelListPresenter!!.onCreate()
        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val menuHost: MenuHost = requireActivity()
        menuHost.addMenuProvider(menuProvider, viewLifecycleOwner, Lifecycle.State.RESUMED)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        modelListPresenter!!.onDestroy()
    }

    override fun onListAvailable() {
        modelListErrorView.visibility = View.GONE
        modelListLoadingView.visibility = View.GONE
        
        // Only show recycler view if adapter has items
        if (adapter!!.itemCount > 0) {
            modelListRecyclerView.visibility = View.VISIBLE
            modelListEmptyView.visibility = View.GONE
        } else {
            modelListRecyclerView.visibility = View.GONE
            modelListEmptyView.visibility = View.VISIBLE
        }
    }

    override fun onLoading() {
        if (adapter!!.itemCount > 0) {
            return
        }
        modelListErrorView.visibility = View.GONE
        modelListLoadingView.visibility = View.VISIBLE
        modelListRecyclerView.visibility = View.GONE
    }

    override fun onListLoadError(error: String?) {
        if (adapter!!.itemCount > 0) {
            return
        }
        modelListErrorText!!.text = getString(R.string.loading_failed_click_tor_retry, error)
        modelListErrorView.visibility = View.VISIBLE
        modelListLoadingView.visibility = View.GONE
        modelListRecyclerView.visibility = View.GONE
    }

    override fun runModel(absolutePath: String?, modelId: String?) {
        (activity as MainActivity).runModel(absolutePath, modelId, null)
    }
}