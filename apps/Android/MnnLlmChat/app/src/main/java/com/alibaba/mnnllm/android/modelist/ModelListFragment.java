// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelist;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.SearchView;
import androidx.core.view.MenuHost;
import androidx.core.view.MenuProvider;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Lifecycle;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.alibaba.mls.api.HfRepoItem;
import com.alibaba.mnnllm.android.MainActivity;
import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.settings.MainSettingsActivity;
import com.alibaba.mnnllm.android.utils.PreferenceUtils;
import com.alibaba.mnnllm.android.utils.RouterUtils;

import java.util.ArrayList;
import java.util.List;

public class ModelListFragment extends Fragment implements ModelListContract.View {
    private RecyclerView modelListRecyclerView;

    private ModelListAdapter modelListAdapter;
    private ModelListPresenter modelListPresenter;
    private final List<HfRepoItem> hfRepoItemList = new ArrayList<>();

    private View modelListLoadingView;
    private View modelListErrorView;

    private TextView modelListErrorText;

    private boolean filterDownloaded = false;
    private String filterQuery = "";

    private void setupSearchView(Menu menu) {
        android.view.MenuItem searchItem = menu.findItem(R.id.action_search);
        SearchView searchView = (SearchView) searchItem.getActionView();
        if (searchView != null) {
            searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
                @Override
                public boolean onQueryTextSubmit(String query) {
                    filterQuery = query;
                    modelListAdapter.setFilter(query, filterDownloaded);
                    return false;
                }

                @Override
                public boolean onQueryTextChange(String query) {
                    filterQuery = query;
                    modelListAdapter.setFilter(query, filterDownloaded);
                    return true;
                }
            });
            searchItem.setOnActionExpandListener(new MenuItem.OnActionExpandListener() {
                @Override
                public boolean onMenuItemActionExpand(MenuItem item) {
                    // SearchView is expanded
                    Log.d("SearchView", "SearchView expanded");
                    return true;
                }

                @Override
                public boolean onMenuItemActionCollapse(MenuItem item) {
                    // SearchView is collapsed
                    Log.d("SearchView", "SearchView collapsed");
                    modelListAdapter.unfilter();;
                    return true;
                }
            });
        }
    }

    private final MenuProvider menuProvider = new MenuProvider() {
        @Override
        public void onCreateMenu(@NonNull Menu menu, @NonNull MenuInflater menuInflater) {
            // Inflate your menu resource here
            menuInflater.inflate(R.menu.menu_main, menu);
            setupSearchView(menu);
            MenuItem issueMenu = menu.findItem(R.id.action_github_issue);
            issueMenu.setOnMenuItemClickListener(item -> {
                if (getActivity() != null) {
                    ((MainActivity) getActivity()).onReportIssue(null);
                }
                return true;
            });

            MenuItem filterDownloadedMenu = menu.findItem(R.id.action_filter_downloaded);
            filterDownloadedMenu.setChecked(PreferenceUtils.isFilterDownloaded(getContext()));
            filterDownloadedMenu.setOnMenuItemClickListener(item -> {
                filterDownloaded = PreferenceUtils.isFilterDownloaded(getContext());
                filterDownloaded = !filterDownloaded;
                PreferenceUtils.setFilterDownloaded(getContext(), filterDownloaded);
                filterDownloadedMenu.setChecked(filterDownloaded);
                modelListAdapter.setFilter(filterQuery, filterDownloaded);
                return true;
            });
            MenuItem settingsMenu = menu.findItem(R.id.action_settings);
            settingsMenu.setOnMenuItemClickListener(item -> {
                if (getActivity() != null) {
                    RouterUtils.INSTANCE.startActivity(getActivity(), MainSettingsActivity.class);
                }
                return true;
            });

            MenuItem starGithub = menu.findItem(R.id.action_star_project);
            starGithub.setOnMenuItemClickListener(item -> {
                if (getActivity() != null) {
                    ((MainActivity) getActivity()).onStarProject(null);
                }
                return true;
            });
        }

        @Override
        public boolean onMenuItemSelected(@NonNull MenuItem menuItem) {
            return true;
        }

        @Override
        public void onPrepareMenu(@NonNull Menu menu) {
            MenuProvider.super.onPrepareMenu(menu);
            MenuItem menuResumeAllDownlods = menu.findItem(R.id.action_resume_all_downloads);
            menuResumeAllDownlods.setVisible(modelListPresenter.getUnfisnishedDownloadsSize() > 0);
            menuResumeAllDownlods.setOnMenuItemClickListener((item)->{
                modelListPresenter.resumeAllDownloads();
                return true;
            });
        }
    };

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_modellist, container, false);
        modelListRecyclerView = view.findViewById(R.id.model_list_recycler_view);
        modelListLoadingView = view.findViewById(R.id.model_list_loading_view);
        modelListErrorView = view.findViewById(R.id.model_list_failed_view);
        modelListErrorText = modelListErrorView.findViewById(R.id.tv_error_text);
        modelListErrorView.setOnClickListener(v -> {
            modelListPresenter.load();
        });
        modelListRecyclerView.setLayoutManager(new LinearLayoutManager(getContext(), LinearLayoutManager.VERTICAL, false));
        modelListAdapter = new ModelListAdapter(hfRepoItemList);

        modelListRecyclerView.setAdapter(modelListAdapter);
        modelListPresenter = new ModelListPresenter(getContext(), this);
        modelListAdapter.setModelListListener(modelListPresenter);
        filterDownloaded = PreferenceUtils.isFilterDownloaded(getContext());
        modelListAdapter.setFilter(filterQuery, filterDownloaded);
        modelListPresenter.onCreate();
        return view;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        MenuHost menuHost = requireActivity();
        menuHost.addMenuProvider(menuProvider, getViewLifecycleOwner(), Lifecycle.State.RESUMED);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        modelListPresenter.onDestroy();
    }

    @Override
    public void onListAvailable() {
        modelListErrorView.setVisibility(View.GONE);
        modelListLoadingView.setVisibility(View.GONE);
        modelListRecyclerView.setVisibility(View.VISIBLE);
    }

    @Override
    public void onLoading() {
        if (modelListAdapter.getItemCount() > 0) {
            return;
        }
        modelListErrorView.setVisibility(View.GONE);
        modelListLoadingView.setVisibility(View.VISIBLE);
        modelListRecyclerView.setVisibility(View.GONE);
    }

    @Override
    public void onListLoadError(String error) {
        if (modelListAdapter.getItemCount() > 0) {
            return;
        }
        modelListErrorText.setText(getString(R.string.loading_failed_click_tor_retry, error));
        modelListErrorView.setVisibility(View.VISIBLE);
        modelListLoadingView.setVisibility(View.GONE);
        modelListRecyclerView.setVisibility(View.GONE);
    }

    @Override
    public ModelListAdapter getAdapter() {
        return modelListAdapter;
    }

    @Override
    public void runModel(String absolutePath, String modelId) {
        ((MainActivity) getActivity()).runModel(absolutePath, modelId, null);
    }
}
