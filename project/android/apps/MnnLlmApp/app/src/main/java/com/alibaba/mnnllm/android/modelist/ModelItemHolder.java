package com.alibaba.mnnllm.android.modelist;

import static com.alibaba.mnnllm.android.utils.ModelUtils.getDrawableId;

import android.annotation.SuppressLint;
import android.text.TextUtils;
import android.view.MenuInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.widget.PopupMenu;
import androidx.recyclerview.widget.RecyclerView;

import com.alibaba.mls.api.HfRepoItem;
import com.alibaba.mls.api.download.DownloadInfo;
import com.alibaba.mls.api.download.ModelDownloadManager;
import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.widgets.TagsLayout;

public class ModelItemHolder extends RecyclerView.ViewHolder implements View.OnClickListener, View.OnLongClickListener {
    public TextView tvModelName,
            tvModelTitle,
            tvModelSubtitle,
            tvStatus;
    private ModelItemListener modelItemListener;
    public ImageView headerIcon;

    public View downloadProgressView;

    private ModelItemState modelItemState;

    private TagsLayout tagsLayout;

    public ModelItemHolder(View itemView, ModelItemListener modelItemListener) {
        super(itemView);
        this.modelItemListener = modelItemListener;
        itemView.setOnClickListener(this);
        itemView.setOnLongClickListener(this);
        tvModelName = itemView.findViewById(R.id.tvModelName);
        tvModelTitle = itemView.findViewById(R.id.tvModelTitle);
        tvModelSubtitle = itemView.findViewById(R.id.tvModelSubtitle);
        tvStatus = itemView.findViewById(R.id.tvStatus);
        headerIcon = itemView.findViewById(R.id.header_section_icon);
        downloadProgressView = itemView.findViewById(R.id.download_progress_view);
        tagsLayout = itemView.findViewById(R.id.tagsLayout);
    }

    void bind(HfRepoItem hfRepoItem, ModelItemState modelItemState) {
        String modelName = hfRepoItem.getModelName();
        this.itemView.setTag(hfRepoItem);
        this.modelItemState = modelItemState;
        this.tvModelTitle.setText(modelName);
        tagsLayout.setTags(
                hfRepoItem.getNewTags()
        );
        int drawableId = getDrawableId(modelName);
        if (drawableId != 0) {
            headerIcon.setVisibility(View.VISIBLE);
            headerIcon.setImageResource(drawableId);
            ((ViewGroup)tvModelName.getParent()).setVisibility(View.INVISIBLE);
        } else {
            headerIcon.setVisibility(View.INVISIBLE);
            ((ViewGroup)tvModelName.getParent()).setVisibility(View.VISIBLE);
            String headerText = modelName == null ? "" : modelName.replace("_", "-");
            this.tvModelName.setText(headerText.contains("-") ?
                    headerText.substring(0, headerText.indexOf("-")) :  headerText);
        }
        assert modelItemState != null;
        int downloadState = modelItemState.downloadInfo.downlodaState;
        downloadProgressView.setVisibility(downloadState == DownloadInfo.DownloadSate.DOWNLOADING ? View.VISIBLE : View.GONE);
        switch (downloadState) {
            case DownloadInfo.DownloadSate.NOT_START:
                tvStatus.setText(tvStatus.getResources().getString(R.string.download_not_started));
                break;
            case DownloadInfo.DownloadSate.COMPLETED:
                tvStatus.setText(tvStatus.getResources().getString(R.string.downloaded_click_to_chat));
                break;
            case DownloadInfo.DownloadSate.DOWNLOADING:
                if (TextUtils.equals("Preparing", modelItemState.downloadInfo.progressStage)) {
                    tvStatus.setText(tvStatus.getResources().getString(R.string.download_preparing));
                } else {
                    updateProgress(modelItemState.downloadInfo.progress);
                }
                break;
            case DownloadInfo.DownloadSate.FAILED:
                tvStatus.setText(tvStatus.getResources().getString(R.string.download_failed_click_retry, modelItemState.downloadInfo.errorMessage));
                break;
            case DownloadInfo.DownloadSate.PAUSED:
                tvStatus.setText(tvStatus.getResources().getString(R.string.downloading_paused, modelItemState.downloadInfo.progress * 100));
                break;
            default:
                break;
        }
    }

    @SuppressLint("DefaultLocale")
    public void updateProgress(double progress) {
        tvStatus.setText(itemView.getResources().getString(R.string.downloading_progress, progress * 100));
    }

    @Override
    public void onClick(View v) {
        HfRepoItem hfRepoItem = (HfRepoItem) v.getTag();
        this.modelItemListener.onItemClicked(hfRepoItem);
    }

    @Override
    public boolean onLongClick(View v) {
        PopupMenu popupMenu = new PopupMenu(v.getContext(), tvStatus);
        MenuInflater inflater = popupMenu.getMenuInflater();
        inflater.inflate(R.menu.model_item_context_menu, popupMenu.getMenu());
        popupMenu.setOnMenuItemClickListener(item -> {
            HfRepoItem hfRepoItem = (HfRepoItem) this.itemView.getTag();
            String modelId = hfRepoItem.getModelId();
            if (item.getItemId() == R.id.menu_delete_model) {
                ModelDownloadManager.getInstance(v.getContext()).removeDownload(modelId);
            } else if (item.getItemId() == R.id.menu_pause_download) {
                ModelDownloadManager.getInstance(v.getContext()).pauseDownload(modelId);
            } else if (item.getItemId() == R.id.menu_start_download) {
                ModelDownloadManager.getInstance(v.getContext()).startDownload(modelId);
            }
            return true;
        });
        ModelItemState modelItemState = this.modelItemState;
        int downloadState =  modelItemState.downloadInfo.downlodaState;
        if (downloadState != DownloadInfo.DownloadSate.COMPLETED
                && downloadState != DownloadInfo.DownloadSate.PAUSED
                && downloadState != DownloadInfo.DownloadSate.FAILED) {
            popupMenu.getMenu().findItem(R.id.menu_delete_model).setVisible(false);
        }
        if (downloadState != DownloadInfo.DownloadSate.DOWNLOADING) {
            popupMenu.getMenu().findItem(R.id.menu_pause_download).setVisible(false);
        }
        if (downloadState != DownloadInfo.DownloadSate.PAUSED
                && downloadState != DownloadInfo.DownloadSate.NOT_START
                && downloadState != DownloadInfo.DownloadSate.FAILED
        ) {
            popupMenu.getMenu().findItem(R.id.menu_start_download).setVisible(false);
        }
        popupMenu.show();
        return true;
    }
}
