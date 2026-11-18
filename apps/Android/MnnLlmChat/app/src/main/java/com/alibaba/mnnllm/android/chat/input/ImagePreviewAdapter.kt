package com.alibaba.mnnllm.android.chat.input

import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.ImageUtils

class ImagePreviewAdapter(
    private val onDeleteClick: (Uri) -> Unit
) : RecyclerView.Adapter<ImagePreviewAdapter.ViewHolder>() {
    private val images = mutableListOf<Uri>()

    fun addImage(uri: Uri) {
        images.add(uri)
        notifyItemInserted(images.size - 1)
    }

    fun addImages(uris: List<Uri>) {
        val startPos = images.size
        images.addAll(uris)
        notifyItemRangeInserted(startPos, uris.size)
    }

    fun removeImage(uri: Uri) {
        val index = images.indexOf(uri)
        if (index != -1) {
            images.removeAt(index)
            notifyItemRemoved(index)
        }
    }

    fun getImages(): List<Uri> {
        return ArrayList(images)
    }

    fun clear() {
        images.clear()
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_image_preview, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(images[position])
    }

    override fun getItemCount(): Int = images.size

    override fun onViewRecycled(holder: ViewHolder) {
        holder.imageRequest.clear()
        super.onViewRecycled(holder)
    }

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val imageView: ImageView = itemView.findViewById(R.id.iv_preview)
        internal val imageRequest = ImageUtils.ImageRequest(imageView, ImageUtils.PREVIEW_MAX_DIMENSION)
        private val deleteView: View? = itemView.findViewById(R.id.iv_delete)

        fun bind(uri: Uri) {
            imageRequest.bind(uri)
            deleteView?.setOnClickListener {
                onDeleteClick(uri)
            }
            // Also allow clicking the image to delete if no explicit delete button found?
            // Or maybe just show it.
            if (deleteView == null) {
                itemView.setOnClickListener {
                    onDeleteClick(uri)
                }
            }
        }
    }
}
