package com.alibaba.mnnllm.android.chat.chatlist

import android.net.Uri
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.ImageUtils
import com.alibaba.mnnllm.android.widgets.FullScreenImageViewer

class ChatImageAdapter(private val images: List<Uri>) : RecyclerView.Adapter<ChatImageAdapter.ViewHolder>() {
    companion object {
        private const val TAG = "ChatImageAdapter"
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_chat_image, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(images[position], position)
    }

    override fun getItemCount(): Int = images.size

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val imageView: ImageView = itemView.findViewById(R.id.chat_image_item)
        private var boundPosition: Int = RecyclerView.NO_POSITION

        fun bind(uri: Uri, position: Int) {
            boundPosition = position
            try {
                imageView.setImageURI(uri)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to decode preview image: $uri", e)
                imageView.setImageDrawable(null)
            }
            imageView.setOnClickListener {
                FullScreenImageViewer.showImagePopup(itemView.context, images, boundPosition, false)
            }
            imageView.setOnLongClickListener {
                it.performHapticFeedback(android.view.HapticFeedbackConstants.LONG_PRESS)
                ImageUtils.showImageMenu(itemView.context, uri, false)
                true
            }
        }
    }
}
