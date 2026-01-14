package com.alibaba.mnnllm.android.chat.chatlist

import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.widgets.FullScreenImageViewer

class ChatImageAdapter(private val images: List<Uri>) : RecyclerView.Adapter<ChatImageAdapter.ViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_chat_image, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(images[position])
    }

    override fun getItemCount(): Int = images.size

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val imageView: ImageView = itemView.findViewById(R.id.chat_image_item)

        fun bind(uri: Uri) {
            imageView.setImageURI(uri)
            imageView.setOnClickListener {
                FullScreenImageViewer.showImagePopup(itemView.context, images, adapterPosition)
            }
        }
    }
}
