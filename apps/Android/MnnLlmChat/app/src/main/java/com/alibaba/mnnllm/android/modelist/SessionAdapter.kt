package com.alibaba.mnnllm.android.modelist

import android.text.TextUtils
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.model.SessionItem

class SessionAdapter(private val sessions: List<SessionItem>) :
    RecyclerView.Adapter<SessionAdapter.SessionViewHolder>() {

    class SessionViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val tvSessionName: TextView = itemView.findViewById(R.id.tv_session_name)
        val tvSessionId: TextView = itemView.findViewById(R.id.tv_session_id)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SessionViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_chat_session, parent, false)
        return SessionViewHolder(view)
    }

    override fun onBindViewHolder(holder: SessionViewHolder, position: Int) {
        val session = sessions[position]
        
        // Display session name or a default name if empty
        val displayName = if (!TextUtils.isEmpty(session.title)) {
            session.title
        } else {
            holder.itemView.context.getString(R.string.unnamed_session)
        }
        
        holder.tvSessionName.text = displayName
        holder.tvSessionId.text = "Session ID: ${session.sessionId}"
    }

    override fun getItemCount(): Int = sessions.size
} 