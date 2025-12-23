package com.mnn.tts.demo

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnn.tts.demo.R
import java.io.File

class ModelAdapter(
    private val onModelSelected: (String) -> Unit
) : RecyclerView.Adapter<ModelAdapter.ModelViewHolder>() {

    private var models: List<String> = emptyList()

    fun updateModels(newModels: List<String>) {
        models = newModels
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ModelViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_model, parent, false)
        return ModelViewHolder(view)
    }

    override fun onBindViewHolder(holder: ModelViewHolder, position: Int) {
        val modelPath = models[position]
        val modelFile = File(modelPath)
        holder.bind(modelFile.name, modelPath)
        holder.itemView.setOnClickListener {
            onModelSelected(modelPath)
        }
    }

    override fun getItemCount(): Int = models.size

    class ModelViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val modelNameText: TextView = itemView.findViewById(R.id.modelNameText)
        private val modelPathText: TextView = itemView.findViewById(R.id.modelPathText)

        fun bind(modelName: String, modelPath: String) {
            modelNameText.text = modelName
            modelPathText.text = modelPath
        }
    }
}

