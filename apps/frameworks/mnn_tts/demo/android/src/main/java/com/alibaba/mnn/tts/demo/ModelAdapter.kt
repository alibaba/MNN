package com.alibaba.mnn.tts.demo

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.io.File

class ModelAdapter(
    private val onModelSelected: (String, ModelConfig) -> Unit,
    private val onSpeakerSelected: (String) -> Unit,
    private val onPlayClicked: (String, String) -> Unit
) : RecyclerView.Adapter<ModelAdapter.ModelViewHolder>() {

    private var models: List<String> = emptyList()
    private var selectedPosition = -1

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
        holder.bind(modelPath, position == selectedPosition)
        
        holder.itemView.setOnClickListener {
            if (selectedPosition != holder.adapterPosition) {
                val previousSelected = selectedPosition
                selectedPosition = holder.adapterPosition
                notifyItemChanged(previousSelected)
                notifyItemChanged(selectedPosition)
                
                // Notify selection
                holder.loadConfig(modelPath)?.let { config ->
                    onModelSelected(modelPath, config)
                }
            }
        }
        
        holder.setSpeakerListener { speakerId ->
            if (position == selectedPosition) {
                onSpeakerSelected(speakerId)
            }
        }

        holder.setPlayListener { modelPath, speakerId ->
            onPlayClicked(modelPath, speakerId)
        }
    }

    override fun getItemCount(): Int = models.size

    class ModelViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val modelNameText: TextView = itemView.findViewById(R.id.modelNameText)
        private val modelDescText: TextView = itemView.findViewById(R.id.modelDescText)
        private val modelRadioButton: android.widget.RadioButton = itemView.findViewById(R.id.modelRadioButton)
        private val voiceSpinner: android.widget.Spinner = itemView.findViewById(R.id.voiceSpinner)
        private val playButton: android.view.View = itemView.findViewById(R.id.playButton)
        private val cardContent: android.view.View = itemView.findViewById(R.id.cardContent)
        
        private var currentModelPath: String? = null
        private var currentConfig: ModelConfig? = null
        private var speakerListener: ((String) -> Unit)? = null
        private var playListener: ((String, String) -> Unit)? = null
        private var currentSpeakerIndex: Int = 0

        fun bind(modelPath: String, isSelected: Boolean) {
            if (currentModelPath != modelPath) {
                currentModelPath = modelPath
                currentConfig = null
                currentSpeakerIndex = 0
                voiceSpinner.adapter = null
            }
            
            val file = File(modelPath)
            modelNameText.text = file.name
            modelDescText.text = modelPath
            modelRadioButton.isChecked = isSelected
            
            // Reload config for each bind if needed (or rely on loadConfig cache)
            val config = loadConfig(modelPath)

            // Highlight selected item
            if (isSelected) {
                cardContent.setBackgroundResource(R.drawable.bg_filter_border)
                playButton.visibility = View.VISIBLE
                
                // Only show speaker selection if config has speakers
                if (config != null && config.speakers.isNotEmpty()) {
                    voiceSpinner.visibility = View.VISIBLE
                    setupSpinner(config)
                } else {
                    voiceSpinner.visibility = View.GONE
                    voiceSpinner.adapter = null
                }
            } else {
                cardContent.background = null
                voiceSpinner.visibility = View.GONE
                playButton.visibility = View.GONE
                voiceSpinner.adapter = null
            }

            playButton.setOnClickListener {
                val speakerId = if (config != null && config.speakers.isNotEmpty()) {
                    config.speakers[currentSpeakerIndex]
                } else ""
                playListener?.invoke(modelPath, speakerId)
            }
        }
        
        fun loadConfig(modelPath: String): ModelConfig? {
            if (currentConfig != null) return currentConfig
            try {
                val configFile = File(modelPath, "config.json")
                if (configFile.exists()) {
                    val content = configFile.readText()
                    val json = org.json.JSONObject(content)
                    val speakers = mutableListOf<String>()
                    if (json.has("speakers")) {
                        val arr = json.getJSONArray("speakers")
                        for (i in 0 until arr.length()) speakers.add(arr.getString(i))
                    }
                    val languages = mutableListOf<String>()
                    if (json.has("languages")) {
                        val arr = json.getJSONArray("languages")
                        for (i in 0 until arr.length()) languages.add(arr.getString(i))
                    }
                    currentConfig = ModelConfig(speakers, languages)
                } else {
                    currentConfig = ModelConfig()
                }
            } catch (e: Exception) {
                currentConfig = ModelConfig()
            }
            return currentConfig
        }

        private fun setupSpinner(config: ModelConfig?) {
            config ?: return
            if (config.speakers.isEmpty()) {
                voiceSpinner.visibility = View.GONE
                return
            }
            
            val adapter = android.widget.ArrayAdapter(itemView.context, R.layout.spinner_item_dark, config.speakers)
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            voiceSpinner.adapter = adapter
            voiceSpinner.setSelection(currentSpeakerIndex)
            
            voiceSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                    currentSpeakerIndex = position
                    val speakerId = config.speakers[position]
                    speakerListener?.invoke(speakerId)
                }
                override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
            }
        }
        
        fun setSpeakerListener(listener: (String) -> Unit) {
            this.speakerListener = listener
        }

        fun setPlayListener(listener: (String, String) -> Unit) {
            this.playListener = listener
        }
    }
}





























