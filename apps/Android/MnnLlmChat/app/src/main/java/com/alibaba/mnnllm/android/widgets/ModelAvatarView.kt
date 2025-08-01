package com.alibaba.mnnllm.android.widgets

import android.content.Context
import android.util.AttributeSet
import android.view.LayoutInflater
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.model.ModelUtils
import com.google.android.material.card.MaterialCardView

class ModelAvatarView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : MaterialCardView(context, attrs, defStyleAttr) {

    private val tvModelName: TextView
    private val headerIcon: ImageView

    init {
        LayoutInflater.from(context).inflate(R.layout.view_model_avatar, this, true)
        tvModelName = findViewById(R.id.tvModelName)
        headerIcon = findViewById(R.id.model_icon)
        
        // Read the modelName attribute from XML if available
        if (attrs != null) {
            val typedArray = context.obtainStyledAttributes(
                attrs,
                R.styleable.ModelAvatarView,
                defStyleAttr,
                0
            )
            
            try {
                val modelName = typedArray.getString(R.styleable.ModelAvatarView_modelName)
                if (!modelName.isNullOrEmpty()) {
                    setModelName(modelName)
                }
            } finally {
                typedArray.recycle()
            }
        }
    }

    fun setModelName(modelName: String?) {
        if (modelName.isNullOrEmpty()) {
            tvModelName.text = ""
            tvModelName.visibility = View.VISIBLE
            headerIcon.visibility = View.GONE
            return
        }

        val drawableId = ModelUtils.getDrawableId(modelName)
        if (drawableId != 0) {
            headerIcon.visibility = View.VISIBLE
            headerIcon.setImageResource(drawableId)
            tvModelName.visibility = View.INVISIBLE
        } else {
            headerIcon.visibility = View.INVISIBLE
            val headerText = modelName?.replace("_", "-") ?: ""
            tvModelName.text =
                if (headerText.contains("-")) headerText.substring(
                    0,
                    headerText.indexOf("-")
                ) else headerText
            tvModelName.visibility = View.VISIBLE
        }
//
//        if (name.contains("qwen", ignoreCase = true)) {
//            tvModelName.visibility = View.GONE
//            headerIcon.visibility = View.VISIBLE
//            headerIcon.setImageResource(R.drawable.qwen_icon)
//        } else {
//            tvModelName.visibility = View.VISIBLE
//            headerIcon.visibility = View.GONE
//            tvModelName.text = name.split("-").joinToString("") { it.firstOrNull()?.toString() ?: "" }.take(2).uppercase()
//        }
    }
}