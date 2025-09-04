package com.alibaba.mnnllm.android.widgets

import android.content.Context
import android.util.AttributeSet
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
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
    private var isCompactMode: Boolean = false

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
                val compactMode = typedArray.getBoolean(R.styleable.ModelAvatarView_compactMode, false)
                if (!modelName.isNullOrEmpty()) {
                    setModelName(modelName)
                }
                setCompactMode(compactMode)
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
    }

    fun setCompactMode(compactMode: Boolean) {
        isCompactMode = compactMode
        if (compactMode) {
            // 在紧凑模式下移除 ImageView 的 margin 和 CardView 的背景
            val layoutParams = headerIcon.layoutParams as? ViewGroup.MarginLayoutParams
            layoutParams?.setMargins(0, 0, 0, 0)
            headerIcon.layoutParams = layoutParams
            
            // 移除 CardView 的背景
            setCardBackgroundColor(android.graphics.Color.TRANSPARENT)
            cardElevation = 0f
            strokeWidth = 0
        }
    }
}