// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.content.Context
import android.util.AttributeSet
import android.view.LayoutInflater
import android.widget.ImageView
import android.widget.RelativeLayout
import android.widget.TextView
import com.alibaba.mnnllm.android.R

class DropDownLineView @JvmOverloads constructor (
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : RelativeLayout(context, attrs, defStyleAttr) {

    private lateinit var labelTextView: TextView
    private lateinit var valueTextView: TextView
    private lateinit var iconImageView: ImageView
    private var onDropdownItemSelected: ((Int, Any) -> Unit)? = null
    private var dropDownItems: List<Any> = emptyList()
    private var itemToString: (Any) -> String = { it.toString() }
    private var selectedIndex = 0

    init {
        initView(context)
        initAttributes(context, attrs)
        setOnClickListener { showDropDownMenu() }
    }

    private fun initView(context: Context) {
        val inflater = context.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        inflater.inflate(R.layout.custom_drop_down_line_view, this, true)

        labelTextView = findViewById(R.id.tv_label_sampler_type)
        valueTextView = findViewById(R.id.tv_sampler_type_value)
        iconImageView = findViewById(R.id.iv_sampler_type_icon)
    }



    private fun initAttributes(context: Context, attrs: AttributeSet?) {
        val typedArray = context.obtainStyledAttributes(attrs, R.styleable.DropDownLineView)
        val labelText = typedArray.getString(R.styleable.DropDownLineView_labelText)
        val valueText = typedArray.getString(R.styleable.DropDownLineView_valueText)
        val iconResId = typedArray.getResourceId(R.styleable.DropDownLineView_icon, R.drawable.baseline_arrow_drop_down_24)

        labelTextView.text = labelText
        valueTextView.text = valueText
        iconImageView.setImageResource(iconResId)

        typedArray.recycle()
    }

    fun setLabelText(text: String) {
        labelTextView.text = text
    }

    fun setValueText(text: String) {
        valueTextView.text = text
    }

    fun setIcon(resId: Int) {
        iconImageView.setImageResource(resId)
    }

    fun setDropDownItems(
        items: List<Any>,
        itemToString: (Any) -> String = { it.toString() },
        onDropdownItemSelected: (Int, Any) -> Unit
    ) {
        this.dropDownItems = items
        this.itemToString = itemToString
        this.onDropdownItemSelected = onDropdownItemSelected
        updateSelectIndex()
    }

    fun setCurrentItem(item: Any) {
        valueTextView.text = itemToString(item)
    }

    private fun updateSelectIndex() {
        dropDownItems.forEach({ item ->
            if (itemToString(item) == valueTextView.text.toString()) {
                selectedIndex = dropDownItems.indexOf(item)
                return@forEach
            }
        })
    }

    private fun showDropDownMenu() {
        updateSelectIndex()
        if (dropDownItems.isEmpty()) return

        DropDownMenuHelper.showDropDownMenu(
            context = context,
            anchorView = valueTextView,
            items = dropDownItems,
            itemToString = itemToString,
            currentIndex = selectedIndex,
            onItemSelected = { index, selectedValue ->
                valueTextView.text = itemToString(selectedValue)
                onDropdownItemSelected?.invoke(index, selectedValue)
            }
        )
    }
}