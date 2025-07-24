package com.alibaba.mnnllm.android.modelmarket

import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import androidx.lifecycle.Lifecycle
import androidx.viewpager2.adapter.FragmentStateAdapter

class ModelMarketPagerAdapter(
    fragmentManager: FragmentManager,
    lifecycle: Lifecycle
) : FragmentStateAdapter(fragmentManager, lifecycle) {

    private var categories: List<String> = emptyList()

    fun setCategories(categories: List<String>) {
        this.categories = categories
        notifyDataSetChanged()
    }

    override fun getItemId(position: Int): Long = categories[position].hashCode().toLong()

    override fun containsItem(itemId: Long): Boolean = categories.any { it.hashCode().toLong() == itemId }

    override fun getItemCount(): Int = categories.size

    override fun createFragment(position: Int): Fragment {
        return ModelMarketPageFragment.newInstance(categories[position])
    }
} 