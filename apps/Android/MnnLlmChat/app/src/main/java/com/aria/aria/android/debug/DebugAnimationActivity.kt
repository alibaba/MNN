package com.alibaba.mnnllm.android.debug

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.constraintlayout.motion.widget.MotionLayout
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.modelmarket.SourceProvider
import com.alibaba.mnnllm.android.modelmarket.SourceSelectionProvider

class DebugAnimationActivity : AppCompatActivity() {

    private lateinit var motionLayout: MotionLayout
    private lateinit var sourceSelectionProvider: SourceSelectionProvider

    private var selectedSourceProvider: SourceProvider? = null

    private val sourceProviders = listOf(
        SourceProvider("HuggingFace", "hf"),
        SourceProvider("Modelscope (魔搭)", "ms"),
        SourceProvider("Modelers (魔乐)", "ml")
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_debug_animation)

        motionLayout = findViewById(R.id.motion_layout)
        val recyclerView: RecyclerView = findViewById(R.id.recycler_view_providers)

        // Set up RecyclerView Adapter
        sourceSelectionProvider = SourceSelectionProvider(sourceProviders) { provider ->
            handleProviderSelection(provider)
        }
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = sourceSelectionProvider

        // Set default selection
        selectedSourceProvider = sourceProviders[1] // Default to Modelscope
        sourceSelectionProvider.setSelected(selectedSourceProvider!!.id)
    }

    private fun handleProviderSelection(sourceProvider: SourceProvider) {
        selectedSourceProvider = sourceProvider
        Toast.makeText(this, getString(R.string.starting_download_from, sourceProvider.name), Toast.LENGTH_SHORT).show()

        // Delay the collapse to show feedback
        Handler(Looper.getMainLooper()).postDelayed({
            // Transition back to the start state
            motionLayout.transitionToState(R.id.collapsed)
        }, 300)
    }
}