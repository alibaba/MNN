package com.alibaba.mnnllm.android.modelmarket

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R

class ModelMarketPageFragment : Fragment() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: ModelMarketAdapter
    private lateinit var viewModel: ModelMarketViewModel
    private var category: String = ""

    companion object {
        private const val ARG_CATEGORY = "category"

        fun newInstance(category: String): ModelMarketPageFragment {
            val fragment = ModelMarketPageFragment()
            val args = Bundle()
            args.putString(ARG_CATEGORY, category)
            fragment.arguments = args
            return fragment
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        category = arguments?.getString(ARG_CATEGORY) ?: ""
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_model_market_page, container, false)
        recyclerView = view.findViewById(R.id.recycler_view)
        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        // Get the shared ViewModel from parent fragment
        viewModel = ViewModelProvider(requireParentFragment())[ModelMarketViewModel::class.java]
        
        adapter = ModelMarketAdapter(parentFragment as ModelMarketFragment)
        recyclerView.layoutManager = LinearLayoutManager(context)
        recyclerView.adapter = adapter

        // Observe filtered models for this category
        viewModel.models.observe(viewLifecycleOwner) { models ->
            val filteredModels = if (category == "All") {
                models
            } else {
                models.filter { it.modelMarketItem.categories.contains(category) }
            }
            adapter.submitList(filteredModels)
        }

        viewModel.progressUpdate.observe(viewLifecycleOwner) { (modelId, downloadInfo) ->
            android.util.Log.d("ModelMarketPageFragment", "progressUpdate observed: modelId=$modelId, progress=${downloadInfo.progress}")
            adapter.updateProgress(modelId, downloadInfo)
        }

        viewModel.itemUpdate.observe(viewLifecycleOwner) { modelId ->
            android.util.Log.d("ModelMarketPageFragment", "[itemUpdate] Observed for modelId: $modelId. Notifying adapter.")
            adapter.updateItem(modelId)
        }
    }
} 