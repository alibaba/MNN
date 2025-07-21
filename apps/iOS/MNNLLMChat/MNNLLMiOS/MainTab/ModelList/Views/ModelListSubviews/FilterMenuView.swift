//
//  FilterMenuView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import SwiftUI

struct FilterMenuView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var viewModel = ModelListViewModel()
    @Binding var selectedTags: Set<String>
    @Binding var selectedCategories: Set<String>
    @Binding var selectedVendors: Set<String>
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("filter.byTag")
                            .font(.headline)
                            .fontWeight(.semibold)
                        
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 8) {
                            ForEach(viewModel.allTags.sorted(), id: \.self) { tag in
                                FilterOptionRow(
                                    text: TagTranslationManager.shared.getLocalizedTag(tag),
                                    isSelected: selectedTags.contains(tag)
                                ) {
                                    if selectedTags.contains(tag) {
                                        selectedTags.remove(tag)
                                    } else {
                                        selectedTags.insert(tag)
                                    }
                                }
                            }
                        }
                    }
                    
                    Divider()
                    
                    VStack(alignment: .leading, spacing: 12) {
                        Text("filter.byVendor")
                            .font(.headline)
                            .fontWeight(.semibold)
                        
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 8) {
                            ForEach(viewModel.allVendors.sorted(), id: \.self) { vendor in
                                FilterOptionRow(
                                    text: vendor,
                                    isSelected: selectedVendors.contains(vendor)
                                ) {
                                    if selectedVendors.contains(vendor) {
                                        selectedVendors.remove(vendor)
                                    } else {
                                        selectedVendors.insert(vendor)
                                    }
                                }
                            }
                        }
                    }
                    
                    Spacer(minLength: 100)
                }
                .padding()
            }
            .navigationTitle("filter.title")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("button.clear") {
                        selectedTags.removeAll()
                        selectedCategories.removeAll()
                        selectedVendors.removeAll()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("button.done") {
                        dismiss()
                    }
                }
            }
        }
        .onAppear {
            Task {
                await viewModel.fetchModels()
            }
        }
    }
}
