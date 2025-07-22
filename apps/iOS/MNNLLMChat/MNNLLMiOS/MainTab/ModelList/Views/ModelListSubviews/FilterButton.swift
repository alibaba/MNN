//
//  FilterButton.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import SwiftUI

struct FilterButton: View {
    @Binding var showFilterMenu: Bool
    @Binding var selectedTags: Set<String>
    @Binding var selectedCategories: Set<String>
    @Binding var selectedVendors: Set<String>
    
    var body: some View {
        Button(action: {
            showFilterMenu.toggle()
        }) {
            Image(systemName: "line.3.horizontal.decrease.circle")
                .font(.system(size: 20))
                .foregroundColor(.primary)
        }
        .sheet(isPresented: $showFilterMenu) {
            FilterMenuView(
                selectedTags: $selectedTags,
                selectedCategories: $selectedCategories,
                selectedVendors: $selectedVendors
            )
            .presentationDetents([.large])
        }
    }
}
