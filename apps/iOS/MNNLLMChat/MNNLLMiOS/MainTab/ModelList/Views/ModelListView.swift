//
//  ModelListView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI


struct ModelListView: View {
    @ObservedObject var viewModel: ModelListViewModel
    
    @State private var scrollOffset: CGFloat = 0
    @State private var showHelp = false
    @State private var showUserGuide = false
    
    @State private var downloadSources: ModelSource?
    @State private var selectedSource = ModelSourceManager.shared.selectedSource
    
    @State private var showOptions = false
    @State private var buttonFrame: CGRect = .zero
    
    var body: some View {
        ZStack {
            VStack {
                HStack {
                    Button {
                        showOptions.toggle()
                    } label: {
                        HStack {
                            Text("下载源:")
                                .font(.system(size: 12, weight: .regular))
                                .foregroundColor(showOptions ? .primaryBlue : .black )
                            Text(selectedSource.rawValue)
                                .font(.system(size: 12, weight: .regular))
                                .foregroundColor(showOptions ? .primaryBlue : .black )
                            Image(systemName: "chevron.down")
                                .frame(width: 10, height: 10, alignment: .leading)
                                .scaledToFit()
                                .foregroundColor(showOptions ? .primaryBlue : .black )
                        }
                        .padding(.leading)
                    }
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: 20)
                .background(
                   GeometryReader { geometry in
                       Color.white.onAppear {
                           buttonFrame = geometry.frame(in: .global)
                       }
                   }
                )
                
                List {
                    SearchBar(text: $viewModel.searchText)
                        .listRowInsets(EdgeInsets())
                        .listRowSeparator(.hidden)
                        .padding(.horizontal)
                    
                    ForEach(viewModel.filteredModels, id: \.modelId) { model in
                        
                        ModelRowView(model: model,
                                     viewModel: viewModel,
                                     downloadProgress: viewModel.downloadProgress[model.modelId] ?? 0,
                                     isDownloading: viewModel.currentlyDownloading == model.modelId,
                                     isOtherDownloading: viewModel.currentlyDownloading != nil) {
                            if model.isDownloaded {
                                viewModel.selectModel(model)
                            } else {
                                Task {
                                    await viewModel.downloadModel(model)
                                }
                            }
                        }
                        .listRowSeparator(.hidden)
                        .listRowBackground(viewModel.pinnedModelIds.contains(model.modelId) ? Color.black.opacity(0.05) : Color.clear)
                        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                            SwipeActionsView(model: model, viewModel: viewModel)
                        }
                    }
                }
                .listStyle(.plain)
                .sheet(isPresented: $showHelp) {
                    HelpView()
                }
                .refreshable {
                    await viewModel.fetchModels()
                }
                .alert("Error", isPresented: $viewModel.showError) {
                    Button("OK", role: .cancel) {}
                } message: {
                    Text(viewModel.errorMessage)
                }
                .onAppear {
                    checkFirstLaunch()
                }
                .alert(isPresented: $showUserGuide) {
                    Alert(
                        title: Text("User Guide"),
                        message: Text("""
                This is a local large model application that requires certain performance from your device.
                It is recommended to choose different model sizes based on your device's memory. 
                
                The model recommendations for iPhone are as follows:
                - For 8GB of RAM, models up to 8B are recommended (e.g., iPhone 16 Pro).
                - For 6GB of RAM, models up to 3B are recommended (e.g., iPhone 15 Pro).
                - For 4GB of RAM, models up to 1B or smaller are recommended (e.g., iPhone 13).
                
                Choosing a model that is too large may cause insufficient memory and crashes.
                """),
                        dismissButton: .default(Text("OK"))
                    )
                }
                
                Spacer()
            }
            
            if showOptions {
                CustomPopupMenu(isPresented: $showOptions,
                                selectedSource: $selectedSource,
                                anchorFrame: buttonFrame)
            }
        }
    }
    
    private func checkFirstLaunch() {
        let hasLaunchedBefore = UserDefaults.standard.bool(forKey: "hasLaunchedBefore")
        if !hasLaunchedBefore {
            // Show the user guide alert
            showUserGuide = true
            // Set the flag to true so it doesn't show again
            UserDefaults.standard.set(true, forKey: "hasLaunchedBefore")
        }
    }
}
