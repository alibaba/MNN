//
//  SideMenuView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/16.
//

import SwiftUI

struct SideMenuView: View {
    @Binding var isOpen: Bool
    @Binding var selectedHistory: ChatHistory?
    @Binding var histories: [ChatHistory]
    @Binding var navigateToMainSettings: Bool
    
    @State private var showingAlert = false
    @State private var historyToDelete: ChatHistory?
    @State private var navigateToSettings = false
    
    @State private var dragOffset: CGFloat = 0
    
    var body: some View {
        ZStack {
            GeometryReader { geometry in
                VStack {
                    HStack {
                        Text(NSLocalizedString("ChatHistroyTitle", comment: "Chat Histroy Title"))
                            .fontWeight(.medium)
                            .font(.system(size: 20))
                        Spacer()
                    }
                    .padding(.top, 80)
                    .padding(.leading, 12)
                    
                    List {
                        ForEach(histories.sorted(by: { $0.updatedAt > $1.updatedAt })) { history in
                            
                                ChatHistoryItemView(history: history)
                                    .onTapGesture {
                                        selectedHistory = history
                                        isOpen = false
                                    }
                                .onLongPressGesture {
                                    historyToDelete = history
                                    showingAlert = true
                                }
                                .listRowBackground(Color.sidemenuBg)
                                .listRowSeparator(.hidden)
                        }
                    }
                    .background(Color.sidemenuBg)
                    .listStyle(PlainListStyle())
                    
                    Spacer()
                    
                    HStack {
                        Button(action: {
                            isOpen = false
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                navigateToMainSettings = true
                            }
                        }) {
                            HStack {
                                Image(systemName: "gear")
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(width: 20, height: 20)
                            }
                            .foregroundColor(.primary)
                            .padding(.leading)
                        }
                        Spacer()
                    }
                    .padding(EdgeInsets(top: 10, leading: 12, bottom: 30, trailing: 0))
                }
                .background(Color.sidemenuBg)
                .frame(width: geometry.size.width * 0.8)
                .offset(x: isOpen ? 0 : -geometry.size.width * 0.8)
                .animation(.easeOut, value: isOpen)
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            if value.translation.width < 0 {
                                dragOffset = value.translation.width
                            }
                        }
                        .onEnded { value in
                            if value.translation.width < -geometry.size.width * 0.25 {
                                isOpen = false
                            }
                            dragOffset = 0
                        }
                )
                .alert("Delete History", isPresented: $showingAlert) {
                    Button("Cancel", role: .cancel) {}
                    Button(LocalizedStringKey("button.delete"), role: .destructive) {
                        if let history = historyToDelete {
                            deleteHistory(history)
                        }
                    }
                } message: {
                    Text("Are you sure you want to delete this history?")
                }
            }
        }
    }
    
    private func deleteHistory(_ history: ChatHistory) {
        ChatHistoryManager.shared.deleteHistory(history)
        histories.removeAll { $0.id == history.id }
    }
}

struct SettingsFullScreenView: View {
    @Binding var isPresented: Bool
    
    var body: some View {
        NavigationView {
            SettingsView()
                .navigationTitle("Settings")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button(action: {
                            isPresented = false
                        }) {
                            Image(systemName: "xmark")
                                .foregroundColor(.primary)
                                .fontWeight(.medium)
                        }
                    }
                }
        }
        .navigationViewStyle(StackNavigationViewStyle())
    }
}
