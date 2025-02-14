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
    
    @State private var showingAlert = false
    @State private var historyToDelete: ChatHistory?
    
    @State private var dragOffset: CGFloat = 0
    
    var body: some View {
        GeometryReader { geometry in
            VStack {
                HStack {
                    Text(NSLocalizedString("ChatHistroyTitle", comment: "Chat Histroy Title"))
                        .fontWeight(.medium)
                        .font(.system(size: 20))
                    Spacer()
                }
                .padding(.top, 80)
                .padding(.leading)
                
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
                    }
                }
                .background(Color.sidemenuBg)
                .listStyle(PlainListStyle())
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
                Button("Delete", role: .destructive) {
                    if let history = historyToDelete {
                        deleteHistory(history)
                    }
                }
            } message: {
                Text("Are you sure you want to delete this history?")
            }
        }
    }
    
    private func deleteHistory(_ history: ChatHistory) {
        // 更新存储
        ChatHistoryManager.shared.deleteHistory(history)
        // 从历史记录中删除
        histories.removeAll { $0.id == history.id }
    }
}
