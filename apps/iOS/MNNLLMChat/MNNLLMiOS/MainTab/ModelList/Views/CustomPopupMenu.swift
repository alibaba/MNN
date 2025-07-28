//
//  CustomPopupMenu.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/6/30.
//

import SwiftUI

struct CustomPopupMenu: View {
    @Binding var isPresented: Bool
    @Binding var selectedSource: ModelSource
    let anchorFrame: CGRect
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .top) {
                
                Color.black.opacity(0.3)
                    .frame(maxWidth: .infinity)
                    .frame(height: UIScreen.main.bounds.height - anchorFrame.maxY)
                    .offset(y: anchorFrame.maxY - 10)
                    .onTapGesture {
                        isPresented = false
                    }
                
                VStack(spacing: 0) {
                    ForEach(ModelSource.allCases) { source in
                        Button {
                            selectedSource = source
                            ModelSourceManager.shared.updateSelectedSource(source)
                            isPresented = false
                        } label: {
                            HStack {
                                Text(source.description)
                                    .font(.system(size: 12, weight: .regular))
                                    .foregroundColor(source == selectedSource ? .primaryBlue : .black)
                                Spacer()
                                if source == selectedSource {
                                    Image(systemName: "checkmark.circle")
                                        .foregroundColor(.primaryBlue)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.white)
                        }
                        Divider()
                    }
                }
                .background(Color.white)
                .cornerRadius(8)
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 5)
                .frame(width: geometry.size.width)
                .position(
                    x: geometry.size.width / 2,
                    y: anchorFrame.maxY - 24
                )
            }
        }
        .transition(.opacity)
        .animation(.spring(response: 0.3, dampingFraction: 0.8, blendDuration: 0), value: isPresented)
    }
}

