//
//  Created by Alex.M on 28.06.2022.
//

import Foundation
import SwiftUI
import ExyteChat

struct ChatExampleView: View {
    @Environment(\.colorScheme) var colorScheme
    @Environment(\.presentationMode) private var presentationMode

    @StateObject private var viewModel: ChatExampleViewModel
    
    private let title: String
    private let recorderSettings = RecorderSettings(sampleRate: 16000, numberOfChannels: 1, linearPCMBitDepth: 16)
    
    init(viewModel: ChatExampleViewModel = ChatExampleViewModel(), title: String) {
        _viewModel = StateObject(wrappedValue: viewModel)
        self.title = title
    }
    
    var body: some View {
        ChatView(messages: viewModel.messages, chatType: .conversation) { draft in
            viewModel.send(draft: draft)
        }
        .enableLoadMore(pageSize: 3) { message in
            viewModel.loadMoreMessage(before: message)
        }
        .messageUseMarkdown(true)
        .setRecorderSettings(recorderSettings)
        .navigationBarBackButtonHidden()
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button { presentationMode.wrappedValue.dismiss() } label: {
                    Image("backArrow", bundle: .current)
                        .renderingMode(.template)
                        .foregroundStyle(colorScheme == .dark ? .white : .black)
                }
            }

            ToolbarItem(placement: .navigationBarLeading) {
                HStack {
                    if let url = viewModel.chatCover {
                        CachedAsyncImage(url: url, urlCache: .shared) { phase in
                            switch phase {
                            case .success(let image):
                                image
                                    .resizable()
                                    .scaledToFill()
                            default:
                                Rectangle().fill(Color(hex: "AFB3B8"))
                            }
                        }
                        .frame(width: 35, height: 35)
                        .clipShape(Circle())
                    }

                    VStack(alignment: .leading, spacing: 0) {
                        Text(viewModel.chatTitle)
                            .fontWeight(.semibold)
                            .font(.headline)
                            .foregroundStyle(colorScheme == .dark ? .white : .black)
                        Text(viewModel.chatStatus)
                            .font(.footnote)
                            .foregroundColor(Color(hex: "AFB3B8"))
                    }
                    Spacer()
                }
                .padding(.leading, 10)
            }
        }
        .onAppear(perform: viewModel.onStart)
        .onDisappear(perform: viewModel.onStop)
    }
}

extension Color {
    static var exampleBlue = Color(hex: "#4962FF")
    static var examplePickerBg = Color(hex: "1F1F1F")
}
