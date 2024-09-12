//
//  ContentView.swift
//  mnn-llm
//
//  Created by wangzhaode on 2023/12/14.
//

import Combine
import SwiftUI

class ChatViewModel: ObservableObject {
    @Published var messages: [Message] = []
    @Published var isModelLoaded = false // 模型是否加载完成
    @Published var isProcessing: Bool = false // 标志表示是否有正在处理的LLM响应
    private var llm: LLMInferenceEngineWrapper?

    init() {
        self.messages.append(Message(id: UUID(), text: " 模型加载中, 请稍等 ...", isUser: false))
        llm = LLMInferenceEngineWrapper { [weak self] success in
            DispatchQueue.main.async {
                self?.isModelLoaded = success
                var loadresult = "模型加载完毕！"
                if !success {
                    loadresult = "模型加载失败！"
                }
                self?.messages.append(Message(id: UUID(), text: loadresult, isUser: false))
            }
        }
    }

    func sendInput(_ input: String) {
        // 将用户输入作为新消息添加
        let userMessage = Message(id: UUID(), text: input, isUser: true)
        DispatchQueue.main.async {
            self.messages.append(userMessage)
        }
        isProcessing = true
        // 在后台线程处理耗时的输入
        DispatchQueue.global(qos: .userInitiated).async {
            self.llm?.processInput(input) { [weak self] output in
                // 切换回主线程来更新UI
                DispatchQueue.main.async {
                    if (output.contains("<eop>")) {
                        self?.isProcessing = false
                    } else {
                        self?.appendResponse(output)
                    }
                }
            }
        }
    }
    
    private func appendResponse(_ output: String) {
        if let lastMessage = messages.last, !lastMessage.isUser {
            // 创建一个更新后的消息
            var updatedMessage = messages[messages.count - 1]
            updatedMessage.text += output
            // 替换数组中的旧消息
            self.messages[messages.count - 1] = updatedMessage
        } else {
            let newMessage = Message(id: UUID(), text: output, isUser: false)
            self.messages.append(newMessage)
        }
    }
}


struct Message: Identifiable, Equatable {
    let id: UUID
    var text: String
    let isUser: Bool
}

struct ChatBubble: View {
    let message: Message
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
            }
            
            Text(message.text)
                .padding(10)
                .foregroundColor(message.isUser ? .white : .black)
                .background(message.isUser ? Color.blue : Color.gray.opacity(0.2))
                .cornerRadius(10)
                .frame(maxWidth: 400, alignment: message.isUser ? .trailing : .leading)
            
            if !message.isUser {
                Spacer()
            }
        }
        .transition(.scale(scale: 0, anchor: message.isUser ? .bottomTrailing : .bottomLeading))
    }
}

struct ChatView: View {
    @StateObject var viewModel = ChatViewModel()
    @State private var inputText: String = ""
    
    var body: some View {
        NavigationView {  // 包裹在 NavigationView 中
            VStack {
                ScrollView {
                    ScrollViewReader { scrollView in
                        VStack(alignment: .leading, spacing: 10) {
                            ForEach(viewModel.messages) { message in
                                ChatBubble(message: message)
                            }
                        }
                        .padding(.horizontal)
                        .onChange(of: viewModel.messages) { _ in
                            scrollView.scrollTo(viewModel.messages.last?.id, anchor: .bottom)
                        }
                    }
                }

                HStack {
                    TextField("Type a message...", text: $inputText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .frame(minHeight: 44)

                    Button(action: {
                        viewModel.sendInput(inputText)
                        inputText = ""
                    }) {
                        Image(systemName: "arrow.up.circle.fill")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 44, height: 44)
                    }
                    .disabled(inputText.isEmpty || viewModel.isProcessing || !viewModel.isModelLoaded)
                }
                .padding()
            }
            .navigationBarTitle("mnn-llm", displayMode: .inline)  // 设置标题
        }
    }
}

extension String {
    var isBlank: Bool {
        return allSatisfy({ $0.isWhitespace })
    }
}

struct ChatView_Previews: PreviewProvider {
    static var previews: some View {
        ChatView()
    }
}
