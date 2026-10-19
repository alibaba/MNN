import SwiftUI
import UIKit

/// Text handed to a sheet opened from the message menu.
struct MessageSheetText: Identifiable {
    let id = UUID()
    let messageId: String
    let text: String
}

/// Edits a sent question; sending replaces it and every later turn.
struct EditMessageSheet: View {
    @State var text: String
    let onSend: (String) -> Void
    @Environment(\.dismiss) private var dismiss
    @FocusState private var isFocused: Bool

    private var trimmed: String { text.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 10) {
                TextEditor(text: $text)
                    .focused($isFocused)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
                Text(AppText.text("chat.edit.note"))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .navigationTitle(AppText.text("chat.edit.title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(AppText.text("Cancel")) { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(AppText.text("chat.edit.send")) {
                        onSend(trimmed)
                        dismiss()
                    }
                    .disabled(trimmed.isEmpty)
                }
            }
            .onAppear { isFocused = true }
        }
        .presentationDetents([.medium, .large])
    }
}

/// Shows a message as plain, freely selectable text.
struct SelectableTextSheet: View {
    let text: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            SelectableTextView(text: text)
                .navigationTitle(AppText.text("chat.menu.select"))
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button(AppText.text("Done")) { dismiss() }
                    }
                }
        }
    }
}

private struct SelectableTextView: UIViewRepresentable {
    let text: String

    func makeUIView(context _: Context) -> UITextView {
        let view = UITextView()
        view.isEditable = false
        view.isSelectable = true
        view.font = .preferredFont(forTextStyle: .body)
        view.adjustsFontForContentSizeCategory = true
        view.textContainerInset = UIEdgeInsets(top: 16, left: 12, bottom: 16, right: 12)
        view.text = text
        return view
    }

    func updateUIView(_ view: UITextView, context _: Context) {
        if view.text != text { view.text = text }
    }
}
