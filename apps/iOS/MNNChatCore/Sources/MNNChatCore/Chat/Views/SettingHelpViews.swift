import SwiftUI

/// A setting's label followed by a small "?" that explains it.
struct SettingLabel: View {
    let titleKey: String
    let helpKey: String
    /// Greys the title for a disabled setting; the "?" stays usable.
    let isDimmed: Bool

    init(_ titleKey: String, help helpKey: String, isDimmed: Bool = false) {
        self.titleKey = titleKey
        self.helpKey = helpKey
        self.isDimmed = isDimmed
    }

    var body: some View {
        HStack(spacing: 4) {
            Text(LocalizedStringKey(titleKey))
                .foregroundStyle(isDimmed ? .secondary : .primary)
            SettingHelpButton(titleKey: titleKey, helpKey: helpKey)
        }
    }
}

/// A toggle whose help stays available while the switch is disabled.
/// (`.disabled` on a whole row would also disable the "?" inside it.)
struct SettingToggleRow: View {
    let titleKey: String
    let helpKey: String
    @Binding var isOn: Bool
    var isDisabled = false

    var body: some View {
        HStack {
            SettingLabel(titleKey, help: helpKey, isDimmed: isDisabled)
            Spacer(minLength: 8)
            Toggle("", isOn: $isOn)
                .labelsHidden()
                .disabled(isDisabled)
        }
    }
}

/// Small question-mark button; shows what a setting means and what changing it does.
struct SettingHelpButton: View {
    let titleKey: String
    let helpKey: String
    /// Bumped per tap. A counter (not a Bool) cannot get stuck "presented"
    /// if one presentation fails, so every tap gets a fresh attempt.
    @State private var requestID = 0

    var body: some View {
        Button { requestID &+= 1 } label: {
            Image(systemName: "questionmark.circle")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .padding(4) // larger hit area without a larger icon
                .contentShape(Rectangle())
        }
        // Borderless keeps a tap on "?" from also toggling or opening the row.
        .buttonStyle(.borderless)
        .accessibilityLabel(String(format: AppText.text("help.accessibility"), AppText.text(titleKey)))
        .background(HelpPopoverPresenter(requestID: requestID,
                                         title: AppText.text(titleKey), message: AppText.text(helpKey)))
    }
}

private struct HelpBubble: View {
    let title: String
    let message: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding()
    }
}

/// Marks our bubble so a new "?" tap can replace one that is still showing.
private final class HelpBubbleController: UIHostingController<AnyView> {}

/// Presents the help bubble as a UIKit popover. SwiftUI's `arrowEdge` is
/// ignored on iOS, so a "?" near the bottom of the screen got a clipped
/// bubble below it. This shows the bubble below the button by default and
/// above it when there is not enough room underneath.
private struct HelpPopoverPresenter: UIViewRepresentable {
    let requestID: Int
    let title: String
    let message: String

    func makeUIView(context _: Context) -> UIView {
        let anchor = UIView()
        anchor.isUserInteractionEnabled = false
        anchor.backgroundColor = .clear
        return anchor
    }

    func updateUIView(_ anchor: UIView, context: Context) {
        let coordinator = context.coordinator
        guard requestID != coordinator.handledRequestID else { return }
        coordinator.handledRequestID = requestID
        let title = title, message = message
        DispatchQueue.main.async { coordinator.present(from: anchor, title: title, message: message) }
    }

    func makeCoordinator() -> Coordinator { Coordinator(handledRequestID: requestID) }

    final class Coordinator: NSObject, UIPopoverPresentationControllerDelegate {
        var handledRequestID: Int

        init(handledRequestID: Int) { self.handledRequestID = handledRequestID }

        func present(from anchor: UIView, title: String, message: String) {
            // Start from the window's root: the anchor's own responder chain can
            // end in a controller that is not part of the presentation chain
            // (e.g. inside a nested List), where `present` silently does nothing.
            guard let window = anchor.window, var presenter = window.rootViewController else { return }
            while let next = presenter.presentedViewController, !next.isBeingDismissed { presenter = next }
            if presenter is HelpBubbleController, let parent = presenter.presentingViewController {
                presenter.dismiss(animated: false)
                presenter = parent
            }

            let width = min(320, window.bounds.width - 32)
            let content = HelpBubbleController(rootView: AnyView(HelpBubble(title: title, message: message)
                .frame(width: width, alignment: .leading)))
            content.view.backgroundColor = .clear
            let height = content.sizeThatFits(in: CGSize(width: width, height: .greatestFiniteMagnitude)).height
            content.preferredContentSize = CGSize(width: width, height: height)
            content.modalPresentationStyle = .popover

            let frame = anchor.convert(anchor.bounds, to: window)
            let roomBelow = window.bounds.height - window.safeAreaInsets.bottom - frame.maxY
            if let popover = content.popoverPresentationController {
                popover.sourceView = anchor
                popover.sourceRect = anchor.bounds
                // .up = arrow on top, bubble below the button.
                popover.permittedArrowDirections = roomBelow >= height + 24 ? .up : .down
                popover.delegate = self
            }
            presenter.present(content, animated: true)
        }

        /// Keep the bubble on iPhone instead of adapting to a full sheet.
        func adaptivePresentationStyle(for _: UIPresentationController,
                                       traitCollection _: UITraitCollection) -> UIModalPresentationStyle {
            .none
        }
    }
}

enum SettingHelp {
    /// Help for the sampler sliders, which are built from their titles.
    static func key(forTitle title: String) -> String? {
        switch title {
        case "Temperature": "help.temperature"
        case "Top K": "help.topK"
        case "Top P": "help.topP"
        case "Min P": "help.minP"
        case "TFS-Z": "help.tfs"
        case "Typical": "help.typical"
        case "Penalty": "help.penalty"
        case "N-gram Size": "help.ngramSize"
        case "N-gram Factor": "help.ngramFactor"
        default: nil
        }
    }
}
