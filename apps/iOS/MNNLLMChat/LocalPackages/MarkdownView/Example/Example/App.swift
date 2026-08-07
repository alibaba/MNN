//
//  App.swift
//  Example
//
//  Created by 秋星桥 on 1/20/25.
//

import SwiftUI

@main
struct TheApp: App {
    var body: some Scene {
        WindowGroup {
            NavigationView {
                Content()
                    .toolbar {
                        ToolbarItem {
                            Button {
                                NotificationCenter.default.post(name: .init("Play"), object: nil)
                            } label: {
                                Image(systemName: "play")
                            }
                        }
                        ToolbarItem {
                            Button {
                                NotificationCenter.default.post(name: .init("Reset"), object: nil)
                            } label: {
                                Image(systemName: "arrow.counterclockwise")
                            }
                        }
                    }
                    .navigationTitle("MarkdownView")
                    .navigationBarTitleDisplayMode(.inline)
            }
            .navigationViewStyle(.stack)
            .frame(minWidth: 200, maxWidth: .infinity)
        }
    }
}

import MarkdownParser
import MarkdownView

final class ContentController: UIViewController {
    let scrollView = UIScrollView()
    let measureLabel = UILabel()

    private var markdownTextView: MarkdownTextView!

    override func viewDidLoad() {
        super.viewDidLoad()

        scrollView.contentInset = .init(top: 64, left: 0, bottom: 64, right: 0)
        view.addSubview(scrollView)

        markdownTextView = MarkdownTextView()
        scrollView.addSubview(markdownTextView)
        markdownTextView.bindContentOffset(from: scrollView)

        measureLabel.numberOfLines = 0
        measureLabel.font = UIFont.preferredFont(forTextStyle: .footnote)
        measureLabel.textColor = .label

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(play),
            name: .init("Play"),
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(resetMarkdown),
            name: .init("Reset"),
            object: nil
        )

        let parser = MarkdownParser()
        let result = parser.parse(testDocument)
        let date = Date()
        markdownTextView.setMarkdownManually(.init(parserResult: result, theme: .default))
        view.setNeedsLayout()
        view.layoutIfNeeded()
        let time = Date().timeIntervalSince(date)
        measureLabel.text = String(format: "Time: %.4f ms", time * 1000)
    }

    private var streamDocument = ""

    @objc func play() {
        print(#function, Date())
        DispatchQueue.global().async { [self] in
            for char in testDocument {
                streamDocument.append(char)
                autoreleasepool {
                    let parser = MarkdownParser()
                    let result = parser.parse(streamDocument)
                    let content = MarkdownTextView.PreprocessedContent(parserResult: result, theme: .default)
                    DispatchQueue.main.asyncAndWait {
                        let date = Date()
                        markdownTextView.setMarkdown(content)
                        self.view.setNeedsLayout()
                        self.view.layoutIfNeeded()
                        let time = Date().timeIntervalSince(date)
                        self.measureLabel.text = String(format: "Time: %.4f ms", time * 1000)
                    }
                }
            }
        }
    }

    @objc func resetMarkdown() {
        markdownTextView.reset()
        view.setNeedsLayout()
    }

    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()

        let date = Date()

        scrollView.frame = view.bounds
        let width = view.bounds.width - 32

        let contentSize = markdownTextView.boundingSize(for: width)
        scrollView.contentSize = contentSize
        markdownTextView.frame = .init(
            x: 16,
            y: 16,
            width: width,
            height: contentSize.height
        )

        measureLabel.removeFromSuperview()
        measureLabel.frame = .init(
            x: 16,
            y: (scrollView.subviews.map(\.frame.maxY).max() ?? 0) + 16,
            width: width,
            height: 50
        )
        scrollView.addSubview(measureLabel)
        scrollView.contentSize = .init(
            width: width,
            height: measureLabel.frame.maxY + 16
        )

        let offset = CGPoint(
            x: 0,
            y: scrollView.contentSize.height - scrollView.frame.height
        )
        _ = offset
        scrollView.setContentOffset(offset, animated: false)

        let time = Date().timeIntervalSince(date)
        measureLabel.text = String(format: "Time: %.4f ms", time * 1000)
    }
}

struct Content: UIViewControllerRepresentable {
    func makeUIViewController(context _: Context) -> ContentController {
        ContentController()
    }

    func updateUIViewController(_: ContentController, context _: Context) {}
}
