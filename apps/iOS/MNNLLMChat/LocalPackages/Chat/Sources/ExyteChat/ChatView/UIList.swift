//
//  UIList.swift
//  
//
//  Created by Alisa Mylnikova on 24.02.2023.
//

import SwiftUI

public extension Notification.Name {
    static let onScrollToBottom = Notification.Name("onScrollToBottom")
}

struct UIList<MessageContent: View, InputView: View>: UIViewRepresentable {

    typealias MessageBuilderClosure = ChatView<MessageContent, InputView, DefaultMessageMenuAction>.MessageBuilderClosure

    @Environment(\.chatTheme) private var theme

    @ObservedObject var viewModel: ChatViewModel
    @ObservedObject var inputViewModel: InputViewModel

    @Binding var isScrolledToBottom: Bool
    @Binding var shouldScrollToTop: () -> ()
    @Binding var tableContentHeight: CGFloat

    var messageBuilder: MessageBuilderClosure?
    var mainHeaderBuilder: (()->AnyView)?
    var headerBuilder: ((Date)->AnyView)?
    var inputView: InputView

    let type: ChatType
    let showDateHeaders: Bool
    let isScrollEnabled: Bool
    let avatarSize: CGFloat
    let showMessageMenuOnLongPress: Bool
    let tapAvatarClosure: ChatView.TapAvatarClosure?
    let paginationHandler: PaginationHandler?
    let messageUseMarkdown: Bool
    let showMessageTimeView: Bool
    let messageFont: UIFont
    let sections: [MessagesSection]
    let ids: [String]

    @State private var isScrolledToTop = false

    private let updatesQueue = DispatchQueue(label: "updatesQueue", qos: .utility)
    @State private var updateSemaphore = DispatchSemaphore(value: 1)
    @State private var tableSemaphore = DispatchSemaphore(value: 0)

    func makeUIView(context: Context) -> UITableView {
        let tableView = UITableView(frame: .zero, style: .grouped)
        tableView.translatesAutoresizingMaskIntoConstraints = false
        tableView.separatorStyle = .none
        tableView.dataSource = context.coordinator
        tableView.delegate = context.coordinator
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "Cell")
        tableView.transform = CGAffineTransform(rotationAngle: (type == .conversation ? .pi : 0))

        tableView.showsVerticalScrollIndicator = false
        tableView.estimatedSectionHeaderHeight = 1
        tableView.estimatedSectionFooterHeight = UITableView.automaticDimension
        tableView.backgroundColor = UIColor(theme.colors.mainBG)
        tableView.scrollsToTop = false
        tableView.isScrollEnabled = isScrollEnabled

        NotificationCenter.default.addObserver(forName: .onScrollToBottom, object: nil, queue: nil) { _ in
            scrollToBottom(tableView)
        }

        DispatchQueue.main.async {
            shouldScrollToTop = {
                tableView.contentOffset = CGPoint(x: 0, y: tableView.contentSize.height - tableView.frame.height)
            }
        }

        return tableView
    }
    
    func scrollToBottom(_ tableView: UITableView) {
        DispatchQueue.main.async {
            guard tableView.contentSize.height > tableView.frame.height else { return }
            // 180 rot
            let bottomOffset = CGPoint(x: 0, y: 0)
            tableView.setContentOffset(bottomOffset, animated: true)
        }
    }

    func updateUIView(_ tableView: UITableView, context: Context) {
        if !isScrollEnabled {
            DispatchQueue.main.async {
                tableContentHeight = tableView.contentSize.height
            }
        }

        if context.coordinator.sections == sections {
            return
        }
        updatesQueue.async {
            updateSemaphore.wait()

            if context.coordinator.sections == sections {
                updateSemaphore.signal()
                return
            }

            if context.coordinator.sections.isEmpty {
                DispatchQueue.main.async {
                    context.coordinator.sections = sections
                    tableView.reloadData()
                    if !isScrollEnabled {
                        DispatchQueue.main.async {
                            tableContentHeight = tableView.contentSize.height
                        }
                    }
                    updateSemaphore.signal()
                }
                return
            }

            if let lastSection = sections.last {
                context.coordinator.paginationTargetIndexPath = IndexPath(row: lastSection.rows.count - 1, section: sections.count - 1)
            }

            let prevSections = context.coordinator.sections
            let (appliedDeletes, appliedDeletesSwapsAndEdits, deleteOperations, swapOperations, editOperations, insertOperations) = operationsSplit(oldSections: prevSections, newSections: sections)

            // step 1
            // preapare intermediate sections and operations
            //print("1 updateUIView sections:", "\n")
            //print("whole previous:\n", formatSections(prevSections), "\n")
            //print("whole appliedDeletes:\n", formatSections(appliedDeletes), "\n")
            //print("whole appliedDeletesSwapsAndEdits:\n", formatSections(appliedDeletesSwapsAndEdits), "\n")
            //print("whole final sections:\n", formatSections(sections), "\n")

            //print("operations delete:\n", deleteOperations.map { $0.description })
            //print("operations swap:\n", swapOperations.map { $0.description })
            //print("operations edit:\n", editOperations.map { $0.description })
            //print("operations insert:\n", insertOperations.map { $0.description })

            DispatchQueue.main.async {
                tableView.performBatchUpdates {
                    // step 2
                    // delete sections and rows if necessary
                    //print("2 apply delete")
                    context.coordinator.sections = appliedDeletes
                    for operation in deleteOperations {
                        applyOperation(operation, tableView: tableView)
                    }
                } completion: { _ in
                    tableSemaphore.signal()
                    //print("2 finished delete")
                }
            }
            tableSemaphore.wait()

            DispatchQueue.main.async {
                tableView.performBatchUpdates {
                    // step 3
                    // swap places for rows that moved inside the table
                    // (example of how this happens. send two messages: first m1, then m2. if m2 is delivered to server faster, then it should jump above m1 even though it was sent later)
                    //print("3 apply swaps")
                    context.coordinator.sections = appliedDeletesSwapsAndEdits // NOTE: this array already contains necessary edits, but won't be a problem for appplying swaps
                    for operation in swapOperations {
                        applyOperation(operation, tableView: tableView)
                    }
                } completion: { _ in
                    tableSemaphore.signal()
                    //print("3 finished swaps")
                }
            }
            tableSemaphore.wait()

            DispatchQueue.main.async {
                UIView.setAnimationsEnabled(false)
                tableView.performBatchUpdates {
                    // step 4
                    // check only sections that are already in the table for existing rows that changed and apply only them to table's dataSource without animation
                    //print("4 apply edits")
                    context.coordinator.sections = appliedDeletesSwapsAndEdits

                    for operation in editOperations {
                        applyOperation(operation, tableView: tableView)
                    }

                } completion: { _ in
                    tableSemaphore.signal()
                    UIView.setAnimationsEnabled(true)
                    //print("4 finished edits")
                }
            }
            tableSemaphore.wait()

            if isScrolledToBottom || isScrolledToTop {
                DispatchQueue.main.sync {
                    // step 5
                    // apply the rest of the changes to table's dataSource, i.e. inserts
                    //print("5 apply inserts")
                    context.coordinator.sections = sections

                    tableView.beginUpdates()
                    for operation in insertOperations {
                        applyOperation(operation, tableView: tableView)
                    }
                    tableView.endUpdates()

                    if !isScrollEnabled {
                        tableContentHeight = tableView.contentSize.height
                    }

                    updateSemaphore.signal()
                }
            } else {
                updateSemaphore.signal()
            }
        }
    }

    // MARK: - Operations

    enum Operation {
        case deleteSection(Int)
        case insertSection(Int)

        case delete(Int, Int) // delete with animation
        case insert(Int, Int) // insert with animation
        case swap(Int, Int, Int) // delete first with animation, then insert it into new position with animation. do not do anything with the second for now
        case edit(Int, Int) // reload the element without animation

        var description: String {
            switch self {
            case .deleteSection(let int):
                return "deleteSection \(int)"
            case .insertSection(let int):
                return "insertSection \(int)"
            case .delete(let int, let int2):
                return "delete section \(int) row \(int2)"
            case .insert(let int, let int2):
                return "insert section \(int) row \(int2)"
            case .swap(let int, let int2, let int3):
                return "swap section \(int) rowFrom \(int2) rowTo \(int3)"
            case .edit(let int, let int2):
                return "edit section \(int) row \(int2)"
            }
        }
    }

    func applyOperation(_ operation: Operation, tableView: UITableView) {
        let animation: UITableView.RowAnimation = .top
        switch operation {
        case .deleteSection(let section):
            tableView.deleteSections([section], with: animation)
        case .insertSection(let section):
            tableView.insertSections([section], with: animation)

        case .delete(let section, let row):
            tableView.deleteRows(at: [IndexPath(row: row, section: section)], with: animation)
        case .insert(let section, let row):
            tableView.insertRows(at: [IndexPath(row: row, section: section)], with: animation)
        case .edit(let section, let row):
            tableView.reconfigureRows(at: [IndexPath(row: row, section: section)])
        case .swap(let section, let rowFrom, let rowTo):
            tableView.deleteRows(at: [IndexPath(row: rowFrom, section: section)], with: animation)
            tableView.insertRows(at: [IndexPath(row: rowTo, section: section)], with: animation)
        }
    }

    func operationsSplit(oldSections: [MessagesSection], newSections: [MessagesSection]) -> ([MessagesSection], [MessagesSection], [Operation], [Operation], [Operation], [Operation]) {
        var appliedDeletes = oldSections // start with old sections, remove rows that need to be deleted
        var appliedDeletesSwapsAndEdits = newSections // take new sections and remove rows that need to be inserted for now, then we'll get array with all the changes except for inserts
        // appliedDeletesSwapsEditsAndInserts == newSection

        var deleteOperations = [Operation]()
        var swapOperations = [Operation]()
        var editOperations = [Operation]()
        var insertOperations = [Operation]()

        // 1 compare sections

        let oldDates = oldSections.map { $0.date }
        let newDates = newSections.map { $0.date }
        let commonDates = Array(Set(oldDates + newDates)).sorted(by: >)
        for date in commonDates {
            let oldIndex = appliedDeletes.firstIndex(where: { $0.date == date } )
            let newIndex = appliedDeletesSwapsAndEdits.firstIndex(where: { $0.date == date } )
            if oldIndex == nil, let newIndex {
                // operationIndex is not the same as newIndex because appliedDeletesSwapsAndEdits is being changed as we go, but to apply changes to UITableView we should have initial index
                if let operationIndex = newSections.firstIndex(where: { $0.date == date } ) {
                    appliedDeletesSwapsAndEdits.remove(at: newIndex)
                    insertOperations.append(.insertSection(operationIndex))
                }
                continue
            }
            if newIndex == nil, let oldIndex {
                if let operationIndex = oldSections.firstIndex(where: { $0.date == date } ) {
                    appliedDeletes.remove(at: oldIndex)
                    deleteOperations.append(.deleteSection(operationIndex))
                }
                continue
            }
            guard let newIndex, let oldIndex else { continue }

            // 2 compare section rows
            // isolate deletes and inserts, and remove them from row arrays, leaving only rows that are in both arrays: 'duplicates'
            // this will allow to compare relative position changes of rows - swaps

            var oldRows = appliedDeletes[oldIndex].rows
            var newRows = appliedDeletesSwapsAndEdits[newIndex].rows
            let oldRowIDs = oldRows.map { $0.id }
            let newRowIDs = newRows.map { $0.id }
            let rowIDsToDelete = oldRowIDs.filter { !newRowIDs.contains($0) }.reversed()
            let rowIDsToInsert = newRowIDs.filter { !oldRowIDs.contains($0) }
            for rowId in rowIDsToDelete {
                if let index = oldRows.firstIndex(where: { $0.id == rowId }) {
                    oldRows.remove(at: index)
                    deleteOperations.append(.delete(oldIndex, index)) // this row was in old section, should not be in final result
                }
            }
            for rowId in rowIDsToInsert {
                if let index = newRows.firstIndex(where: { $0.id == rowId }) {
                    // this row was not in old section, should add it to final result
                    insertOperations.append(.insert(newIndex, index))
                }
            }

            for rowId in rowIDsToInsert {
                if let index = newRows.firstIndex(where: { $0.id == rowId }) {
                    // remove for now, leaving only 'duplicates'
                    newRows.remove(at: index)
                }
            }

            // 3 isolate swaps and edits

            for i in 0..<oldRows.count {
                let oldRow = oldRows[i]
                let newRow = newRows[i]
                if oldRow.id != newRow.id { // a swap: rows in same position are not actually the same rows
                    if let index = newRows.firstIndex(where: { $0.id == oldRow.id }) {
                        if !swapsContain(swaps: swapOperations, section: oldIndex, index: i) ||
                            !swapsContain(swaps: swapOperations, section: oldIndex, index: index) {
                            swapOperations.append(.swap(oldIndex, i, index))
                        }
                    }
                } else if oldRow != newRow { // same ids om same positions but something changed - reload rows without animation
                    editOperations.append(.edit(oldIndex, i))
                }
            }

            // 4 store row changes in sections

            appliedDeletes[oldIndex].rows = oldRows
            appliedDeletesSwapsAndEdits[newIndex].rows = newRows
        }

        return (appliedDeletes, appliedDeletesSwapsAndEdits, deleteOperations, swapOperations, editOperations, insertOperations)
    }

    func swapsContain(swaps: [Operation], section: Int, index: Int) -> Bool {
        swaps.filter {
            if case let .swap(section, rowFrom, rowTo) = $0 {
                return section == section && (rowFrom == index || rowTo == index)
            }
            return false
        }.count > 0
    }

    // MARK: - Coordinator

    func makeCoordinator() -> Coordinator {
        Coordinator(viewModel: viewModel, inputViewModel: inputViewModel, isScrolledToBottom: $isScrolledToBottom, isScrolledToTop: $isScrolledToTop, messageBuilder: messageBuilder, mainHeaderBuilder: mainHeaderBuilder, headerBuilder: headerBuilder, type: type, showDateHeaders: showDateHeaders, avatarSize: avatarSize, showMessageMenuOnLongPress: showMessageMenuOnLongPress, tapAvatarClosure: tapAvatarClosure, paginationHandler: paginationHandler, messageUseMarkdown: messageUseMarkdown, showMessageTimeView: showMessageTimeView, messageFont: messageFont, sections: sections, ids: ids, mainBackgroundColor: theme.colors.mainBG)
    }

    class Coordinator: NSObject, UITableViewDataSource, UITableViewDelegate {

        @ObservedObject var viewModel: ChatViewModel
        @ObservedObject var inputViewModel: InputViewModel

        @Binding var isScrolledToBottom: Bool
        @Binding var isScrolledToTop: Bool

        let messageBuilder: MessageBuilderClosure?
        let mainHeaderBuilder: (()->AnyView)?
        let headerBuilder: ((Date)->AnyView)?

        let type: ChatType
        let showDateHeaders: Bool
        let avatarSize: CGFloat
        let showMessageMenuOnLongPress: Bool
        let tapAvatarClosure: ChatView.TapAvatarClosure?
        let paginationHandler: PaginationHandler?
        let messageUseMarkdown: Bool
        let showMessageTimeView: Bool
        let messageFont: UIFont
        var sections: [MessagesSection] {
            didSet {
                if let lastSection = sections.last {
                    paginationTargetIndexPath = IndexPath(row: lastSection.rows.count - 1, section: sections.count - 1)
                }
            }
        }
        let ids: [String]
        let mainBackgroundColor: Color

        init(viewModel: ChatViewModel, inputViewModel: InputViewModel, isScrolledToBottom: Binding<Bool>, isScrolledToTop: Binding<Bool>, messageBuilder: MessageBuilderClosure?, mainHeaderBuilder: (()->AnyView)?, headerBuilder: ((Date)->AnyView)?, type: ChatType, showDateHeaders: Bool, avatarSize: CGFloat, showMessageMenuOnLongPress: Bool, tapAvatarClosure: ChatView.TapAvatarClosure?, paginationHandler: PaginationHandler?, messageUseMarkdown: Bool, showMessageTimeView: Bool, messageFont: UIFont, sections: [MessagesSection], ids: [String], mainBackgroundColor: Color, paginationTargetIndexPath: IndexPath? = nil) {
            self.viewModel = viewModel
            self.inputViewModel = inputViewModel
            self._isScrolledToBottom = isScrolledToBottom
            self._isScrolledToTop = isScrolledToTop
            self.messageBuilder = messageBuilder
            self.mainHeaderBuilder = mainHeaderBuilder
            self.headerBuilder = headerBuilder
            self.type = type
            self.showDateHeaders = showDateHeaders
            self.avatarSize = avatarSize
            self.showMessageMenuOnLongPress = showMessageMenuOnLongPress
            self.tapAvatarClosure = tapAvatarClosure
            self.paginationHandler = paginationHandler
            self.messageUseMarkdown = messageUseMarkdown
            self.showMessageTimeView = showMessageTimeView
            self.messageFont = messageFont
            self.sections = sections
            self.ids = ids
            self.mainBackgroundColor = mainBackgroundColor
            self.paginationTargetIndexPath = paginationTargetIndexPath
        }

        /// call pagination handler when this row is reached
        /// without this there is a bug: during new cells insertion willDisplay is called one extra time for the cell which used to be the last one while it is being updated (its position in group is changed from first to middle)
        var paginationTargetIndexPath: IndexPath?

        func numberOfSections(in tableView: UITableView) -> Int {
            sections.count
        }

        func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
            sections[section].rows.count
        }

        func tableView(_ tableView: UITableView, viewForHeaderInSection section: Int) -> UIView? {
            if type == .comments {
                return sectionHeaderView(section)
            }
            return nil
        }

        func tableView(_ tableView: UITableView, viewForFooterInSection section: Int) -> UIView? {
            if type == .conversation {
                return sectionHeaderView(section)
            }
            return nil
        }

        func tableView(_ tableView: UITableView, heightForHeaderInSection section: Int) -> CGFloat {
            if !showDateHeaders && (section != 0 || mainHeaderBuilder == nil) {
                return 0.1
            }
            return type == .conversation ? 0.1 : UITableView.automaticDimension
        }

        func tableView(_ tableView: UITableView, heightForFooterInSection section: Int) -> CGFloat {
            if !showDateHeaders && (section != 0 || mainHeaderBuilder == nil) {
                return 0.1
            }
            return type == .conversation ? UITableView.automaticDimension : 0.1
        }

        func sectionHeaderView(_ section: Int) -> UIView? {
            if !showDateHeaders && (section != 0 || mainHeaderBuilder == nil) {
                return nil
            }

            let header = UIHostingController(rootView:
                sectionHeaderViewBuilder(section)
                    .rotationEffect(Angle(degrees: (type == .conversation ? 180 : 0)))
            ).view
            header?.backgroundColor = UIColor(mainBackgroundColor)
            return header
        }

        @ViewBuilder
        func sectionHeaderViewBuilder(_ section: Int) -> some View {
            if let mainHeaderBuilder, section == 0 {
                VStack(spacing: 0) {
                    mainHeaderBuilder()
                    dateViewBuilder(section)
                }
            } else {
                dateViewBuilder(section)
            }
        }

        @ViewBuilder
        func dateViewBuilder(_ section: Int) -> some View {
            if showDateHeaders {
                if let headerBuilder {
                    headerBuilder(sections[section].date)
                } else {
                    Text(sections[section].formattedDate)
                        .font(.system(size: 11))
                        .padding(10)
                        .padding(.bottom, 8)
                        .foregroundColor(.gray)
                }
            }
        }

        func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {

            let tableViewCell = tableView.dequeueReusableCell(withIdentifier: "Cell", for: indexPath)
            tableViewCell.selectionStyle = .none
            tableViewCell.backgroundColor = UIColor(mainBackgroundColor)

            let row = sections[indexPath.section].rows[indexPath.row]
            tableViewCell.contentConfiguration = UIHostingConfiguration {
                ChatMessageView(viewModel: viewModel, messageBuilder: messageBuilder, row: row, chatType: type, avatarSize: avatarSize, tapAvatarClosure: tapAvatarClosure, messageUseMarkdown: messageUseMarkdown, isDisplayingMessageMenu: false, showMessageTimeView: showMessageTimeView, messageFont: messageFont)
                    .transition(.scale)
                    .background(MessageMenuPreferenceViewSetter(id: row.id))
                    .rotationEffect(Angle(degrees: (type == .conversation ? 180 : 0)))
                    .onTapGesture { }
                    .applyIf(showMessageMenuOnLongPress) {
                        $0.onLongPressGesture {
                            self.viewModel.messageMenuRow = row
                        }
                    }
            }
            .minSize(width: 0, height: 0)
            .margins(.all, 0)

            return tableViewCell
        }

        func tableView(_ tableView: UITableView, willDisplay cell: UITableViewCell, forRowAt indexPath: IndexPath) {
            guard let paginationHandler = self.paginationHandler, let paginationTargetIndexPath, indexPath == paginationTargetIndexPath else {
                return
            }

            let row = self.sections[indexPath.section].rows[indexPath.row]
            Task.detached {
                await paginationHandler.handleClosure(row.message)
            }
        }

        func scrollViewDidScroll(_ scrollView: UIScrollView) {
            isScrolledToBottom = scrollView.contentOffset.y <= 0
            isScrolledToTop = scrollView.contentOffset.y >= scrollView.contentSize.height - scrollView.frame.height - 1
        }
    }

    func formatRow(_ row: MessageRow) -> String {
        if let status = row.message.status {
            return String("id: \(row.id) text: \(row.message.text) status: \(status) date: \(row.message.createdAt) position: \(row.positionInUserGroup) trigger: \(row.message.triggerRedraw)")
        }
        return ""
    }

    func formatSections(_ sections: [MessagesSection]) -> String {
        var res = "{\n"
        for section in sections.reversed() {
            res += String("\t{\n")
            for row in section.rows {
                res += String("\t\t\(formatRow(row))\n")
            }
            res += String("\t}\n")
        }
        res += String("}")
        return res
    }
}
