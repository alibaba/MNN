import SwiftUI

struct TaskModifier<T: Equatable>: ViewModifier {
  private let id: T
  private let priority: TaskPriority
  private let action: @Sendable () async -> Void

  @State private var task: Task<Void, Never>?

  init(
    id: T,
    priority: TaskPriority = .userInitiated,
    action: @escaping @Sendable () async -> Void
  ) {
    self.id = id
    self.priority = priority
    self.action = action
  }

  func body(content: Content) -> some View {
    content
      .onAppear {
        self.task = Task(priority: priority, operation: action)
      }
      .onDisappear {
        self.task?.cancel()
        self.task = nil
      }
      .onChange(of: self.id) { _ in
        self.task?.cancel()
        self.task = Task(priority: priority, operation: action)
      }
  }
}
