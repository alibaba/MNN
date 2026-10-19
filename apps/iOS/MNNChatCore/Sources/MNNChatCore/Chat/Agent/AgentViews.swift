import SwiftUI

/// Agent mode switch shown next to the model picker. The mode belongs to the
/// conversation, so it can only change before the first message.
struct AgentModeChip: View {
    @ObservedObject var viewModel: LLMChatViewModel
    @State private var showsInfo = false

    var body: some View {
        let isOn = viewModel.isAgentMode
        Button {
            if viewModel.canChangeAgentMode {
                viewModel.setAgentMode(!isOn)
            } else {
                showsInfo = true
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isOn ? "globe.asia.australia.fill" : "globe.asia.australia")
                Text(AppText.text("agent.mode"))
            }
            .font(.subheadline.weight(.medium))
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(isOn ? Color.customBlue.opacity(0.15) : Color(uiColor: .secondarySystemBackground), in: Capsule())
            .foregroundStyle(isOn ? Color.customBlue : .secondary)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(AppText.text("agent.mode"))
        .accessibilityValue(AppText.text(isOn ? "agent.on" : "agent.off"))
        .accessibilityIdentifier("chat.agentMode")
        .alert(AppText.text(isOn ? "agent.locked.onTitle" : "agent.locked.offTitle"), isPresented: $showsInfo) {
            Button(AppText.text("OK"), role: .cancel) {}
        } message: {
            Text(AppText.text("agent.locked.message"))
        }
    }
}

/// Lists and clears what Agent mode has remembered.
struct AgentMemoryView: View {
    @State private var memories: [AgentMemory] = []
    @State private var skills: [AgentSkill] = []
    @State private var confirmsClear = false

    var body: some View {
        List {
            Section {
                if memories.isEmpty {
                    Text(AppText.text("agent.memory.empty")).foregroundStyle(.secondary)
                }
                ForEach(memories, id: \.content) { memory in
                    Text(memory.content).font(.subheadline)
                }
            } header: {
                Text(AppText.text("agent.memory.section"))
            }
            Section {
                if skills.isEmpty {
                    Text(AppText.text("agent.skills.empty")).foregroundStyle(.secondary)
                }
                ForEach(skills, id: \.name) { skill in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(skill.name).font(.subheadline.weight(.medium))
                        Text(skill.actionTemplate).font(.caption).foregroundStyle(.secondary)
                    }
                }
            } header: {
                Text(AppText.text("agent.skills.section"))
            } footer: {
                Text(AppText.text("agent.memory.footer"))
            }
            Section {
                Button(AppText.text("agent.memory.clear"), role: .destructive) { confirmsClear = true }
                    .disabled(memories.isEmpty && skills.isEmpty)
            }
        }
        .navigationTitle(AppText.text("agent.memory.title"))
        .navigationBarTitleDisplayMode(.inline)
        .onAppear(perform: reload)
        .confirmationDialog(AppText.text("agent.memory.clear"), isPresented: $confirmsClear, titleVisibility: .visible) {
            Button(AppText.text("agent.memory.clear"), role: .destructive) {
                AgentStore.shared?.clearMemories()
                AgentStore.shared?.clearSkills()
                reload()
            }
            Button(AppText.text("Cancel"), role: .cancel) {}
        }
    }

    private func reload() {
        memories = AgentStore.shared?.allMemories() ?? []
        skills = AgentStore.shared?.allSkills() ?? []
    }
}
