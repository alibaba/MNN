import Foundation
import SQLite

/// A durable fact about the user, written by the agent through `memory_updates`.
struct AgentMemory: Equatable {
    let content: String
    let category: String
    let updatedAt: Date
}

/// A reusable procedure, written by the agent through `skill_updates`.
struct AgentSkill: Equatable {
    let name: String
    let triggers: [String]
    let actionTemplate: String
}

/// Agent state kept outside the chat history database: which conversations
/// run in Agent mode, plus the agent's memory and skills (shared by all
/// Agent conversations, as in the Android implementation).
final class AgentStore {
    static let shared: AgentStore? = try? AgentStore()

    static let maxMemories = 30
    static let maxSkills = 30
    static let maxMemoryLength = 168
    static let maxSkillLength = 350

    private typealias Column<T> = SQLite.Expression<T>

    private let db: Connection
    private let queue = DispatchQueue(label: "mnn.agent-store")
    private let sessions = Table("agentSessions")
    private let sessionId = Column<String>("historyId")
    private let memories = Table("agentMemories")
    private let memoryContent = Column<String>("content")
    private let memoryCategory = Column<String>("category")
    private let memoryUpdated = Column<Date>("updatedAt")
    private let skills = Table("agentSkills")
    private let skillName = Column<String>("name")
    private let skillTriggers = Column<String>("triggers")
    private let skillAction = Column<String>("actionTemplate")
    private let skillUpdated = Column<Date>("updatedAt")

    private init() throws {
        guard let path = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first else {
            throw NSError(domain: "AgentStore", code: -1)
        }
        db = try Connection("\(path)/agent.sqlite3")
        try db.run(sessions.create(ifNotExists: true) { $0.column(sessionId, primaryKey: true) })
        try db.run(memories.create(ifNotExists: true) { t in
            t.column(memoryContent, primaryKey: true)
            t.column(memoryCategory)
            t.column(memoryUpdated)
        })
        try db.run(skills.create(ifNotExists: true) { t in
            t.column(skillName, primaryKey: true)
            t.column(skillTriggers)
            t.column(skillAction)
            t.column(skillUpdated)
        })
    }

    // MARK: Session mode

    func isAgentSession(_ historyId: String) -> Bool {
        queue.sync { ((try? db.scalar(sessions.filter(sessionId == historyId).count)) ?? 0) > 0 }
    }

    func setAgentSession(_ historyId: String, enabled: Bool) {
        queue.sync {
            if enabled {
                _ = try? db.run(sessions.insert(or: .ignore, sessionId <- historyId))
            } else {
                _ = try? db.run(sessions.filter(sessionId == historyId).delete())
            }
        }
    }

    // MARK: Memory

    func allMemories() -> [AgentMemory] {
        queue.sync {
            let rows = (try? db.prepare(memories.order(memoryUpdated.desc).limit(Self.maxMemories))) ?? AnySequence([])
            return rows.map { AgentMemory(content: $0[memoryContent], category: $0[memoryCategory], updatedAt: $0[memoryUpdated]) }
        }
    }

    func addMemory(_ content: String, category: String) {
        let text = String(content.trimmingCharacters(in: .whitespacesAndNewlines).prefix(Self.maxMemoryLength))
        guard !text.isEmpty else { return }
        queue.sync {
            _ = try? db.run(memories.insert(or: .replace, memoryContent <- text,
                                            memoryCategory <- (category.isEmpty ? "fact" : category),
                                            memoryUpdated <- Date()))
            // Keep only the most recent entries.
            let stale = memories.order(memoryUpdated.desc).limit(-1, offset: Self.maxMemories)
            if let rows = try? db.prepare(stale) {
                for row in Array(rows) { _ = try? db.run(memories.filter(memoryContent == row[memoryContent]).delete()) }
            }
        }
    }

    func clearMemories() { queue.sync { _ = try? db.run(memories.delete()) } }

    // MARK: Skills

    func allSkills() -> [AgentSkill] {
        queue.sync {
            let rows = (try? db.prepare(skills.order(skillUpdated.desc).limit(Self.maxSkills))) ?? AnySequence([])
            return rows.map {
                AgentSkill(name: $0[skillName],
                           triggers: $0[skillTriggers].split(separator: "\n").map(String.init),
                           actionTemplate: $0[skillAction])
            }
        }
    }

    func saveSkill(_ skill: AgentSkill) {
        let name = skill.name.trimmingCharacters(in: .whitespacesAndNewlines)
        let action = String(skill.actionTemplate.trimmingCharacters(in: .whitespacesAndNewlines).prefix(Self.maxSkillLength))
        guard !name.isEmpty, !action.isEmpty else { return }
        let triggers = skill.triggers.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        queue.sync {
            _ = try? db.run(skills.insert(or: .replace, skillName <- String(name.prefix(64)),
                                          skillTriggers <- triggers.joined(separator: "\n"),
                                          skillAction <- action, skillUpdated <- Date()))
        }
    }

    func clearSkills() { queue.sync { _ = try? db.run(skills.delete()) } }
}
