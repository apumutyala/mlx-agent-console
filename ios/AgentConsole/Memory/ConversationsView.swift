import SwiftUI

/// Saved chats: resume, rename, delete. Shown as a sheet from the chat screen.
struct ConversationsView: View {
    let agent: LocalAgent
    let store: ConversationStore
    @Environment(\.dismiss) private var dismiss
    @State private var renaming: ConversationSummary?
    @State private var newTitle = ""
    @State private var deleting: ConversationSummary?
    @State private var query = ""

    private var rows: [ConversationSummary] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return store.summaries }
        return store.summaries.filter { $0.title.lowercased().contains(q) || $0.preview.lowercased().contains(q) }
    }

    var body: some View {
        NavigationStack {
            List {
                if store.summaries.isEmpty {
                    ContentUnavailableView("No saved chats", systemImage: "bubble.left.and.bubble.right",
                                           description: Text("Chats are saved automatically as you talk."))
                }
                ForEach(rows) { c in
                    Button {
                        agent.resume(c.id)
                        dismiss()
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text(c.title.isEmpty ? "Chat" : c.title).font(.headline).lineLimit(1)
                                if c.id == agent.conversationID {
                                    Text("current").font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                            Text(c.preview).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                            Text("\(c.updatedAt.formatted(date: .abbreviated, time: .shortened)) · \(c.turnCount) turns")
                                .font(.caption2).foregroundStyle(.tertiary)
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(agent.isRunning)
                    .swipeActions {
                        Button("Delete", systemImage: "trash", role: .destructive) { deleting = c }
                        Button("Rename", systemImage: "pencil") { newTitle = c.title; renaming = c }
                    }
                    .contextMenu {
                        Button("Rename", systemImage: "pencil") { newTitle = c.title; renaming = c }
                        Button("Delete", systemImage: "trash", role: .destructive) { deleting = c }
                    }
                }
            }
            .searchable(text: $query, prompt: "Search chats")
            .navigationTitle("Chats")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .primaryAction) {
                    Button("New chat", systemImage: "square.and.pencil") { agent.reset(); dismiss() }
                        .disabled(agent.isRunning)
                }
            }
            .alert("Rename chat", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
                TextField("Title", text: $newTitle)
                Button("Save") {
                    if let c = renaming { store.rename(c.id, to: newTitle) }
                    renaming = nil
                }
                Button("Cancel", role: .cancel) { renaming = nil }
            }
            .confirmationDialog("Delete this chat?", isPresented: Binding(get: { deleting != nil },
                                                                           set: { if !$0 { deleting = nil } }),
                                titleVisibility: .visible) {
                Button("Delete chat", role: .destructive) {
                    if let c = deleting {
                        if c.id == agent.conversationID { agent.reset() }
                        store.delete(c.id)
                    }
                    deleting = nil
                }
            } message: {
                Text("The transcript is removed. Memories learned from it stay until you delete them in Settings › Memory.")
            }
        }
    }
}

/// One-line hook for ChatView: a "Chats" toolbar button that opens the history.
struct ConversationHistoryButton: ViewModifier {
    let agent: LocalAgent
    @State private var showing = false

    func body(content: Content) -> some View {
        content
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Chats", systemImage: "clock.arrow.circlepath") { showing = true }
                }
            }
            .sheet(isPresented: $showing) {
                if let memory = agent.memory {
                    ConversationsView(agent: agent, store: memory.conversations)
                }
            }
    }
}

extension View {
    func conversationHistory(_ agent: LocalAgent) -> some View { modifier(ConversationHistoryButton(agent: agent)) }
}
