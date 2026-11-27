import SwiftUI

struct ChatView: View {
    @Bindable var agent: LocalAgent
    let backend: any ModelBackend
    @State private var draft = ""
    @State private var loadProgress: Double?
    @State private var loadError: String?
    /// Re-runs the load task when the model is switched in Settings.
    @AppStorage(ModelCatalog.storageKey) private var selectedModelID = ModelCatalog.defaultEntry.id

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    List(agent.turns) { turn in
                        TurnRow(turn: turn).id(turn.id)
                    }
                    .listStyle(.plain)
                    .onChange(of: agent.turns.last?.text) { _, _ in
                        if let id = agent.turns.last?.id { proxy.scrollTo(id, anchor: .bottom) }
                    }
                }
                if let p = loadProgress, p < 1 {
                    ProgressView("Loading \(backend.displayName)…", value: p).padding()
                }
                if let loadError {
                    Label(loadError, systemImage: "memorychip")
                        .font(.footnote).foregroundStyle(.orange).padding(.horizontal)
                }
                SpeechStatusBar()
                HStack {
                    TextField("Ask the on-device model", text: $draft, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...5)
                    SpeechMicButton { text in agent.send(text) }
                        .disabled(agent.isRunning)
                    if agent.isRunning {
                        Button("Stop", systemImage: "stop.circle.fill") { agent.stop() }
                    } else {
                        Button("Send", systemImage: "arrow.up.circle.fill") {
                            agent.send(draft); draft = ""
                        }
                        .disabled(draft.isEmpty)
                    }
                }
                .labelStyle(.iconOnly)
                .font(.title2)
                .padding()
            }
            .navigationTitle("Local agent")
            .speechReadsReplies(agent)
            .toolbar {
                SpeakerToggle()
                Button("New chat", systemImage: "square.and.pencil") { agent.reset() }
            }
            .task(id: selectedModelID) {
                loadError = nil
                guard !(await backend.isLoaded) else { return }
                loadProgress = 0
                do {
                    try await backend.load { p in Task { @MainActor in loadProgress = p } }
                } catch is CancellationError {
                    // superseded by another model switch
                } catch {
                    if !Task.isCancelled { loadError = error.localizedDescription }
                }
                loadProgress = 1
            }
            .alert("Allow this action?",
                   isPresented: Binding(get: { agent.pendingConfirmation != nil }, set: { _ in })) {
                Button("Approve") { agent.answerConfirmation(true) }
                Button("Deny", role: .cancel) { agent.answerConfirmation(false) }
            } message: {
                Text(agent.pendingConfirmation?.prompt ?? "")
            }
        }
    }
}

private struct TurnRow: View {
    let turn: LocalAgent.Turn
    var body: some View {
        switch turn.role {
        case .user: Text(turn.text).bold()
        case .assistant:
            VStack(alignment: .leading, spacing: 4) {
                if !turn.thinking.isEmpty {
                    DisclosureGroup("Thinking") {
                        Text(turn.thinking).font(.caption).foregroundStyle(.secondary)
                    }
                    .font(.caption)
                }
                Text(turn.text.isEmpty ? "…" : turn.text)
            }
        case .tool: Text(turn.text).font(.caption.monospaced()).foregroundStyle(.secondary)
        case .error: Label(turn.text, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
        }
    }
}
