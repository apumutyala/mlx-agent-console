import Foundation
import Observation
import UIKit

/// On-device agent loop: model replies → maybe a tool call → Swift validates
/// and runs it → result goes back to the model. Bounded number of tool steps.
@MainActor @Observable
final class LocalAgent {
    struct Turn: Identifiable {
        enum Role { case user, assistant, tool, error }
        let id = UUID()
        let role: Role
        var text: String
        /// Hidden reasoning (`<think>`), shown collapsed under assistant turns.
        var thinking = ""
        /// Tool output that came from another machine (untrusted for memory).
        var remote = false
    }

    private(set) var turns: [Turn] = []
    private(set) var isRunning = false
    var pendingConfirmation: (prompt: String, resume: CheckedContinuation<Bool, Never>)?

    private let backend: any ModelBackend
    private let documents: DocumentStore
    private let toolsProvider: () -> ToolRegistry
    private var conversation: (any ModelConversation)?
    private var tools: ToolRegistry?
    private var task: Task<Void, Never>?
    /// Bumped whenever the model's weights change; a turn started on an older
    /// generation must not keep decoding on its stale KV cache.
    private(set) var modelGeneration = 0
    static let maxToolSteps = 4

    enum AgentError: LocalizedError {
        case modelChanged
        var errorDescription: String? {
            "The model's weights changed during this reply (a compute-node task updated the adapter). Send your message again."
        }
    }

    /// Persistent memory (history, retrieval, reflection). Optional so the
    /// agent still works without it (previews/tests).
    let memory: MemoryHarness?
    /// Identity of the current chat in the conversation store.
    private(set) var conversationID = UUID()
    /// Turns of a resumed chat, replayed into the first prompt of the new session.
    private var resumedTurns: [StoredTurn] = []

    init(backend: any ModelBackend, documents: DocumentStore, memory: MemoryHarness? = nil,
         tools: @escaping () -> ToolRegistry) {
        self.backend = backend
        self.documents = documents
        self.memory = memory
        self.toolsProvider = tools
    }

    /// New chat. The finished one stays in history and is reflected on.
    func reset() {
        task?.cancel()
        // A turn parked on a confirmation alert must not run its tool later.
        answerConfirmation(false)
        persist()
        memory?.conversationEnded(conversationID)
        conversation = nil
        tools = nil
        turns = []
        resumedTurns = []
        conversationID = UUID()
    }

    /// Reopens a saved chat. The model gets a fresh session primed with the
    /// tail of the saved transcript.
    func resume(_ id: UUID) {
        guard id != conversationID, let stored = memory?.storedTurns(id) else { return }
        reset()
        conversationID = id
        resumedTurns = stored
        turns = stored.map { t in
            let role: Turn.Role = switch t.role {
            case .user: .user
            case .assistant: .assistant
            case .tool: .tool
            case .error: .error
            }
            return Turn(role: role, text: t.text, thinking: t.thinking, remote: t.remote)
        }
    }

    private func persist() {
        guard let memory else { return }
        let stored = turns.map { t -> StoredTurn in
            let role: StoredTurn.Role = switch t.role {
            case .user: .user
            case .assistant: .assistant
            case .tool: .tool
            case .error: .error
            }
            return StoredTurn(role: role, text: t.text, thinking: t.thinking, remote: t.remote)
        }
        memory.persist(conversationID: conversationID, turns: stored)
    }

    func stop() { task?.cancel() }

    /// The model's weights changed (e.g. a LoRA adapter was trained or loaded),
    /// so the conversation's KV cache no longer matches. The next message starts
    /// a fresh conversation; the transcript stays on screen.
    func invalidateConversation() {
        conversation = nil
        modelGeneration += 1
    }

    /// Model work holds the shared GPU gate so fleet tasks can't interleave.
    /// Tool calls run outside it (the summarize tool takes the gate itself).
    /// `draining` is the conversation the body streams from: on every exit
    /// (including Stop and errors) we wait for its generation to really stop
    /// before handing the model to the next holder.
    private func withGPU<T>(draining conversation: (any ModelConversation)? = nil,
                            _ body: () async throws -> T) async throws -> T {
        try await GPUGate.shared.acquire("chat")
        do {
            try Task.checkCancellation()
            let value = try await body()
            await conversation?.synchronize()
            await GPUGate.shared.release()
            return value
        } catch {
            await conversation?.synchronize()
            await GPUGate.shared.release()
            throw error
        }
    }

    func send(_ text: String) {
        guard !isRunning, !text.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        turns.append(Turn(role: .user, text: text))
        isRunning = true
        let chat = conversationID
        task = Task { [weak self] in
            await self?.run(userMessage: text)
            self?.isRunning = false
            // Skip if the user already switched chats (reset() saved this one).
            if self?.conversationID == chat { self?.persist() }
        }
    }

    func answerConfirmation(_ ok: Bool) {
        pendingConfirmation?.resume.resume(returning: ok)
        pendingConfirmation = nil
    }

    private func instructions(tools: ToolRegistry, firstMessage: String) async -> String {
        let files = await documents.all().map { "- \($0.id): \($0.name)" }.joined(separator: "\n")
        let device = UIDevice.current.model   // "iPad" / "iPhone"
        let base = """
        You are Agent Console's on-device model (\(backend.displayName)), running locally on the user's \(device) via MLX. \
        Nothing leaves the device unless a tool sends it, and worker jobs always need the user's approval.
        Be concise: answer in a few sentences or a short list, no preamble.
        You have these tools; call one only when the request needs it, otherwise just answer:
        \(tools.descriptions.map { "- " + $0 }.joined(separator: "\n"))
        Imported files (id: name):
        \(files.isEmpty ? "(none)" : files)
        """
        guard let memory else { return base }
        let context = await memory.systemContext(firstMessage: firstMessage, conversationID: conversationID,
                                                 resumed: resumedTurns)
        return base + "\n\n" + context
    }

    private enum Next {
        case user(String)
        case toolResult(ToolResultMessage)
    }

    /// Appends only while `chat` is still the open conversation, so a turn
    /// cancelled by New chat / delete / resume can't write into the next one.
    private func append(to chat: UUID, _ turn: Turn) {
        guard conversationID == chat else { return }
        turns.append(turn)
    }

    private func ensureCurrent(_ chat: UUID) throws {
        guard conversationID == chat else { throw CancellationError() }
    }

    private func run(userMessage: String) async {
        let chat = conversationID
        do {
            // Reflection/embedding must not share the GPU with this turn
            // (cancelled before we queue for the gate, so no gate deadlock).
            await memory?.preemptBackgroundWork()
            // Model load and conversation setup touch the GPU: take the shared gate so a
            // fleet task or batch summary can't run at the same time.
            let (conversation, tools, prefix, generation): (any ModelConversation, ToolRegistry, String, Int) = try await withGPU {
                if !(await backend.isLoaded) {
                    try await backend.load { _ in }
                }
                if let existing = self.conversation, let registry = self.tools {
                    // Later turns: memories newly relevant to this message ride along with it.
                    let note = await memory?.turnContext(userMessage, conversationID: conversationID)
                    return (existing, registry, note ?? "", self.modelGeneration)
                }
                let registry = toolsProvider()
                let fresh = try await backend.makeConversation(
                    instructions: await instructions(tools: registry, firstMessage: userMessage),
                    tools: registry.specifications)
                self.tools = registry
                self.conversation = fresh
                return (fresh, registry, "", self.modelGeneration)
            }

            var next = Next.user(prefix + userMessage)
            for _ in 0...Self.maxToolSteps {
                try Task.checkCancellation()
                try ensureCurrent(chat)
                turns.append(Turn(role: .assistant, text: ""))
                let idx = turns.count - 1
                var raw = ""
                var reasoning = ""
                var nativeCall: ToolCallProposal?
                var rejection: String?
                let request = next
                // Hold the GPU gate only while tokens stream (tool calls run outside it,
                // so a tool that itself needs the model can't deadlock). The stream is
                // created inside the gate, since creating it starts generation.
                try await withGPU(draining: conversation) {
                // A fleet task may have trained or swapped the adapter while a tool ran.
                guard self.modelGeneration == generation else { throw AgentError.modelChanged }
                let stream: AsyncThrowingStream<ModelEvent, Error>
                switch request {
                case .user(let text): stream = conversation.send(text)
                case .toolResult(let result): stream = conversation.sendToolResults([result])
                }
                for try await event in stream {
                    try Task.checkCancellation()
                    try ensureCurrent(chat)
                    switch event {
                    case .text(let chunk): raw += chunk
                    case .reasoning(let chunk): reasoning += chunk
                    case .toolCall(let call): if nativeCall == nil { nativeCall = call }
                    case .rejectedToolCall(let reason): rejection = reason
                    }
                    // Fallback: hide any <think> the library didn't separate.
                    let split = ThinkSplitter.split(raw)
                    turns[idx].text = split.answer
                    turns[idx].thinking = [reasoning, split.thinking]
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }.joined(separator: "\n")
                }
                }
                // Cancelling the consumer just ends the stream (no error), so a Stop
                // would otherwise look like a finished reply (and be read aloud).
                try Task.checkCancellation()

                try ensureCurrent(chat)
                if let rejection, nativeCall == nil {
                    // The library rolls a rejected turn (and its prompt) out of the
                    // transcript, so restate the request along with the problem.
                    turns[idx].text = turns[idx].text.isEmpty ? "→ (invalid tool call)" : turns[idx].text
                    append(to: chat, Turn(role: .tool, text: "ERROR: tool call rejected (\(rejection))"))
                    let original: String
                    switch next {
                    case .user(let text): original = "The user's request was: \(text)"
                    case .toolResult(let r): original = "The last tool result was:\n\(r.content)"
                    }
                    next = .user("""
                    Your tool call could not be used (\(rejection)). Call a listed tool with valid \
                    arguments, or answer directly. \(original)
                    """)
                    continue
                }

                let answer = ThinkSplitter.split(raw).answer
                guard let call = nativeCall ?? ToolCallParser.firstCall(in: answer) else { return }

                turns[idx].text = "→ \(call.name)"
                let result: String
                do {
                    result = try await tools.execute(call) { [weak self] prompt in
                        await self?.confirm(prompt) ?? false
                    }
                } catch {
                    // Errors go back to the model so it can correct itself.
                    result = "ERROR: \(error.localizedDescription)"
                }
                try ensureCurrent(chat)
                append(to: chat, Turn(role: .tool, text: result, remote: MemoryHarness.isRemoteTool(call.name)))
                next = .toolResult(ToolResultMessage(callID: call.id, name: call.name, content: result))
            }
            append(to: chat, Turn(role: .error, text: "Stopped after \(Self.maxToolSteps) tool steps."))
        } catch is CancellationError {
            append(to: chat, Turn(role: .error, text: "Stopped."))
        } catch {
            append(to: chat, Turn(role: .error, text: error.localizedDescription))
        }
    }

    private func confirm(_ prompt: String) async -> Bool {
        await withCheckedContinuation { cont in
            pendingConfirmation = (prompt, cont)
        }
    }
}
