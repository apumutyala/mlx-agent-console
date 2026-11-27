import Foundation
import Observation

/// On-device agent loop: model replies → maybe a tool call → Swift validates
/// and runs it → result goes back to the model. Bounded number of tool steps.
@MainActor @Observable
final class LocalAgent {
    struct Turn: Identifiable {
        enum Role { case user, assistant, tool, error }
        let id = UUID()
        let role: Role
        var text: String
    }

    private(set) var turns: [Turn] = []
    private(set) var isRunning = false
    var pendingConfirmation: (prompt: String, resume: CheckedContinuation<Bool, Never>)?

    private let backend: any ModelBackend
    private let documents: DocumentStore
    private let toolsProvider: () -> ToolRegistry
    private var conversation: (any ModelConversation)?
    private var task: Task<Void, Never>?
    static let maxToolSteps = 4

    init(backend: any ModelBackend, documents: DocumentStore, tools: @escaping () -> ToolRegistry) {
        self.backend = backend
        self.documents = documents
        self.toolsProvider = tools
    }

    func reset() {
        task?.cancel()
        conversation = nil
        turns = []
    }

    func stop() { task?.cancel() }

    /// The model's weights changed (e.g. a LoRA adapter was trained or loaded),
    /// so the conversation's KV cache no longer matches. The next message starts
    /// a fresh conversation; the transcript stays on screen.
    func invalidateConversation() {
        conversation = nil
    }

    /// Model work holds the shared GPU gate so fleet tasks can't interleave.
    /// Tool calls run outside it (the summarize tool takes the gate itself).
    private func withGPU<T>(_ body: () async throws -> T) async throws -> T {
        await GPUGate.shared.acquire("chat")
        do {
            let value = try await body()
            await GPUGate.shared.release()
            return value
        } catch {
            await GPUGate.shared.release()
            throw error
        }
    }

    func send(_ text: String) {
        guard !isRunning, !text.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        turns.append(Turn(role: .user, text: text))
        isRunning = true
        task = Task { [weak self] in
            await self?.run(userMessage: text)
            self?.isRunning = false
        }
    }

    func answerConfirmation(_ ok: Bool) {
        pendingConfirmation?.resume.resume(returning: ok)
        pendingConfirmation = nil
    }

    private func instructions(tools: ToolRegistry) async -> String {
        let files = await documents.all().map { "- \($0.id): \($0.name)" }.joined(separator: "\n")
        return """
        You are a helpful assistant running entirely on the user's device.
        You can call tools by replying with exactly one block:
        <tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>
        and nothing else. You will then receive the tool result. Only call a tool when needed.
        Tools:
        \(tools.descriptions.map { "- " + $0 }.joined(separator: "\n"))
        Imported files:
        \(files.isEmpty ? "(none)" : files)
        """
    }

    private func run(userMessage: String) async {
        do {
            let tools = toolsProvider()
            let system = await instructions(tools: tools)
            let conversation: any ModelConversation = try await withGPU {
                if !(await backend.isLoaded) {
                    try await backend.load { _ in }
                }
                if let existing = self.conversation { return existing }
                let fresh = try await backend.makeConversation(instructions: system)
                self.conversation = fresh
                return fresh
            }

            var message = userMessage
            for _ in 0...Self.maxToolSteps {
                try Task.checkCancellation()
                turns.append(Turn(role: .assistant, text: ""))
                let idx = turns.count - 1
                let reply: String = try await withGPU {
                    var reply = ""
                    for try await chunk in conversation.send(message) {
                        try Task.checkCancellation()
                        reply += chunk
                        self.turns[idx].text = reply
                    }
                    return reply
                }
                guard let call = ToolCallParser.firstCall(in: reply) else { return }

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
                turns.append(Turn(role: .tool, text: result))
                message = "<tool_response>\n\(result)\n</tool_response>"
            }
            turns.append(Turn(role: .error, text: "Stopped after \(Self.maxToolSteps) tool steps."))
        } catch is CancellationError {
            turns.append(Turn(role: .error, text: "Stopped."))
        } catch {
            turns.append(Turn(role: .error, text: error.localizedDescription))
        }
    }

    private func confirm(_ prompt: String) async -> Bool {
        await withCheckedContinuation { cont in
            pendingConfirmation = (prompt, cont)
        }
    }
}
