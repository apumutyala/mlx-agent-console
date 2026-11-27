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
    static let maxToolSteps = 4

    init(backend: any ModelBackend, documents: DocumentStore, tools: @escaping () -> ToolRegistry) {
        self.backend = backend
        self.documents = documents
        self.toolsProvider = tools
    }

    func reset() {
        task?.cancel()
        conversation = nil
        tools = nil
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
        let device = UIDevice.current.model   // "iPad" / "iPhone"
        return """
        You are Agent Console's on-device model (\(backend.displayName)), running locally on the user's \(device) via MLX. \
        Nothing leaves the device unless a tool sends it, and worker jobs always need the user's approval.
        Be concise: answer in a few sentences or a short list, no preamble.
        You have these tools; call one only when the request needs it, otherwise just answer:
        \(tools.descriptions.map { "- " + $0 }.joined(separator: "\n"))
        Imported files (id: name):
        \(files.isEmpty ? "(none)" : files)
        """
    }

    private enum Next {
        case user(String)
        case toolResult(ToolResultMessage)
    }

    private func run(userMessage: String) async {
        do {
            // Model load and conversation setup touch the GPU: take the shared gate so a
            // fleet task or batch summary can't run at the same time.
            let (conversation, tools): (any ModelConversation, ToolRegistry) = try await withGPU {
                if !(await backend.isLoaded) {
                    try await backend.load { _ in }
                }
                if let existing = self.conversation, let registry = self.tools { return (existing, registry) }
                let registry = toolsProvider()
                let fresh = try await backend.makeConversation(
                    instructions: await instructions(tools: registry), tools: registry.specifications)
                self.tools = registry
                self.conversation = fresh
                return (fresh, registry)
            }

            var next = Next.user(userMessage)
            for _ in 0...Self.maxToolSteps {
                try Task.checkCancellation()
                turns.append(Turn(role: .assistant, text: ""))
                let idx = turns.count - 1
                var raw = ""
                var reasoning = ""
                var nativeCall: ToolCallProposal?
                var rejection: String?
                let stream: AsyncThrowingStream<ModelEvent, Error>
                switch next {
                case .user(let text): stream = conversation.send(text)
                case .toolResult(let result): stream = conversation.sendToolResults([result])
                }
                // Hold the GPU gate only while tokens stream (tool calls run outside it,
                // so a tool that itself needs the model can't deadlock).
                try await withGPU {
                for try await event in stream {
                    try Task.checkCancellation()
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

                if let rejection, nativeCall == nil {
                    // The library rolls a rejected turn (and its prompt) out of the
                    // transcript, so restate the request along with the problem.
                    turns[idx].text = turns[idx].text.isEmpty ? "→ (invalid tool call)" : turns[idx].text
                    turns.append(Turn(role: .tool, text: "ERROR: tool call rejected (\(rejection))"))
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
                turns.append(Turn(role: .tool, text: result))
                next = .toolResult(ToolResultMessage(callID: call.id, name: call.name, content: result))
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
