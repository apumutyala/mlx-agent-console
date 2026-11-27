import Foundation

/// Everything the app needs from a language model. The rest of the app never
/// imports MLX directly, so the backend can be swapped (another MLX model, a
/// stub for previews/tests, or a remote model) without touching the harness.
protocol ModelBackend: AnyObject, Sendable {
    var displayName: String { get }
    var isLoaded: Bool { get async }

    /// Download (first run) and load weights. `progress` is 0...1.
    func load(progress: @escaping @Sendable (Double) -> Void) async throws

    /// Start a fresh conversation. Each conversation owns its own KV cache.
    /// `tools` are declared to the model natively (rendered by its chat template
    /// and parsed back into structured calls by the inference library).
    func makeConversation(instructions: String, tools: [ToolSpecification]) async throws -> any ModelConversation

    /// Persist `entry` as the selected model and unload the current one if it differs.
    func switchModel(to entry: ModelCatalog.Entry) async
}

extension ModelBackend {
    func makeConversation(instructions: String) async throws -> any ModelConversation {
        try await makeConversation(instructions: instructions, tools: [])
    }
    func switchModel(to entry: ModelCatalog.Entry) async {}
}

/// A tool as declared to the model: an OpenAI-style function schema.
struct ToolSpecification: Sendable {
    let name: String
    let description: String
    /// JSON-Schema object for the arguments (`type`, `properties`, `required`).
    /// Only JSON-compatible values (String, Bool, numbers, arrays, dictionaries).
    let parameters: [String: any Sendable]
}

/// One streamed item of a model reply.
enum ModelEvent: Sendable {
    /// Answer text.
    case text(String)
    /// Private reasoning (`<think>` content), already separated from the answer.
    case reasoning(String)
    /// A structured tool call parsed by the inference library.
    case toolCall(ToolCallProposal)
    /// Tool-call-shaped output the library refused to parse/authorize.
    case rejectedToolCall(reason: String)
}

struct ToolResultMessage: Sendable {
    let callID: String?
    let name: String
    let content: String
}

protocol ModelConversation: AnyObject, Sendable {
    /// Stream the reply to one user message.
    func send(_ message: String) -> AsyncThrowingStream<ModelEvent, Error>
    /// Answer the previous reply's tool call(s) with tool-role messages and stream the follow-up.
    func sendToolResults(_ results: [ToolResultMessage]) -> AsyncThrowingStream<ModelEvent, Error>
}

extension ModelConversation {
    /// Answer text only (reasoning dropped), for non-chat uses like summaries.
    func complete(_ message: String) async throws -> String {
        var out = ""
        for try await event in send(message) {
            try Task.checkCancellation()
            if case .text(let chunk) = event { out += chunk }
        }
        return ThinkSplitter.split(out).answer.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Fallback for models whose reasoning the library didn't separate: splits
/// `<think>…</think>` out of raw text. Works on partial (streaming) text — an
/// unclosed `<think>` counts as thinking so far.
enum ThinkSplitter {
    static func split(_ text: String) -> (thinking: String, answer: String) {
        guard text.contains("<think>") || text.contains("</think>") else { return ("", text) }
        var thinking = ""
        var answer = ""
        var rest = Substring(text)
        // A reply can start mid-thought when the template pre-fills `<think>`.
        if let close = rest.range(of: "</think>"),
           rest.range(of: "<think>").map({ $0.lowerBound > close.lowerBound }) ?? true {
            thinking += rest[..<close.lowerBound]
            rest = rest[close.upperBound...]
        }
        while let open = rest.range(of: "<think>") {
            answer += rest[..<open.lowerBound]
            let inner = rest[open.upperBound...]
            if let close = inner.range(of: "</think>") {
                thinking += inner[..<close.lowerBound]
                rest = inner[close.upperBound...]
            } else {
                thinking += inner
                rest = ""
            }
        }
        answer += rest
        return (thinking.trimmingCharacters(in: .whitespacesAndNewlines),
                answer.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

/// Deterministic backend for SwiftUI previews and logic tests on the simulator.
final class EchoBackend: ModelBackend, ModelConversation, @unchecked Sendable {
    let displayName = "Echo (no model)"
    var isLoaded: Bool { get async { true } }
    func load(progress: @escaping @Sendable (Double) -> Void) async throws { progress(1) }
    func makeConversation(instructions: String, tools: [ToolSpecification]) async throws -> any ModelConversation { self }
    func send(_ message: String) -> AsyncThrowingStream<ModelEvent, Error> {
        AsyncThrowingStream { c in
            c.yield(.text("Echo: " + message.prefix(200)))
            c.finish()
        }
    }
    func sendToolResults(_ results: [ToolResultMessage]) -> AsyncThrowingStream<ModelEvent, Error> {
        send(results.map(\.content).joined(separator: "\n"))
    }
}
