import Foundation

/// The fixed, versioned workload set. Changing a prompt, length or the text
/// generator means bumping `setVersion` (and the workload's `version`), so
/// receipts with the same ids stay comparable.
///
/// Prompts are fed as raw token ids (no chat template) and cut to an exact
/// token count per model, so "1K" means 1024 prompt tokens for every model.
enum BenchWorkloads {
    /// w2: an untimed warm-up decode at each workload's own prompt length (w1 warmed up
    /// only the short prompt, so rep 1 of 1K/4K included Metal kernel compilation) and a
    /// 10 s idle settle before the baseline. Prompts and lengths are unchanged from w1.
    static let setVersion = "w2"

    struct Workload: Identifiable, Hashable, Sendable {
        enum Prompt: Hashable, Sendable {
            /// A short instruction, tokenized as is.
            case short(String)
            /// The bundled document cut to exactly `tokens` tokens, ending with `instruction`.
            case filled(tokens: Int, instruction: String)
        }
        let id: String
        let version: Int
        let description: String
        let prompt: Prompt
        let maxTokens: Int
    }

    static let short = Workload(
        id: "short-128", version: 2, description: "short prompt → 128 tokens",
        prompt: .short("Explain step by step how a bicycle's gears, chain and brakes work together, "
                       + "then list three common maintenance mistakes."),
        maxTokens: 128)
    static let oneK = Workload(
        id: "1k-256", version: 2, description: "1024-token prompt → 256 tokens",
        prompt: .filled(tokens: 1024, instruction: "\n\nSummarize the field notes above in detail.\n"),
        maxTokens: 256)
    static let fourK = Workload(
        id: "4k-256", version: 2, description: "4096-token prompt → 256 tokens",
        prompt: .filled(tokens: 4096, instruction: "\n\nSummarize the field notes above in detail.\n"),
        maxTokens: 256)

    static let all: [Workload] = [short, oneK, fourK]

    static func workloads(ids: [String]?) throws -> [Workload] {
        guard let ids, !ids.isEmpty else { return all }
        return try ids.map { id in
            guard let w = all.first(where: { $0.id == id }) else {
                throw BenchError.unknownWorkload(id, known: all.map(\.id))
            }
            return w
        }
    }

    /// Token ids for a workload: `encode` is the model's tokenizer
    /// (`addSpecialTokens` adds BOS where the tokenizer defines one).
    static func promptTokens(_ w: Workload, encode: (String, Bool) -> [Int]) -> [Int] {
        switch w.prompt {
        case .short(let text):
            return encode(text, true)
        case .filled(let target, let instruction):
            let tail = encode(instruction, false)
            var body = encode(document, true)
            while body.count < target {   // never for w1/w2 (~7K tokens), but stay exact if a tokenizer packs densely
                body += encode(" " + document, false)
            }
            return Array(body.prefix(target - tail.count)) + tail
        }
    }

    /// ~30 KB of deterministic synthetic prose ("field notes" from a fictional
    /// survey), generated from a fixed seed with integer-only arithmetic so it is
    /// byte-identical on every device and OS. Bundled as code: no resource files.
    static let document: String = {
        var rng = SplitMix64(seed: 0x5EED_BE4C_0001)
        let sites = ["the north ridge", "the river delta", "the old quarry", "the salt marsh", "the cedar valley",
                     "the eastern terraces", "the harbor flats", "the upper meadow", "the lava field", "the pine basin"]
        let observers = ["The survey team", "Our second crew", "The night shift", "A visiting hydrologist",
                         "The lead botanist", "Two volunteers", "The drone operator", "The soil chemist"]
        let verbs = ["recorded", "measured", "photographed", "sampled", "mapped", "counted", "logged", "tagged"]
        let things = ["fourteen nesting sites", "a sharp drop in water clarity", "unusual lichen growth",
                      "three new springs", "erosion along the access trail", "a herd of roughly forty deer",
                      "high nitrate readings", "fresh rockfall below the cliff", "flowering orchids",
                      "damaged fence posts", "a stable beaver dam", "migrating warblers", "silt on the gravel bars"]
        let times = ["at dawn", "shortly after noon", "during light rain", "under a clear sky", "before the tide turned",
                     "in strong wind", "late in the evening", "after two dry weeks"]
        let follow = ["The readings match last season within the usual margin.",
                      "This contradicts the forecast in the spring report.",
                      "We flagged the location for a return visit next month.",
                      "Equipment calibration was checked before and after.",
                      "Samples were sealed, labeled and sent to the lab.",
                      "The change is most visible on the south-facing slopes.",
                      "Local residents confirmed the pattern from memory.",
                      "No disturbance from the nearby road was detected."]
        func pick(_ xs: [String]) -> String { xs[Int(rng.next() % UInt64(xs.count))] }
        var out = ""
        var day = 1
        while out.utf8.count < 30_000 {
            out += "Day \(day), \(pick(sites)). "
            for _ in 0..<(3 + Int(rng.next() % 3)) {
                out += "\(pick(observers)) \(pick(verbs)) \(pick(things)) \(pick(times)). \(pick(follow)) "
            }
            out += "Station \(100 + Int(rng.next() % 900)) logged \(Int(rng.next() % 40)) mm of rain "
                + "and a mean temperature of \(Int(rng.next() % 25)) degrees.\n"
            day += 1
        }
        return out
    }()
}

/// Small, well-known PRNG (public-domain SplitMix64). Stable across platforms.
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

enum BenchError: LocalizedError {
    case unknownWorkload(String, known: [String])
    case unknownModel(String, known: [String])
    case busy
    var errorDescription: String? {
        switch self {
        case .unknownWorkload(let id, let known): "Unknown workload \(id); known: \(known.joined(separator: ", "))"
        case .unknownModel(let id, let known): "Unknown model \(id); known: \(known.joined(separator: ", "))"
        case .busy: "A benchmark is already running."
        }
    }
}
