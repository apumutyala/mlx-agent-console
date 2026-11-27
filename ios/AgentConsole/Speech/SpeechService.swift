import AVFoundation
import Foundation
import Observation
import Xybrid

/// On-device speech for the chat: push-to-talk STT (Whisper via Xybrid /
/// whisper.cpp) and read-aloud TTS (Kokoro via Xybrid / ONNX Runtime).
/// The LLM itself stays on MLX; this file only turns audio <-> text.
///
/// Models come from the Xybrid registry and are downloaded into the SDK cache
/// on first use (network needed once), then load offline.
@MainActor @Observable
final class SpeechService {
    static let shared = SpeechService()

    static let asrModelId = "whisper-tiny-ggml"   // 16 kHz mono in, text out
    static let ttsModelId = "kokoro-82m"          // text in, 24 kHz PCM16 out
    static let ttsVoice = "af"

    enum Phase: Equatable {
        case idle
        case downloading(model: String, progress: Double)  // first use: needs network
        case loading(model: String)
        case recording
        case transcribing
        case synthesizing
        case speaking
    }

    private(set) var phase: Phase = .idle
    private(set) var lastError: String?
    /// Read assistant replies aloud. Persisted.
    var speakReplies: Bool = UserDefaults.standard.bool(forKey: "speech.speakReplies") {
        didSet {
            UserDefaults.standard.set(speakReplies, forKey: "speech.speakReplies")
            if !speakReplies { stopSpeaking() }
        }
    }

    var isRecording: Bool { phase == .recording }
    var isBusy: Bool {
        switch phase {
        case .idle, .speaking: return false
        default: return true
        }
    }

    private var asr: XybridModel?
    private var tts: XybridModel?
    private let recorder = MicRecorder()
    private var speakTask: Task<Void, Never>?
    private var player: AVAudioPlayer?
    private let playerDelegate = PlayerDelegate()

    private init() {}

    // MARK: - Model loading

    /// Whether a model still needs a download (and therefore network) before it can load.
    func needsDownload(_ id: String) -> Bool {
        !((try? Xybrid.hasCachedModelData(id)) ?? false)
    }

    private func loadModel(_ id: String) async throws -> XybridModel {
        Xybrid.initialize()   // no apiKey: fully on-device; idempotent
        let loader = Xybrid.model(id)
        if needsDownload(id), let download = loader.download() {
            phase = .downloading(model: id, progress: 0)
            for await status in download.progress() {
                phase = .downloading(model: id, progress: Double(status.progress))
                if status.state == .failed {
                    throw SpeechError.downloadFailed(id)
                }
                if status.state != .downloading { break }
            }
        }
        phase = .loading(model: id)
        return try await loader.load()
    }

    private func asrModel() async throws -> XybridModel {
        if let asr { return asr }
        let m = try await loadModel(Self.asrModelId)
        asr = m
        return m
    }

    private func ttsModel() async throws -> XybridModel {
        if let tts { return tts }
        let m = try await loadModel(Self.ttsModelId)
        tts = m
        return m
    }

    // MARK: - Push-to-talk STT

    /// Starts capturing the microphone. Any speech playback stops first.
    func startRecording() async {
        stopSpeaking()
        lastError = nil
        guard await AVAudioApplication.requestRecordPermission() else {
            lastError = "Microphone access is off. Enable it in Settings > Privacy > Microphone."
            return
        }
        do {
            try Self.configureSession()
            try recorder.start()
            phase = .recording
        } catch {
            lastError = "Couldn't start recording: \(error.localizedDescription)"
            phase = .idle
        }
    }

    /// Stops capturing and returns the transcript ("" if nothing was heard).
    func stopRecordingAndTranscribe() async -> String {
        let pcm = recorder.stop()
        guard !pcm.isEmpty else { phase = .idle; return "" }
        do {
            let model = try await asrModel()
            phase = .transcribing
            let wav = WAV.encode(pcm16: pcm, sampleRate: MicRecorder.sampleRate)
            let result = try await model.runAsync(
                envelope: XybridEnvelope.audio(pcmData: wav, sampleRate: UInt32(MicRecorder.sampleRate), channels: 1))
            phase = .idle
            return (result.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            lastError = "Transcription failed: \(error.localizedDescription)"
            phase = .idle
            return ""
        }
    }

    func cancelRecording() {
        _ = recorder.stop()
        if phase == .recording { phase = .idle }
    }

    // MARK: - TTS

    /// Reads `text` aloud if `speakReplies` is on. Replaces anything already playing.
    func speakIfEnabled(_ text: String) {
        guard speakReplies else { return }
        speak(text)
    }

    func speak(_ text: String) {
        stopSpeaking()
        let chunks = Self.chunks(of: Self.speakable(text))
        guard !chunks.isEmpty else { return }
        speakTask = Task { [weak self] in
            guard let self else { return }
            do {
                let model = try await self.ttsModel()
                try Self.configureSession()
                for chunk in chunks {
                    try Task.checkCancellation()
                    self.phase = .synthesizing
                    let result = try await model.runAsync(
                        envelope: XybridEnvelope.text(text: chunk, voiceId: Self.ttsVoice, speed: 1.0))
                    try Task.checkCancellation()
                    guard let audio = result.audioBytes, !audio.isEmpty else { continue }
                    self.phase = .speaking
                    try await self.play(Self.playableWAV(audio, metadata: result.envelope.metadata))
                }
            } catch is CancellationError {
            } catch {
                if !Task.isCancelled { self.lastError = "Speech failed: \(error.localizedDescription)" }
            }
            if !Task.isCancelled { self.phase = .idle }
        }
    }

    func stopSpeaking() {
        speakTask?.cancel()
        speakTask = nil
        player?.stop()
        player = nil
        playerDelegate.finish()
        if phase == .speaking || phase == .synthesizing { phase = .idle }
    }

    private func play(_ wav: Data) async throws {
        let p = try AVAudioPlayer(data: wav)
        p.delegate = playerDelegate
        player = p
        await withTaskCancellationHandler {
            await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
                playerDelegate.onFinish = { c.resume() }
                if !p.play() { playerDelegate.finish() }
            }
        } onCancel: {
            Task { @MainActor in
                SpeechService.shared.player?.stop()
                SpeechService.shared.playerDelegate.finish()
            }
        }
        player = nil
    }

    // MARK: - Helpers

    private static func configureSession() throws {
        let s = AVAudioSession.sharedInstance()
        try s.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothA2DP])
        try s.setActive(true)
    }

    /// Kokoro returns raw PCM16 (24 kHz mono); wrap it so AVAudioPlayer can read it.
    private static func playableWAV(_ audio: Data, metadata: [XybridMetadataEntry]) -> Data {
        if audio.count >= 4, audio.prefix(4) == Data("RIFF".utf8) { return audio }
        let rate = metadata.first { $0.key == "sample_rate" }.flatMap { Int($0.value) } ?? 24_000
        return WAV.encode(pcm16Bytes: audio, sampleRate: rate)
    }

    /// Drop tool-call blocks and markdown punctuation that sound bad spoken.
    static func speakable(_ text: String) -> String {
        var t = text.replacingOccurrences(of: #"<tool_call>[\s\S]*?</tool_call>"#, with: "", options: .regularExpression)
        t = t.replacingOccurrences(of: #"```[\s\S]*?```"#, with: " (code omitted) ", options: .regularExpression)
        t = t.replacingOccurrences(of: #"[*_#`>|]"#, with: "", options: .regularExpression)
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Sentence-ish chunks of at most ~300 characters so the first audio starts quickly.
    static func chunks(of text: String, limit: Int = 300) -> [String] {
        var out: [String] = []
        var current = ""
        text.enumerateSubstrings(in: text.startIndex..., options: .bySentences) { s, _, _, _ in
            guard let s = s?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty else { return }
            if !current.isEmpty, current.count + s.count + 1 > limit {
                out.append(current); current = ""
            }
            current += current.isEmpty ? s : " " + s
            while current.count > limit {
                let cut = current.index(current.startIndex, offsetBy: limit)
                let head = current[..<cut]
                let split = head.lastIndex(of: " ") ?? cut
                out.append(String(current[..<split]))
                current = String(current[split...]).trimmingCharacters(in: .whitespaces)
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }
}

enum SpeechError: LocalizedError {
    case downloadFailed(String)
    var errorDescription: String? {
        switch self {
        case .downloadFailed(let id): return "Downloading \(id) failed. Check your network connection."
        }
    }
}

/// AVAudioPlayer delegate that resumes one waiting continuation.
private final class PlayerDelegate: NSObject, AVAudioPlayerDelegate {
    var onFinish: (() -> Void)?
    func finish() {
        let f = onFinish
        onFinish = nil
        f?()
    }
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        DispatchQueue.main.async { self.finish() }
    }
    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        DispatchQueue.main.async { self.finish() }
    }
}

/// Microphone capture → 16 kHz mono PCM16, accumulated until `stop()`.
final class MicRecorder: @unchecked Sendable {
    static let sampleRate = 16_000
    private let engine = AVAudioEngine()
    private let lock = NSLock()
    private var samples: [Int16] = []
    private var running = false
    private let maxSamples = 16_000 * 120   // 2 minutes cap

    func start() throws {
        lock.lock(); samples = []; lock.unlock()
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        guard inFormat.sampleRate > 0,
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: Double(Self.sampleRate),
                                            channels: 1, interleaved: true),
              let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else { throw RecorderError.noInput }

        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buffer, _ in
            self?.append(buffer, converter: converter, outFormat: outFormat)
        }
        engine.prepare()
        try engine.start()
        running = true
    }

    private func append(_ buffer: AVAudioPCMBuffer, converter: AVAudioConverter, outFormat: AVAudioFormat) {
        let ratio = outFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
        var fed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, let ch = out.int16ChannelData, out.frameLength > 0 else { return }
        let chunk = UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength))
        lock.lock()
        if samples.count < maxSamples { samples.append(contentsOf: chunk) }
        lock.unlock()
    }

    /// Stops capture and returns everything recorded.
    func stop() -> [Int16] {
        if running {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            running = false
        }
        lock.lock(); defer { lock.unlock() }
        let s = samples
        samples = []
        return s
    }

    enum RecorderError: LocalizedError {
        case noInput
        var errorDescription: String? { "No microphone input is available." }
    }
}

/// Minimal RIFF/WAVE (PCM16, mono) encoder.
enum WAV {
    static func encode(pcm16: [Int16], sampleRate: Int) -> Data {
        let bytes = pcm16.withUnsafeBufferPointer { Data(buffer: $0) }   // little-endian on Apple silicon
        return encode(pcm16Bytes: bytes, sampleRate: sampleRate)
    }

    static func encode(pcm16Bytes pcm: Data, sampleRate: Int, channels: Int = 1) -> Data {
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let byteRate = sampleRate * channels * 2
        d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + pcm.count))
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(UInt16(channels))
        u32(UInt32(sampleRate)); u32(UInt32(byteRate)); u16(UInt16(channels * 2)); u16(16)
        d.append(contentsOf: Array("data".utf8)); u32(UInt32(pcm.count))
        d.append(pcm)
        return d
    }
}
