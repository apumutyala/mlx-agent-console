import AVFoundation
import Foundation
import Observation
import UIKit
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
    /// Kokoro v1.0's default voice (`voices.default` of kokoro-82m in the Xybrid
    /// registry). Voice ids are exact-match; there is no voice called "af".
    static let defaultVoice = "af_heart"
    /// Selecting an LLM at least this big (download GB) releases the speech models first.
    static let largeModelGB = 3.0

    enum Phase: Equatable {
        case idle
        case downloading(model: String, progress: Double)  // first use: needs network
        case loading(model: String)
        case recording
        case transcribing
        case synthesizing
        case speaking
    }

    enum FinishReason { case user, limit }

    private(set) var phase: Phase = .idle
    private(set) var lastError: String?
    /// Audio from a recording that couldn't be transcribed or delivered yet
    /// (download failed, interrupted, agent busy). Kept so the user can retry.
    private(set) var pendingAudio: [Int16]?
    /// Voices the loaded Kokoro model reports (empty until it has loaded once).
    private(set) var voices: [XybridVoiceInfo] = []
    /// Persisted TTS voice; checked against the model's catalog before each reply.
    var voiceId: String = UserDefaults.standard.string(forKey: "speech.voice") ?? SpeechService.defaultVoice {
        didSet { UserDefaults.standard.set(voiceId, forKey: "speech.voice") }
    }
    /// Read assistant replies aloud. Persisted.
    var speakReplies: Bool = UserDefaults.standard.bool(forKey: "speech.speakReplies") {
        didSet {
            UserDefaults.standard.set(speakReplies, forKey: "speech.speakReplies")
            if !speakReplies { stopSpeaking(cancelLoad: true) }
        }
    }
    /// Receives finished transcripts. Set by the mic button; auto-stops (time
    /// limit) deliver through it too.
    var onTranscript: ((String) -> Void)?
    /// True while the agent is running a turn (kept by `speechReadsReplies`):
    /// a transcript can't be sent then, so it is held instead of dropped.
    var hostIsBusy = false

    var isRecording: Bool { phase == .recording }
    /// Model being downloaded or loaded, if any.
    var loadingModel: String? {
        switch phase {
        case .downloading(let m, _), .loading(let m): return m
        default: return nil
        }
    }
    /// Speech-to-text work in progress (the mic is unavailable). A Kokoro
    /// download does not block the mic.
    var isBusy: Bool { isTranscribing || phase == .transcribing || loadingModel == Self.asrModelId }

    private var asr: XybridModel?
    private var tts: XybridModel?
    /// In-flight loads, shared by every caller so a model is never loaded twice at once.
    private var loads: [String: (token: UUID, task: Task<XybridModel, Error>)] = [:]
    private var isTranscribing = false
    private let recorder = MicRecorder()
    private var speakTask: Task<Void, Never>?
    private var player: AVAudioPlayer?
    private let playerDelegate = PlayerDelegate()
    private var observers: [NSObjectProtocol] = []

    private init() {
        recorder.onLimitReached = {
            Task { @MainActor in await SpeechService.shared.finishRecording(reason: .limit) }
        }
        let nc = NotificationCenter.default
        observers.append(nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { n in
            let raw = (n.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt) ?? 0
            Task { @MainActor in SpeechService.shared.handleInterruption(AVAudioSession.InterruptionType(rawValue: raw)) }
        })
        observers.append(nc.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main) { n in
            let raw = (n.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt) ?? 0
            Task { @MainActor in SpeechService.shared.handleRouteChange(AVAudioSession.RouteChangeReason(rawValue: raw)) }
        })
        observers.append(nc.addObserver(forName: .AVAudioEngineConfigurationChange, object: recorder.engine, queue: .main) { _ in
            Task { @MainActor in SpeechService.shared.handleEngineConfigurationChange() }
        })
        observers.append(nc.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { _ in
            Task { @MainActor in SpeechService.shared.releaseModels() }
        })
    }

    // MARK: - Model loading

    /// Whether a model still needs a download (and therefore network) before it can load.
    func needsDownload(_ id: String) -> Bool {
        !((try? Xybrid.hasCachedModelData(id)) ?? false)
    }

    /// The loaded model, or a shared in-flight load of it. Cancelling the caller
    /// does not cancel the load (another caller may be waiting); `cancelLoad` does.
    private func model(_ id: String) async throws -> XybridModel {
        if id == Self.asrModelId, let asr { return asr }
        if id == Self.ttsModelId, let tts { return tts }
        if let inFlight = loads[id] { return try await inFlight.task.value }
        let token = UUID()
        let task = Task<XybridModel, Error> { @MainActor in
            defer {
                if self.loads[id]?.token == token { self.loads[id] = nil }
                self.endLoadingPhase(id)
            }
            let m = try await self.loadModel(id)
            if id == Self.asrModelId {
                self.asr = m
            } else {
                self.tts = m
                self.voices = m.voices()
            }
            return m
        }
        loads[id] = (token, task)
        return try await task.value
    }

    private func cancelLoad(_ id: String) {
        loads[id]?.task.cancel()
        loads[id] = nil
        endLoadingPhase(id)
    }

    private func loadModel(_ id: String) async throws -> XybridModel {
        // Opt out of Xybrid's client telemetry header before the SDK starts (privacy default).
        setenv("XYBRID_TELEMETRY_OPTOUT", "1", 1)
        Xybrid.initialize()   // no apiKey: fully on-device; idempotent
        Xybrid.setAutoRelease(true)   // under memory pressure, evict idle models before loading
        let loader = Xybrid.model(id)
        if needsDownload(id), let download = loader.download() {
            showLoading(.downloading(model: id, progress: 0))
            try await withTaskCancellationHandler {
                for await status in download.progress() {
                    showLoading(.downloading(model: id, progress: Double(status.progress)))
                    if status.state == .failed { throw SpeechError.downloadFailed(id, download.error()) }
                    if status.state == .cancelled { throw CancellationError() }
                    if status.state != .downloading { break }
                }
            } onCancel: {
                download.cancel()   // ending progress() only unsubscribes; this stops the transfer
            }
            try Task.checkCancellation()
        }
        showLoading(.loading(model: id))
        return try await loader.load()
    }

    /// Shows download/load progress unless the UI is in a recording or speaking phase.
    private func showLoading(_ p: Phase) {
        switch phase {
        case .idle, .downloading, .loading: phase = p
        default: break
        }
    }

    private func endLoadingPhase(_ id: String) {
        if loadingModel == id {
            phase = .idle
            deactivateSessionIfIdle()
        }
    }

    /// Frees Whisper and Kokoro (memory warning, or a large LLM was selected).
    /// Released models reload from the on-disk cache on next use; runs in flight are skipped.
    func releaseModels() {
        Xybrid.releaseMemory()
        if !isTranscribing && !isRecording { asr = nil }
        if speakTask == nil { tts = nil }
    }

    /// Settings hook: frees the speech models before a big LLM loads.
    func modelSelected(_ entry: ModelCatalog.Entry) {
        if entry.downloadGB >= Self.largeModelGB { releaseModels() }
    }

    /// Loads Kokoro so its voice list can be shown (downloads on first use).
    func loadVoices() {
        Task {
            do { _ = try await self.model(Self.ttsModelId) } catch {
                if !(error is CancellationError) { self.lastError = "Couldn't load voices: \(error.localizedDescription)" }
            }
        }
    }

    // MARK: - Push-to-talk STT

    /// Starts capturing the microphone. Any speech playback stops first.
    func startRecording() async {
        guard !isRecording, !isBusy else { return }
        stopSpeaking()
        lastError = nil
        guard await AVAudioApplication.requestRecordPermission() else {
            lastError = "Microphone access is off. Enable it in Settings > Privacy > Microphone."
            return
        }
        guard !isRecording, !isBusy else { return }   // state may have changed during the prompt
        stopSpeaking()
        pendingAudio = nil   // a new recording replaces an unsent one
        do {
            try Self.configureSession()
            try recorder.start()
            phase = .recording
            prefetchASR()
        } catch {
            _ = recorder.stop()
            lastError = "Couldn't start recording: \(error.localizedDescription)"
            phase = .idle
            deactivateSessionIfIdle()
        }
    }

    /// Fetch Whisper while the user talks, so a first-use download overlaps the recording.
    private func prefetchASR() {
        guard asr == nil, loads[Self.asrModelId] == nil else { return }
        Task { _ = try? await self.model(Self.asrModelId) }
    }

    /// Stops capturing, transcribes, and hands the transcript to `onTranscript`.
    /// If transcription fails the audio is kept in `pendingAudio` for a retry.
    func finishRecording(reason: FinishReason = .user) async {
        guard isRecording else { return }
        let pcm = recorder.stop()
        phase = .idle
        guard !pcm.isEmpty else { deactivateSessionIfIdle(); return }
        if reason == .limit {
            lastError = "Reached the \(MicRecorder.maxSeconds / 60)-minute limit, so recording stopped and was sent."
        }
        await transcribe(pcm)
    }

    /// Transcribes the kept recording again (after a failed download, an interruption…).
    func retryTranscription() async {
        guard let pcm = pendingAudio, !isRecording, !isBusy else { return }
        lastError = nil
        await transcribe(pcm)
    }

    func discardPendingAudio() {
        pendingAudio = nil
        lastError = nil
    }

    private func transcribe(_ pcm: [Int16]) async {
        pendingAudio = pcm   // kept until the transcript is delivered
        isTranscribing = true
        defer { isTranscribing = false }
        do {
            let whisper = try await self.model(Self.asrModelId)
            phase = .transcribing
            let wav = WAV.encode(pcm16: pcm, sampleRate: MicRecorder.sampleRate)
            let result = try await whisper.runAsync(
                envelope: XybridEnvelope.audio(pcmData: wav, sampleRate: UInt32(MicRecorder.sampleRate), channels: 1))
            if phase == .transcribing { phase = .idle }
            deactivateSessionIfIdle()
            let text = (result.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else {
                pendingAudio = nil
                lastError = "No speech was recognized."
                return
            }
            guard !hostIsBusy, let deliver = onTranscript else {
                lastError = "The agent is busy, so the recording was kept. Tap Send when the reply finishes."
                return
            }
            pendingAudio = nil
            deliver(text)
        } catch {
            if phase == .transcribing { phase = .idle }
            lastError = "Transcription failed: \(error.localizedDescription) The recording was kept; tap Retry."
            deactivateSessionIfIdle()
        }
    }

    /// Stops capture without transcribing, keeping the audio for Send/Discard.
    func suspendRecording(_ message: String) {
        guard isRecording else { return }
        let pcm = recorder.stop()
        phase = .idle
        if !pcm.isEmpty { pendingAudio = pcm }
        lastError = pcm.isEmpty ? nil : message
        deactivateSessionIfIdle()
    }

    func cancelRecording() {
        _ = recorder.stop()
        if phase == .recording { phase = .idle }
        deactivateSessionIfIdle()
    }

    // MARK: - Audio session events

    private func handleInterruption(_ type: AVAudioSession.InterruptionType?) {
        guard type == .began else { return }
        // The system paused our player without a delegate callback; end the read-aloud cleanly.
        stopSpeaking()
        suspendRecording("Recording was interrupted. Your audio was kept: tap Send or Discard.")
    }

    private func handleRouteChange(_ reason: AVAudioSession.RouteChangeReason?) {
        // Headphones unplugged mid-reply: stop rather than continue on the speaker.
        if reason == .oldDeviceUnavailable, phase == .speaking || phase == .synthesizing {
            stopSpeaking()
        }
    }

    /// The engine stops itself when the input device changes (headset, USB mic).
    private func handleEngineConfigurationChange() {
        guard isRecording else { return }
        do {
            try recorder.restart()
        } catch {
            suspendRecording("The microphone changed and recording couldn't continue. Your audio was kept: tap Send or Discard.")
        }
    }

    // MARK: - TTS

    /// Reads `text` aloud if `speakReplies` is on. Replaces anything already playing.
    func speakIfEnabled(_ text: String) {
        guard speakReplies else { return }
        speak(text)
    }

    func speak(_ text: String) {
        // Never take the audio session away from a live recording or transcription.
        guard !isRecording, !isBusy else { return }
        stopSpeaking()
        let chunks = Self.chunks(of: Self.speakable(text))
        guard !chunks.isEmpty else { return }
        speakTask = Task { @MainActor in
            do {
                let kokoro = try await self.model(Self.ttsModelId)
                try Task.checkCancellation()
                guard !self.isRecording, !self.isBusy else { throw CancellationError() }
                let voice = self.resolvedVoice(kokoro)
                try Self.configureSession()
                for chunk in chunks {
                    try Task.checkCancellation()
                    self.phase = .synthesizing
                    let result = try await kokoro.runAsync(
                        envelope: XybridEnvelope.text(text: chunk, voiceId: voice, speed: 1.0))
                    try Task.checkCancellation()
                    guard let audio = result.audioBytes, !audio.isEmpty else { continue }
                    self.phase = .speaking
                    try await self.play(Self.playableWAV(audio, metadata: result.envelope.metadata))
                }
            } catch is CancellationError {
            } catch {
                if !Task.isCancelled { self.lastError = "Speech failed: \(error.localizedDescription)" }
            }
            if !Task.isCancelled {
                if self.phase == .speaking || self.phase == .synthesizing { self.phase = .idle }
                self.speakTask = nil
                self.deactivateSessionIfIdle()
            }
        }
    }

    /// The chosen voice if the model has it, else the model's default (persisted).
    /// `nil` lets the SDK pick the model default.
    private func resolvedVoice(_ model: XybridModel) -> String? {
        if model.voice(voiceId: voiceId) != nil { return voiceId }
        guard let fallback = model.defaultVoice()?.id else { return nil }
        voiceId = fallback
        return fallback
    }

    /// Stops playback. `cancelLoad` also abandons a Kokoro download/load in
    /// flight (the speaker toggle was turned off).
    func stopSpeaking(cancelLoad: Bool = false) {
        speakTask?.cancel()
        speakTask = nil
        player?.stop()
        player = nil
        playerDelegate.finish()
        if phase == .speaking || phase == .synthesizing { phase = .idle }
        if cancelLoad { self.cancelLoad(Self.ttsModelId) }
        deactivateSessionIfIdle()
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

    /// Hand the audio back to other apps (music resumes) once nothing is recording or playing.
    private func deactivateSessionIfIdle() {
        guard !isRecording, player == nil, speakTask == nil,
              phase != .speaking, phase != .synthesizing else { return }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
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
    case downloadFailed(String, String?)
    var errorDescription: String? {
        switch self {
        case .downloadFailed(let id, let detail):
            return "Downloading \(id) failed\(detail.map { " (\($0))" } ?? ""). Check your network connection."
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
    static let maxSeconds = 120
    let engine = AVAudioEngine()
    private let lock = NSLock()
    private var samples: [Int16] = []
    private var running = false
    private var limitReached = false
    private let maxSamples = MicRecorder.sampleRate * MicRecorder.maxSeconds
    /// Called once (on the audio thread) when the cap is hit; the owner stops and transcribes.
    var onLimitReached: (@Sendable () -> Void)?

    func start() throws {
        lock.lock(); samples = []; limitReached = false; lock.unlock()
        try installTapAndStart()
    }

    /// After an engine configuration change (input device changed) the engine
    /// has stopped itself: re-tap with the new input format, keeping what was captured.
    func restart() throws {
        guard running else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        running = false
        try installTapAndStart()
    }

    private func installTapAndStart() throws {
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
        var hitLimit = false
        lock.lock()
        if !limitReached {
            samples.append(contentsOf: chunk.prefix(maxSamples - samples.count))
            if samples.count >= maxSamples { limitReached = true; hitLimit = true }
        }
        lock.unlock()
        if hitLimit { onLimitReached?() }
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
