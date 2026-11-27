import Foundation
import Observation
import SwiftUI
import UIKit

/// Receipts on disk: Application Support/receipts/<UTC stamp>-<model id>.json.
enum ReceiptStore {
    static var directory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = base.appending(path: "receipts", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    struct Saved: Identifiable, Sendable {
        let url: URL
        let receipt: BenchReceipt
        var id: String { url.lastPathComponent }
    }

    @discardableResult
    static func save(_ receipt: BenchReceipt) throws -> URL {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        let safe = receipt.model.id.map { $0.isLetter || $0.isNumber || "._-".contains($0) ? $0 : "_" }
        var url = directory.appending(path: "\(f.string(from: Date()))-\(String(safe)).json")
        var n = 2
        while FileManager.default.fileExists(atPath: url.path) {
            url = directory.appending(path: "\(f.string(from: Date()))-\(String(safe))-\(n).json")
            n += 1
        }
        try BenchReceipt.encoder.encode(receipt).write(to: url, options: .atomic)
        return url
    }

    /// Newest first; unreadable files are skipped.
    static func all() -> [Saved] {
        let urls = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return urls.filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url),
                      let r = try? BenchReceipt.decoder.decode(BenchReceipt.self, from: data) else { return nil }
                return Saved(url: url, receipt: r)
            }
    }

    private static let uploadedKey = "bench.uploadedReceipts"

    static func isUploaded(_ name: String) -> Bool {
        (UserDefaults.standard.stringArray(forKey: uploadedKey) ?? []).contains(name)
    }

    static func markUploaded(_ name: String) {
        var names = UserDefaults.standard.stringArray(forKey: uploadedKey) ?? []
        guard !names.contains(name) else { return }
        names.append(name)
        UserDefaults.standard.set(Array(names.suffix(2000)), forKey: uploadedKey)
    }

    /// POST to the worker's /receipts (worker/receipts.py). nil = not configured.
    static func upload(_ url: URL) async throws -> String? {
        guard let cfg = WorkerConfig.load() else { return nil }
        var req = URLRequest(url: cfg.baseURL.appending(path: "receipts"))
        req.httpMethod = "POST"
        req.timeoutInterval = 30
        req.setValue("Bearer \(cfg.token)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try Data(contentsOf: url)
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        guard (200..<300).contains(code) else {
            throw WorkerError.http(code, body?["error"] as? String ?? "")
        }
        return body?["name"] as? String ?? "uploaded"
    }
}

/// Bench screen state: runs one model or the whole catalog, sequentially,
/// holding the GPU gate (so chat, batches and fleet tasks wait), saving and
/// uploading a receipt per model.
///
/// Foreground only: the idle timer is disabled while running, and leaving the
/// foreground stops the run at the next decode step (Metal rejects background
/// GPU work), saving the model in progress as an `aborted` receipt.
@MainActor @Observable
final class BenchCoordinator {
    private(set) var isRunning = false
    private(set) var status = "Idle"
    private(set) var fraction = 0.0
    private(set) var saved: [ReceiptStore.Saved] = ReceiptStore.all()
    private(set) var uploadStatus: [String: String] = [:]   // receipt file name → result

    @ObservationIgnored private let backend: MLXBackend?
    @ObservationIgnored private let onModelUnloaded: @MainActor () -> Void
    @ObservationIgnored private let releaseOtherModels: @MainActor () async -> Void
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private let cancelFlag = CancelFlag()
    @ObservationIgnored private var abortReason: String?
    @ObservationIgnored private var idleTimerWasDisabled = false
    @ObservationIgnored private var backgroundGrace: UIBackgroundTaskIdentifier = .invalid
    @ObservationIgnored private var uploading = false

    init(backend: MLXBackend?, onModelUnloaded: @escaping @MainActor () -> Void,
         releaseOtherModels: @escaping @MainActor () async -> Void = {}) {
        self.backend = backend
        self.onModelUnloaded = onModelUnloaded
        self.releaseOtherModels = releaseOtherModels
    }

    var canRun: Bool { backend != nil && !isRunning }

    func refresh() { saved = ReceiptStore.all() }

    /// `entries` run in order; catalog entries added later (e.g. Inkling) are included automatically.
    func run(_ entries: [ModelCatalog.Entry], repetitions: Int) {
        guard let backend, !isRunning else { return }
        guard UIApplication.shared.applicationState == .active else { return }
        isRunning = true
        abortReason = nil
        cancelFlag.reset()
        idleTimerWasDisabled = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true   // auto-lock would background the app mid-run
        let flag = cancelFlag
        status = "Waiting for the model to be free"
        // Strong self: the coordinator lives as long as the app (AppModel).
        task = Task {
            await releaseOtherModels()   // embedder etc.; takes the gate itself, so before ours
            await GPUGate.shared.acquire("benchmark")
            onModelUnloaded()   // drop the chat's session so its weights can actually be freed
            let count = entries.count
            for (i, entry) in entries.enumerated() {
                if flag.isSet || Task.isCancelled { break }
                let runner = BenchmarkRunner(backend: backend)
                var receipt = await runner.run(
                    entry: entry, repetitions: repetitions,
                    progress: { [self] p in
                        Task { @MainActor in
                            guard self.isRunning, !flag.isSet else { return }
                            self.status = "[\(i + 1)/\(count)] \(p.message)"
                            self.fraction = (Double(i) + p.fraction) / Double(count)
                        }
                    },
                    isCancelled: { flag.isSet })
                if receipt.status == "cancelled", let reason = abortReason {
                    receipt.status = "aborted"
                    receipt.error = reason
                }
                await store(receipt)
            }
            await GPUGate.shared.release()
            finish()
        }
    }

    func cancel() {
        cancelFlag.set()
        status = "Stopping after the current step…"
    }

    /// Called from the app's scene-phase handler. Leaving `.active` stops the run
    /// before any more GPU work is submitted; a short background grace lets the
    /// in-flight step finish and the receipt be saved.
    func scenePhaseChanged(_ phase: ScenePhase) {
        guard isRunning, phase != .active, abortReason == nil else { return }
        abortReason = "app left the foreground (\(phase == .background ? "background" : "inactive")) during the run"
        cancelFlag.set()
        status = "Aborting: the app left the foreground"
        if backgroundGrace == .invalid {
            backgroundGrace = UIApplication.shared.beginBackgroundTask(withName: "bench-abort") { [weak self] in
                Task { @MainActor in self?.endBackgroundGrace() }
            }
        }
    }

    private func endBackgroundGrace() {
        guard backgroundGrace != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundGrace)
        backgroundGrace = .invalid
    }

    private func finish() {
        isRunning = false
        task = nil
        status = abortReason.map { "Aborted: \($0)" } ?? (cancelFlag.isSet ? "Stopped" : "Done")
        fraction = 0
        UIApplication.shared.isIdleTimerDisabled = idleTimerWasDisabled
        onModelUnloaded()   // the chat reloads its model on the next message
        endBackgroundGrace()
    }

    private func store(_ receipt: BenchReceipt) async {
        do {
            let url = try ReceiptStore.save(receipt)
            refresh()
            await upload(url)
        } catch {
            status = "Couldn't save receipt: \(error.localizedDescription)"
        }
    }

    func upload(_ url: URL) async {
        let name = url.lastPathComponent
        do {
            if let stored = try await ReceiptStore.upload(url) {
                ReceiptStore.markUploaded(name)
                uploadStatus[name] = "Uploaded as \(stored)"
            } else {
                uploadStatus[name] = "Worker not configured"
            }
        } catch {
            uploadStatus[name] = "Upload failed (retried when the Bench tab opens): \(error.localizedDescription)"
        }
    }

    /// Retry receipts that never reached the worker. Safe to repeat: the worker
    /// dedupes on receipt_id.
    func uploadPending() async {
        guard !uploading, !isRunning, WorkerConfig.load() != nil else { return }
        uploading = true
        defer { uploading = false }
        for s in saved where !ReceiptStore.isUploaded(s.id) {
            await upload(s.url)
            if uploadStatus[s.id]?.hasPrefix("Upload failed") == true { break }   // worker unreachable
        }
    }
}

final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    func set() { lock.withLock { flag = true } }
    func reset() { lock.withLock { flag = false } }
    var isSet: Bool { lock.withLock { flag } }
}
