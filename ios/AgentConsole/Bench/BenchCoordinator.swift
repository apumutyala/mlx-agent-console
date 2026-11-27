import Foundation
import Observation

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
@MainActor @Observable
final class BenchCoordinator {
    private(set) var isRunning = false
    private(set) var status = "Idle"
    private(set) var fraction = 0.0
    private(set) var saved: [ReceiptStore.Saved] = ReceiptStore.all()
    private(set) var uploadStatus: [String: String] = [:]   // receipt file name → result

    @ObservationIgnored private let backend: MLXBackend?
    @ObservationIgnored private let onModelUnloaded: @MainActor () -> Void
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private let cancelFlag = CancelFlag()

    init(backend: MLXBackend?, onModelUnloaded: @escaping @MainActor () -> Void) {
        self.backend = backend
        self.onModelUnloaded = onModelUnloaded
    }

    var canRun: Bool { backend != nil && !isRunning }

    func refresh() { saved = ReceiptStore.all() }

    /// `entries` run in order; catalog entries added later (e.g. Inkling) are included automatically.
    func run(_ entries: [ModelCatalog.Entry], repetitions: Int) {
        guard let backend, !isRunning else { return }
        isRunning = true
        cancelFlag.reset()
        let flag = cancelFlag
        status = "Waiting for the model to be free"
        // Strong self: the coordinator lives as long as the app (AppModel).
        task = Task {
            await GPUGate.shared.acquire("benchmark")
            onModelUnloaded()   // drop the chat's session so its weights can actually be freed
            let count = entries.count
            for (i, entry) in entries.enumerated() {
                if flag.isSet || Task.isCancelled { break }
                let runner = BenchmarkRunner(backend: backend)
                let receipt = await runner.run(
                    entry: entry, repetitions: repetitions,
                    progress: { [self] p in
                        Task { @MainActor in
                            self.status = "[\(i + 1)/\(count)] \(p.message)"
                            self.fraction = (Double(i) + p.fraction) / Double(count)
                        }
                    },
                    isCancelled: { flag.isSet })
                await store(receipt)
            }
            await GPUGate.shared.release()
            finish()
        }
    }

    func cancel() {
        cancelFlag.set()
        status = "Stopping after the current rep…"
    }

    private func finish() {
        isRunning = false
        task = nil
        status = cancelFlag.isSet ? "Stopped" : "Done"
        fraction = 0
        onModelUnloaded()   // the chat reloads its model on the next message
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
            uploadStatus[name] = try await ReceiptStore.upload(url).map { "Uploaded as \($0)" } ?? "Worker not configured"
        } catch {
            uploadStatus[name] = "Upload failed: \(error.localizedDescription)"
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
