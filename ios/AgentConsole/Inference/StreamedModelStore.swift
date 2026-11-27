import Foundation

/// Downloads SSD-streamed model builds (core shards + `experts.bin` + `experts.json`, made offline
/// by `tools/expert_repack.py`) from a configurable server. No MLX import.
///
/// The server must answer `GET <base>/<name>` with `{"files": [{"name", "size"}]}` and serve
/// `GET <base>/<name>/<file>` with single byte ranges. The worker does both with `--models-dir`
/// (`worker/models.py`); by default `<base>` is the paired worker's URL + `/models` and requests
/// carry its token. A custom base URL gets the worker token only when it is on the worker's host.
enum StreamedModelStore {
    /// Settings: an alternative base URL (blank = the paired worker's `/models`).
    static let baseURLKey = "streamedModels.baseURL"
    private static let completeMarker = ".complete.json"

    struct RemoteFile: Codable, Hashable, Sendable {
        let name: String
        let size: Int64
    }

    private struct Listing: Codable { let files: [RemoteFile] }

    enum StoreError: LocalizedError {
        case noSource
        case badListing(String)
        case http(Int, String)
        case sizeMismatch(String, expected: Int64, got: Int64)
        case incomplete(String)

        var errorDescription: String? {
            switch self {
            case .noSource:
                return "Streamed builds download from your worker (or the URL in Settings). Pair a worker first."
            case .badListing(let m): return "The model server's file list is not usable: \(m)"
            case .http(let code, let what): return "Download of \(what) failed with HTTP \(code)."
            case .sizeMismatch(let file, let expected, let got):
                return "\(file) is \(got) bytes, the server listed \(expected)."
            case .incomplete(let name):
                return "\(name) is missing experts.json or experts.bin; was it made with tools/expert_repack.py?"
            }
        }
    }

    /// Application Support/StreamedModels, excluded from backups.
    static var root: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("StreamedModels", isDirectory: true)
    }

    static func directory(for build: ModelCatalog.StreamedBuild) -> URL {
        root.appendingPathComponent(build.name, isDirectory: true)
    }

    static func isDownloaded(_ build: ModelCatalog.StreamedBuild) -> Bool {
        FileManager.default.fileExists(atPath: directory(for: build).appendingPathComponent(completeMarker).path)
    }

    /// Base URL and the token to send, from Settings or the paired worker.
    static func source() throws -> (base: URL, token: String?) {
        let worker = WorkerConfig.load()
        if let custom = UserDefaults.standard.string(forKey: baseURLKey)?
            .trimmingCharacters(in: .whitespaces), !custom.isEmpty, let url = URL(string: custom) {
            let sameHost = worker.map { $0.baseURL.host == url.host && $0.baseURL.port == url.port } ?? false
            return (url, sameHost ? worker?.token : nil)
        }
        guard let worker else { throw StoreError.noSource }
        return (worker.baseURL.appendingPathComponent("models", isDirectory: true), worker.token)
    }

    /// The local directory of `build`, downloading what is missing first. Resumes partial files.
    static func ensureDownloaded(_ build: ModelCatalog.StreamedBuild,
                                 progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        let dir = directory(for: build)
        if isDownloaded(build) { progress(1); return dir }
        try prepare(dir)
        let (base, token) = try source()
        let buildURL = base.appendingPathComponent(build.name, isDirectory: false)
        var request = URLRequest(url: buildURL, timeoutInterval: 30)
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let (data, response) = try await URLSession.shared.data(for: request)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw StoreError.http(code, build.name) }
        let files: [RemoteFile]
        do { files = try JSONDecoder().decode(Listing.self, from: data).files } catch {
            throw StoreError.badListing(error.localizedDescription)
        }
        guard files.contains(where: { $0.name == "experts.json" }), files.contains(where: { $0.name == "experts.bin" }),
              files.allSatisfy({ isSafeName($0.name) }) else { throw StoreError.incomplete(build.name) }

        let total = max(1, files.reduce(Int64(0)) { $0 + $1.size })
        let counter = ByteCounter(total: total, progress: progress)
        // Small files first, experts.bin last: a failure surfaces before the long transfer.
        for file in files.sorted(by: { $0.size < $1.size }) {
            try Task.checkCancellation()
            let target = dir.appendingPathComponent(file.name)
            if let size = fileSize(target), size == file.size { counter.add(size); continue }
            let part = dir.appendingPathComponent(file.name + ".part")
            let have = fileSize(part) ?? 0
            counter.add(min(have, file.size))
            var fileRequest = URLRequest(url: buildURL.appendingPathComponent(file.name), timeoutInterval: 120)
            if let token { fileRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
            try await RangeDownload(part: part, alreadyHave: have, counter: counter).run(fileRequest, what: file.name)
            let got = fileSize(part) ?? -1
            guard got == file.size else { throw StoreError.sizeMismatch(file.name, expected: file.size, got: got) }
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.moveItem(at: part, to: target)
        }
        let marker = try JSONEncoder().encode(files)
        try marker.write(to: dir.appendingPathComponent(completeMarker), options: .atomic)
        progress(1)
        return dir
    }

    private static func prepare(_ dir: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true,
                               attributes: [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication])
        var rootURL = root
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? rootURL.setResourceValues(values)
    }

    private static func isSafeName(_ name: String) -> Bool {
        !name.isEmpty && !name.hasPrefix(".") && !name.contains("/") && !name.contains("\\") && name != ".."
    }

    private static func fileSize(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
    }
}

/// Bytes received across files, reported as a fraction.
private final class ByteCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var done: Int64 = 0
    private var lastReported = -1.0
    let total: Int64
    let progress: @Sendable (Double) -> Void

    init(total: Int64, progress: @escaping @Sendable (Double) -> Void) {
        self.total = total
        self.progress = progress
    }

    func add(_ bytes: Int64) {
        let fraction: Double? = lock.withLock {
            done += bytes
            let f = min(1, Double(done) / Double(total))
            guard f - lastReported >= 0.001 else { return nil }
            lastReported = f
            return f
        }
        if let fraction { progress(fraction) }
    }
}

/// One file, appended to `part` from byte `alreadyHave` with a `Range` request. A server that
/// ignores the range (200) restarts the file.
private final class RangeDownload: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let part: URL
    private let alreadyHave: Int64
    private let counter: ByteCounter
    private var handle: FileHandle?
    private var failure: Error?
    private var continuation: CheckedContinuation<Void, Error>?
    private var what = ""
    private let queue: OperationQueue = {
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 1
        return q
    }()

    init(part: URL, alreadyHave: Int64, counter: ByteCounter) {
        self.part = part
        self.alreadyHave = alreadyHave
        self.counter = counter
    }

    func run(_ request: URLRequest, what: String) async throws {
        self.what = what
        var request = request
        if alreadyHave > 0 { request.setValue("bytes=\(alreadyHave)-", forHTTPHeaderField: "Range") }
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: queue)
        defer { session.finishTasksAndInvalidate() }
        let task = session.dataTask(with: request)
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                queue.addOperation {
                    self.continuation = c
                    task.resume()
                }
            }
        } onCancel: {
            task.cancel()
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        do {
            let fm = FileManager.default
            if code == 206 {
                if !fm.fileExists(atPath: part.path) { fm.createFile(atPath: part.path, contents: nil) }
                handle = try FileHandle(forWritingTo: part)
                try handle?.seekToEnd()
            } else if code == 200 {
                fm.createFile(atPath: part.path, contents: nil)   // truncates: the server sent the whole file
                handle = try FileHandle(forWritingTo: part)
            } else {
                failure = StreamedModelStore.StoreError.http(code, what)
                completionHandler(.cancel)
                return
            }
            completionHandler(.allow)
        } catch {
            failure = error
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let handle, failure == nil else { return }
        do {
            try handle.write(contentsOf: data)
            counter.add(Int64(data.count))
        } catch {
            failure = error
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? handle?.close()
        handle = nil
        let result = failure ?? error
        if let result { continuation?.resume(throwing: result) } else { continuation?.resume() }
        continuation = nil
    }
}
