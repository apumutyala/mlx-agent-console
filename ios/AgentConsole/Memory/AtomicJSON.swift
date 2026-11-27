import Foundation

/// Atomic JSON files in Application Support. `.atomic` writes go to a temp
/// file and rename over the target, so a crash mid-write leaves the previous
/// version intact. The files are the user's data: device backups include them,
/// and the app itself never uploads them.
enum AtomicJSON {
    static let root: URL = .applicationSupportDirectory.appending(path: "Memory", directoryHint: .isDirectory)

    static func directory(_ name: String) -> URL {
        let url = root.appending(path: name, directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.sortedKeys]
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    /// Nil when the file is missing. When it exists but doesn't decode, it is
    /// moved aside to `<name>.corrupt-<time>.json` (kept, never deleted) before
    /// returning nil, so the caller's next write can't destroy the only copy.
    static func read<T: Decodable>(_ type: T.Type, from url: URL) -> T? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            quarantine(url, reason: String(describing: error))
            return nil
        }
    }

    /// Arrays decode element by element: bad elements are skipped (and a copy
    /// of the original file kept) instead of losing every record.
    static func readArray<T: Decodable>(_ type: T.Type, from url: URL) -> [T]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard let boxes = try? decoder.decode([Failable<T>].self, from: data) else {
            quarantine(url, reason: "not a JSON array")
            return nil
        }
        let good = boxes.compactMap(\.value)
        if good.count < boxes.count {
            preserveCopy(url, reason: "\(boxes.count - good.count) unreadable records skipped")
        }
        return good
    }

    private struct Failable<T: Decodable>: Decodable {
        let value: T?
        init(from decoder: Decoder) throws { value = try? T(from: decoder) }
    }

    /// Files moved or copied aside this launch, for the Memory screen.
    nonisolated(unsafe) private static var _problems: [String] = []
    private static let lock = NSLock()
    static var problems: [String] { lock.withLock { _problems } }

    private static func asideURL(_ url: URL) -> URL {
        let stamp = Int(Date.now.timeIntervalSince1970)
        return url.deletingPathExtension().appendingPathExtension("corrupt-\(stamp).json")
    }

    private static func quarantine(_ url: URL, reason: String) {
        let aside = asideURL(url)
        do {
            try FileManager.default.moveItem(at: url, to: aside)
            lock.withLock { _problems.append("\(url.lastPathComponent) could not be read and was set aside as \(aside.lastPathComponent)") }
        } catch {
            lock.withLock { _problems.append("\(url.lastPathComponent) could not be read (\(reason))") }
        }
    }

    private static func preserveCopy(_ url: URL, reason: String) {
        let aside = asideURL(url)
        try? FileManager.default.copyItem(at: url, to: aside)
        lock.withLock { _problems.append("\(url.lastPathComponent): \(reason); original kept as \(aside.lastPathComponent)") }
    }

    static func write<T: Encodable>(_ value: T, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        // completeUntilFirstUserAuthentication: readable by background tasks after first unlock.
        try encoder.encode(value).write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
    }
}
