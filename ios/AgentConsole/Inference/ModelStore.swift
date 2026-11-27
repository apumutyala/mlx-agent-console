import CryptoKit
import Foundation
import HuggingFace

/// Where model weights live on disk, so nothing is downloaded twice.
///
/// - MLX / Hugging Face snapshots: `Application Support/Models/huggingface/hub` (same
///   `models--org--repo/{blobs,refs,snapshots}` layout as Python). Application Support is
///   never purged by iOS (Library/Caches is), and the `Models` folder is excluded from
///   iCloud backup.
/// - GGUF files for the Xybrid engine: `Application Support/Models/gguf/<entry id>/<file>`,
///   one directory per file (Xybrid writes `model_metadata.json` beside the file).
/// - Xybrid speech/registry models: the SDK's own `Application Support/Xybrid/Models`,
///   excluded from backup here. `Xybrid.initSdkCacheDir` must never be called: it sets
///   `HF_HOME`, which would move the default Hugging Face cache (we pass an explicit cache,
///   so MLX is safe either way, but the SDK would still lose its own cache).
///
/// The first access to `cache` migrates the old `Library/Caches/huggingface/hub` folder
/// (a same-volume move: instant, symlinks stay valid) and records a report.
enum ModelStore {
    // MARK: Locations

    static var modelsRoot: URL {
        URL.applicationSupportDirectory.appending(path: "Models", directoryHint: .isDirectory)
    }
    static var hubRoot: URL { modelsRoot.appending(path: "huggingface/hub", directoryHint: .isDirectory) }
    static var ggufRoot: URL { modelsRoot.appending(path: "gguf", directoryHint: .isDirectory) }
    /// Where swift-huggingface's `HubCache.default` put snapshots before this store existed.
    static var legacyHubRoot: URL { URL.cachesDirectory.appending(path: "huggingface/hub", directoryHint: .isDirectory) }
    /// Xybrid's default SDK cache on iOS (crates/xybrid-sdk/src/cache/cache_manager.rs, v0.21.0).
    static var xybridRoot: URL { URL.applicationSupportDirectory.appending(path: "Xybrid", directoryHint: .isDirectory) }
    private static var manifestURL: URL { modelsRoot.appending(path: "manifest.json") }

    /// The explicit cache every Hugging Face download goes through (MLX chat models, the
    /// memory embedder, benchmarks). Creating it runs the one-time migration first.
    static let cache: HubCache = {
        prepare()
        return HubCache(cacheDirectory: hubRoot)
    }()

    /// Hub client bound to `cache`. Use with `#hubDownloader(ModelStore.hub)`.
    static let hub = HubClient(cache: cache)

    // MARK: Migration

    static let migratedKey = "store.migrated.v1"
    static let migrationReportKey = "store.migrationReport"

    /// Creates the folders, migrates the old Caches snapshots once, and excludes the model
    /// folders from backup. Idempotent; runs inside `cache`'s initializer (once per process).
    static func prepare() {
        let fm = FileManager.default
        try? fm.createDirectory(at: modelsRoot, withIntermediateDirectories: true)
        let defaults = UserDefaults.standard
        if !defaults.bool(forKey: migratedKey) {
            let report = migrateLegacyCache()
            defaults.set(report.text, forKey: migrationReportKey)
            if report.complete { defaults.set(true, forKey: migratedKey) }
        }
        try? fm.createDirectory(at: hubRoot, withIntermediateDirectories: true)
        try? fm.createDirectory(at: ggufRoot, withIntermediateDirectories: true)
        excludeFromBackup(modelsRoot)
        excludeXybridFromBackup()
    }

    struct MigrationReport {
        var text: String
        /// Everything that could move did; the flag is set and the migration never reruns.
        var complete: Bool
    }

    /// Moves `Caches/huggingface/hub` into Application Support. When the destination is
    /// new, the whole folder moves in one rename. When both exist, each `models--*` folder
    /// (and its `.metadata` entry) that the destination lacks moves; duplicates stay where
    /// they are and are listed in the report (nothing is deleted).
    static func migrateLegacyCache() -> MigrationReport {
        let fm = FileManager.default
        let src = legacyHubRoot
        guard fm.fileExists(atPath: src.path) else {
            return .init(text: "Nothing to migrate (no Library/Caches/huggingface/hub).", complete: true)
        }
        let before = treeBytes(src)
        do {
            if !fm.fileExists(atPath: hubRoot.path) {
                try fm.createDirectory(at: hubRoot.deletingLastPathComponent(), withIntermediateDirectories: true)
                try fm.moveItem(at: src, to: hubRoot)
                let after = treeBytes(hubRoot)
                let ok = after.bytes == before.bytes && after.files == before.files
                return .init(text: String(
                    format: "Moved Caches/huggingface/hub → Application Support (%ld files, %.2f GB before, %.2f GB after%@).",
                    after.files, Double(before.bytes) / 1e9, Double(after.bytes) / 1e9,
                    ok ? ", sizes match" : ", SIZE MISMATCH"), complete: true)
            }
            var moved: [String] = []
            var kept: [String] = []
            for name in (try? fm.contentsOfDirectory(atPath: src.path)) ?? [] where name.hasPrefix("models--") {
                let to = hubRoot.appending(path: name)
                if fm.fileExists(atPath: to.path) { kept.append(name); continue }
                try fm.moveItem(at: src.appending(path: name), to: to)
                let meta = src.appending(path: ".metadata/\(name)")
                let metaTo = hubRoot.appending(path: ".metadata/\(name)")
                if fm.fileExists(atPath: meta.path), !fm.fileExists(atPath: metaTo.path) {
                    try? fm.createDirectory(at: metaTo.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try? fm.moveItem(at: meta, to: metaTo)
                }
                moved.append(name)
            }
            return .init(text: "Merged into Application Support: moved \(moved.count) repo(s) "
                         + "[\(moved.joined(separator: ", "))]; left \(kept.count) duplicate(s) in Caches "
                         + "[\(kept.joined(separator: ", "))].", complete: true)
        } catch {
            // Not marked complete: the next launch tries again (a move is all-or-nothing per item).
            return .init(text: "Migration failed: \(error.localizedDescription)", complete: false)
        }
    }

    // MARK: Local-first resolution

    /// A complete local snapshot of `repo` at the commit `refs/main` points to, or nil.
    /// Complete = config.json, a tokenizer file and every weight shard exist (through their
    /// symlinks), and, when a size manifest exists for that commit, every size matches it.
    /// Blobs are written to `.incomplete` and renamed when done, so an existing file is whole.
    static func localSnapshot(repo: String) -> URL? {
        guard let id = Repo.ID(rawValue: repo),
              let commit = cache.resolveRevision(repo: id, kind: .model, ref: "main"),
              let snap = try? cache.snapshotPath(repo: id, kind: .model, commitHash: commit)
        else { return nil }
        guard let files = requiredFiles(in: snap) else { return nil }
        var sizes: [String: Int64] = [:]
        for name in files {
            guard let size = fileSize(snap.appending(path: name)), size > 0 else { return nil }
            sizes[name] = size
        }
        let manifest = loadManifest()
        if let recorded = manifest[repo], recorded.commit == commit {
            for (name, size) in recorded.files where sizes[name] != nil && sizes[name] != size { return nil }
        } else {
            // First sight of this snapshot (e.g. migrated): record what is there.
            recordManifest(repo: repo, commit: commit, sizes: sizes)
        }
        return snap
    }

    /// After a network load, remember the snapshot's file sizes so later loads can verify it.
    static func noteDownloaded(repo: String) {
        guard let id = Repo.ID(rawValue: repo),
              let commit = cache.resolveRevision(repo: id, kind: .model, ref: "main"),
              let snap = try? cache.snapshotPath(repo: id, kind: .model, commitHash: commit),
              let files = requiredFiles(in: snap) else { return }
        var sizes: [String: Int64] = [:]
        for name in files { if let s = fileSize(snap.appending(path: name)) { sizes[name] = s } }
        recordManifest(repo: repo, commit: commit, sizes: sizes)
    }

    /// config.json + tokenizer + weight shards, or nil if any required piece is missing.
    private static func requiredFiles(in snap: URL) -> [String]? {
        let fm = FileManager.default
        func exists(_ name: String) -> Bool { fm.fileExists(atPath: snap.appending(path: name).path) }
        guard exists("config.json") else { return nil }
        var files = ["config.json"]
        if exists("tokenizer.json") { files.append("tokenizer.json") }
        else if exists("tokenizer_config.json") { files.append("tokenizer_config.json") }
        else { return nil }
        let index = snap.appending(path: "model.safetensors.index.json")
        if let data = try? Data(contentsOf: index),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = json["weight_map"] as? [String: Any] {
            let shards = Set(map.values.compactMap { $0 as? String }).sorted()
            guard !shards.isEmpty, shards.allSatisfy(exists) else { return nil }
            files.append(contentsOf: shards)
        } else {
            let shards = ((try? fm.contentsOfDirectory(atPath: snap.path)) ?? [])
                .filter { $0.hasSuffix(".safetensors") }.sorted()
            guard !shards.isEmpty else { return nil }
            files.append(contentsOf: shards)
        }
        return files
    }

    // MARK: Manifest

    struct ManifestEntry: Codable {
        var commit: String
        var files: [String: Int64]
        var recordedAt: Date
    }

    private static let manifestLock = NSLock()

    static func loadManifest() -> [String: ManifestEntry] {
        manifestLock.withLock {
            guard let data = try? Data(contentsOf: manifestURL) else { return [:] }
            return (try? JSONDecoder().decode([String: ManifestEntry].self, from: data)) ?? [:]
        }
    }

    private static func recordManifest(repo: String, commit: String, sizes: [String: Int64]) {
        manifestLock.withLock {
            var all: [String: ManifestEntry] = [:]
            if let data = try? Data(contentsOf: manifestURL) {
                all = (try? JSONDecoder().decode([String: ManifestEntry].self, from: data)) ?? [:]
            }
            all[repo] = ManifestEntry(commit: commit, files: sizes, recordedAt: Date())
            if let data = try? JSONEncoder().encode(all) { try? data.write(to: manifestURL, options: .atomic) }
        }
    }

    private static func dropManifest(repo: String) {
        manifestLock.withLock {
            guard let data = try? Data(contentsOf: manifestURL),
                  var all = try? JSONDecoder().decode([String: ManifestEntry].self, from: data) else { return }
            all[repo] = nil
            if let out = try? JSONEncoder().encode(all) { try? out.write(to: manifestURL, options: .atomic) }
        }
    }

    // MARK: Sizes

    /// Size of the file a (possibly symlinked) URL points to. Snapshot files are relative
    /// symlinks into `blobs/`, so `.fileSizeKey` on the link itself is the link length (the
    /// `weight_files_bytes: 76` bug).
    static func fileSize(_ url: URL) -> Int64? {
        let real = url.resolvingSymlinksInPath()
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: real.path),
              let n = attrs[.size] as? NSNumber else { return nil }
        return n.int64Value
    }

    /// Sum of the resolved sizes of the `*.safetensors` files in a model directory.
    static func weightFilesBytes(in dir: URL) -> Int64? {
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .filter { $0.hasSuffix(".safetensors") }
        let sizes = names.compactMap { fileSize(dir.appending(path: $0)) }
        return sizes.isEmpty ? nil : sizes.reduce(0, +)
    }

    /// Total bytes of a repo's `blobs/` (what is really on disk; snapshots are symlinks).
    static func blobBytes(repo: String) -> Int64 {
        guard let id = Repo.ID(rawValue: repo) else { return 0 }
        return treeBytes(cache.blobsDirectory(repo: id, kind: .model)).bytes
    }

    /// Regular files under `dir` (symlinks not followed) and their total size.
    static func treeBytes(_ dir: URL) -> (files: Int, bytes: Int64) {
        guard let e = FileManager.default.enumerator(
            at: dir, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey, .isSymbolicLinkKey]) else { return (0, 0) }
        var files = 0
        var bytes: Int64 = 0
        for case let url as URL in e {
            guard let v = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey, .isSymbolicLinkKey]),
                  v.isRegularFile == true, v.isSymbolicLink != true else { continue }
            files += 1
            bytes += Int64(v.fileSize ?? 0)
        }
        return (files, bytes)
    }

    static var freeBytes: Int64 {
        let v = try? URL.applicationSupportDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage ?? 0
    }

    // MARK: Downloaded-models listing (Settings)

    struct StoredFile: Hashable, Sendable {
        let name: String
        /// Resolved size (through the snapshot symlink).
        let bytes: Int64
    }

    struct StoredModel: Identifiable, Sendable {
        enum Kind: String, Sendable { case huggingFace = "MLX", gguf = "GGUF", xybrid = "Xybrid" }
        var id: String { "\(kind.rawValue):\(name)" }
        let kind: Kind
        /// Repo id (`org/name`), GGUF entry id, or Xybrid model id.
        let name: String
        let location: URL
        let commit: String?
        let bytes: Int64
        /// Files with resolved sizes, largest first.
        let files: [StoredFile]
    }

    static func storedModels() -> [StoredModel] {
        var out: [StoredModel] = []
        let fm = FileManager.default
        for dir in ((try? fm.contentsOfDirectory(at: hubRoot, includingPropertiesForKeys: nil)) ?? [])
        where dir.lastPathComponent.hasPrefix("models--") {
            let repo = dir.lastPathComponent.dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")
            let commit = (try? String(contentsOf: dir.appending(path: "refs/main"), encoding: .utf8))?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            var files: [StoredFile] = []
            if let commit {
                let snap = dir.appending(path: "snapshots/\(commit)")
                for name in (try? fm.contentsOfDirectory(atPath: snap.path)) ?? [] {
                    if let s = fileSize(snap.appending(path: name)) { files.append(StoredFile(name: name, bytes: s)) }
                }
            }
            out.append(StoredModel(kind: .huggingFace, name: repo, location: dir, commit: commit,
                                   bytes: treeBytes(dir.appending(path: "blobs")).bytes,
                                   files: files.sorted { $0.bytes > $1.bytes }))
        }
        for dir in ((try? fm.contentsOfDirectory(at: ggufRoot, includingPropertiesForKeys: nil)) ?? []) {
            var files: [StoredFile] = []
            for name in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? [] {
                if let s = fileSize(dir.appending(path: name)) { files.append(StoredFile(name: name, bytes: s)) }
            }
            guard !files.isEmpty else { continue }
            out.append(StoredModel(kind: .gguf, name: dir.lastPathComponent, location: dir, commit: nil,
                                   bytes: files.reduce(0) { $0 + $1.bytes }, files: files.sorted { $0.bytes > $1.bytes }))
        }
        return out
    }

    /// Deletes a Hugging Face repo's cache folder (and its metadata) or a GGUF folder.
    /// The caller must have unloaded the model first. Xybrid entries go through the SDK
    /// (`Xybrid.removeCachedModel`), not here.
    static func delete(_ model: StoredModel) throws {
        let fm = FileManager.default
        switch model.kind {
        case .huggingFace:
            try fm.removeItem(at: model.location)
            let meta = hubRoot.appending(path: ".metadata/\(model.location.lastPathComponent)")
            if fm.fileExists(atPath: meta.path) { try? fm.removeItem(at: meta) }
            dropManifest(repo: model.name)
        case .gguf:
            try fm.removeItem(at: model.location)
        case .xybrid:
            break
        }
    }

    // MARK: GGUF files (Xybrid engine)

    /// `Application Support/Models/gguf/<entryID>/<file>`.
    static func ggufURL(entryID: String, file: String) -> URL {
        ggufRoot.appending(path: entryID, directoryHint: .isDirectory).appending(path: file)
    }

    /// The GGUF exists with the expected size (or any non-zero size when none is known).
    static func ggufIsComplete(_ url: URL, expectedBytes: Int64?) -> Bool {
        guard let size = fileSize(url), size > 0 else { return false }
        return expectedBytes.map { $0 == size } ?? true
    }

    // MARK: Backup exclusion

    static func excludeFromBackup(_ url: URL) {
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        var u = url
        var v = URLResourceValues()
        v.isExcludedFromBackup = true
        try? u.setResourceValues(v)
    }

    /// The Xybrid SDK creates its folder lazily; create it ourselves so the flag can be set
    /// before the first speech/LLM download (the SDK uses an existing folder as is).
    static func excludeXybridFromBackup() {
        try? FileManager.default.createDirectory(at: xybridRoot, withIntermediateDirectories: true)
        excludeFromBackup(xybridRoot)
    }

    static func isExcludedFromBackup(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isExcludedFromBackupKey]))?.isExcludedFromBackup ?? false
    }
}

/// Identity facts recorded in every receipt, to prove (or disprove) that the app container,
/// and therefore the downloaded weights, survived a Sideloadly reinstall.
enum ContainerIdentity {
    static var bundleIdentifier: String { Bundle.main.bundleIdentifier ?? "unknown" }

    /// SHA-256 (first 16 hex) of the data container path. iOS may move the container on an
    /// update while keeping its contents, so a changed hash alone is not proof of data loss;
    /// `markerID` is the real check.
    static var pathHash: String {
        let digest = SHA256.hash(data: Data(NSHomeDirectory().utf8))
        return digest.map { String(format: "%02x", $0) }.joined().prefix(16).description
    }

    private static var markerURL: URL {
        URL.applicationSupportDirectory.appending(path: "container-marker.json")
    }

    struct Marker: Codable {
        var id: String
        var createdAt: Date
        var bundleIdentifier: String
    }

    /// A random id written into Application Support on first launch. If the same id shows up
    /// in a receipt after a reinstall, the container (and the weights in it) persisted.
    static let marker: Marker = {
        let fm = FileManager.default
        if let data = try? Data(contentsOf: markerURL),
           let m = try? JSONDecoder().decode(Marker.self, from: data) { return m }
        let m = Marker(id: UUID().uuidString.lowercased(), createdAt: Date(), bundleIdentifier: bundleIdentifier)
        try? fm.createDirectory(at: URL.applicationSupportDirectory, withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(m) { try? data.write(to: markerURL, options: .atomic) }
        return m
    }()
}
