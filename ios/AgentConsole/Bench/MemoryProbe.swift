import Foundation
import Metal

/// Metal-side memory numbers for receipts. `currentAllocatedSize` counts every buffer
/// allocated on the device by this process (MLX and llama.cpp both use the system default
/// device); `recommendedMaxWorkingSetSize` is Metal's advice for the GPU working set.
enum MetalProbe {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    static var recommendedMaxWorkingSetBytes: UInt64? { device?.recommendedMaxWorkingSetSize }

    static var currentAllocatedBytes: UInt64? { device.map { UInt64($0.currentAllocatedSize) } }
}

/// A breadcrumb written before a benchmark load and cleared after the run. iOS kills an app
/// over its memory limit (jetsam) without any callback, so a breadcrumb still present at the
/// next launch means the run died: most likely jetsam. `BenchCoordinator` turns it into a
/// `killed` receipt with `jetsam_suspected: true`.
enum JetsamBreadcrumb {
    struct Crumb: Codable, Sendable {
        var engine: String            // mlx | xybrid
        var entryID: String
        var displayName: String
        var repo: String
        var abStep: String?
        var phase: String             // e.g. "first load", "decode ab-256 rep 2"
        var startedAt: Date
        var updatedAt: Date
        var availableBeforeBytes: UInt64
        var footprintAtUpdateBytes: UInt64
        var metalAllocatedAtUpdateBytes: UInt64?
        var noResidency: Bool
        var forced: Bool
        /// Set when the next launch found the crumb (the run never cleared it).
        var foundAtLaunch: Date?
    }

    private static let pendingKey = "bench.jetsam.pending"
    private static let killedKey = "bench.jetsam.killed"
    private static let lock = NSLock()

    /// Call once per launch, before any benchmark: a crumb left by the previous process
    /// becomes the "killed" record.
    static func noteLaunch() {
        lock.withLock {
            let d = UserDefaults.standard
            guard let data = d.data(forKey: pendingKey),
                  var crumb = try? JSONDecoder().decode(Crumb.self, from: data) else { return }
            crumb.foundAtLaunch = Date()
            if let out = try? JSONEncoder().encode(crumb) { d.set(out, forKey: killedKey) }
            d.removeObject(forKey: pendingKey)
        }
    }

    static func set(_ crumb: Crumb) {
        lock.withLock {
            if let data = try? JSONEncoder().encode(crumb) { UserDefaults.standard.set(data, forKey: pendingKey) }
        }
    }

    /// Updates the phase (and memory readings) of the current crumb, if any.
    static func update(phase: String) {
        lock.withLock {
            let d = UserDefaults.standard
            guard let data = d.data(forKey: pendingKey),
                  var crumb = try? JSONDecoder().decode(Crumb.self, from: data) else { return }
            crumb.phase = phase
            crumb.updatedAt = Date()
            crumb.footprintAtUpdateBytes = ResourceProbe.physFootprint()
            crumb.metalAllocatedAtUpdateBytes = MetalProbe.currentAllocatedBytes
            if let out = try? JSONEncoder().encode(crumb) { d.set(out, forKey: pendingKey) }
        }
    }

    static func clear() {
        lock.withLock { UserDefaults.standard.removeObject(forKey: pendingKey) }
    }

    /// The run the previous process died in, once; nil when there is none.
    static func takeKilled() -> Crumb? {
        lock.withLock {
            let d = UserDefaults.standard
            defer { d.removeObject(forKey: killedKey) }
            guard let data = d.data(forKey: killedKey) else { return nil }
            return try? JSONDecoder().decode(Crumb.self, from: data)
        }
    }

    static func make(engine: String, entry: ModelCatalog.Entry, abStep: String?, phase: String,
                     noResidency: Bool, forced: Bool) -> Crumb {
        Crumb(engine: engine, entryID: entry.id, displayName: entry.displayName, repo: entry.repo,
              abStep: abStep, phase: phase, startedAt: Date(), updatedAt: Date(),
              availableBeforeBytes: ResourceProbe.availableMemory(),
              footprintAtUpdateBytes: ResourceProbe.physFootprint(),
              metalAllocatedAtUpdateBytes: MetalProbe.currentAllocatedBytes,
              noResidency: noResidency, forced: forced, foundAtLaunch: nil)
    }
}
