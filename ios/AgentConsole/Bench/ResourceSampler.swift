import Darwin
import Foundation
import os

/// Process memory and thermal probes for benchmark receipts.
enum ResourceProbe {
    /// The figure jetsam enforces against: `task_vm_info.phys_footprint`, in bytes.
    static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }

    /// Bytes this process can still allocate before hitting its jetsam limit.
    static func availableMemory() -> UInt64 { UInt64(os_proc_available_memory()) }

    static func thermal() -> ProcessInfo.ThermalState { ProcessInfo.processInfo.thermalState }

    static func name(_ t: ProcessInfo.ThermalState) -> String {
        switch t {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}

/// Samples footprint, available memory and thermal state every `interval`
/// on its own queue while a benchmark rep runs (the MLX work blocks its thread).
final class ResourceSampler: @unchecked Sendable {
    struct Summary: Sendable {
        var peakFootprint: UInt64
        var minAvailable: UInt64
        var maxThermal: ProcessInfo.ThermalState
        var samples: Int
    }

    static let intervalMs = 50

    private let lock = NSLock()
    private let queue = DispatchQueue(label: "bench.sampler", qos: .userInitiated)
    private var timer: DispatchSourceTimer?
    private var summary = Summary(peakFootprint: 0, minAvailable: .max, maxThermal: .nominal, samples: 0)

    func start() {
        sample()
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + .milliseconds(Self.intervalMs), repeating: .milliseconds(Self.intervalMs),
                   leeway: .milliseconds(5))
        t.setEventHandler { [weak self] in self?.sample() }
        lock.withLock { timer = t }
        t.resume()
    }

    /// Stops sampling (taking one last sample) and returns the extremes seen.
    func stop() -> Summary {
        let t: DispatchSourceTimer? = lock.withLock { defer { timer = nil }; return timer }
        t?.cancel()
        queue.sync {}   // let an in-flight sample finish
        sample()
        return lock.withLock { summary }
    }

    private func sample() {
        let footprint = ResourceProbe.physFootprint()
        let available = ResourceProbe.availableMemory()
        let thermal = ResourceProbe.thermal()
        lock.withLock {
            summary.peakFootprint = max(summary.peakFootprint, footprint)
            summary.minAvailable = min(summary.minAvailable, available)
            if thermal.rawValue > summary.maxThermal.rawValue { summary.maxThermal = thermal }
            summary.samples += 1
        }
    }
}
