import Foundation
import MLX
import MLXVLM

/// Settings › Inkling lamq › self-check: downloads one layer of the 5.4 GB LookAtMyQuant build
/// (~190 MB, pinned revision) and checks the Metal dequant/matvec kernels against the reference
/// decoder's SHA-256 digests (ios/tools/lamq/lamq_digests.json). Does not load a model.
enum LamqSelfCheck {
    static let expected = InklingLamqDigests.layer7

    static var directory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("lamq-selfcheck/\(expected.revision.prefix(12))", isDirectory: true)
    }

    /// Downloads the check's files (skips ones already present), then runs it under the GPU gate.
    static func run(progress: @escaping @Sendable (String) -> Void) async throws -> [String] {
        let dir = directory
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in expected.files {
            let dest = dir.appendingPathComponent(file)
            if FileManager.default.fileExists(atPath: dest.path) { continue }
            progress("Downloading \(file)…")
            guard let url = URL(string: "https://huggingface.co/\(expected.repo)/resolve/\(expected.revision)/\(file)") else {
                throw URLError(.badURL)
            }
            let (tmp, response) = try await URLSession.shared.download(from: url)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: tmp, to: dest)
        }
        await GPUGate.shared.acquire("lamq self-check")
        progress("Running kernels…")
        let result: InklingLamqSelfCheck.Result
        do {
            result = try await Task.detached(priority: .userInitiated) {
                try InklingLamqSelfCheck.run(directory: dir, expected: expected) { progress($0) }
            }.value
        } catch {
            MLX.Memory.clearCache()
            await GPUGate.shared.release()
            throw error
        }
        MLX.Memory.clearCache()
        await GPUGate.shared.release()
        let report = result.lines.joined(separator: "\n")
        UserDefaults.standard.set(report, forKey: reportKey)
        return result.lines
    }

    static let reportKey = "lamq.selfCheck.lastReport"

    /// Frees the downloaded layer (~190 MB).
    static func removeDownloads() {
        try? FileManager.default.removeItem(at: directory)
    }
}
