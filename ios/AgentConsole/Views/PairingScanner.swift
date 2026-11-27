import SwiftUI
import VisionKit

/// What `worker/pair.ps1` shows as a QR code on the worker machine:
/// `{"v":1,"url":"https://<host>.ts.net:8443","token":"<WORKER_TOKEN>"}`.
/// Scanning it fills in Settings so nobody has to type a 43-character token.
struct PairingPayload: Decodable, Equatable {
    let v: Int
    let url: URL
    let token: String

    /// Strict parse: anything that isn't exactly a v1 pairing payload is ignored,
    /// so a random QR code in view can't reconfigure the app.
    static func parse(_ text: String) -> PairingPayload? {
        guard text.count < 2048,
              let data = text.data(using: .utf8),
              let p = try? JSONDecoder().decode(PairingPayload.self, from: data),
              p.v == 1,
              let scheme = p.url.scheme?.lowercased(), scheme == "https" || scheme == "http",
              let host = p.url.host, !host.isEmpty,
              (24...256).contains(p.token.count),
              p.token.allSatisfy({ $0.isASCII && !$0.isWhitespace })
        else { return nil }
        return p
    }
}

/// Live camera QR scanner (VisionKit). Calls `onPayload` once, for the first valid code.
struct PairingScannerView: UIViewControllerRepresentable {
    let onPayload: (PairingPayload) -> Void

    static var isAvailable: Bool { DataScannerViewController.isSupported }

    func makeCoordinator() -> Coordinator { Coordinator(onPayload: onPayload) }

    func makeUIViewController(context: Context) -> DataScannerViewController {
        let scanner = DataScannerViewController(
            recognizedDataTypes: [.barcode(symbologies: [.qr])],
            qualityLevel: .balanced,
            recognizesMultipleItems: false,
            isHighFrameRateTrackingEnabled: false,
            isHighlightingEnabled: true)
        scanner.delegate = context.coordinator
        return scanner
    }

    func updateUIViewController(_ scanner: DataScannerViewController, context: Context) {
        // Start once the view is on screen (starting earlier is a no-op on some devices).
        if !scanner.isScanning, !context.coordinator.done {
            try? scanner.startScanning()
        }
    }

    static func dismantleUIViewController(_ scanner: DataScannerViewController, coordinator: Coordinator) {
        scanner.stopScanning()
    }

    @MainActor
    final class Coordinator: NSObject, DataScannerViewControllerDelegate {
        let onPayload: (PairingPayload) -> Void
        var done = false
        init(onPayload: @escaping (PairingPayload) -> Void) { self.onPayload = onPayload }

        func dataScanner(_ dataScanner: DataScannerViewController, didAdd addedItems: [RecognizedItem],
                         allItems: [RecognizedItem]) {
            guard !done else { return }
            for item in addedItems {
                guard case .barcode(let code) = item,
                      let text = code.payloadStringValue,
                      let payload = PairingPayload.parse(text) else { continue }
                done = true
                dataScanner.stopScanning()
                onPayload(payload)
                return
            }
        }
    }
}
