import AppIntents
import Foundation

/// Start/stop serving fleet tasks. Opens the app because the node only runs
/// in the foreground (GPU + networking).
struct ToggleComputeNodeIntent: AppIntent {
    static let title: LocalizedStringResource = "Toggle Compute Node"
    static let description = IntentDescription("Start or stop serving fleet tasks from this iPad.")
    static let openAppWhenRun: Bool = true

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let node = AppModel.shared.node
        node.setServing(!node.isServing)
        let text = node.isServing ? "Compute node is serving tasks." : "Compute node stopped."
        return .result(dialog: "\(text)")
    }
}
