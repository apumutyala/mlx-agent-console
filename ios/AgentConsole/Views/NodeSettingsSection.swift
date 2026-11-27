import SwiftUI

/// Settings → Compute node: serve fleet tasks from this iPad and pick up the
/// federation's averaged adapter.
struct NodeSettingsSection: View {
    let node: NodeService

    var body: some View {
        Section {
            Toggle("Serve tasks", isOn: Binding(get: { node.isServing }, set: { node.setServing($0) }))
            Toggle("Keep screen awake while serving", isOn: Binding(get: { node.keepAwake }, set: { node.keepAwake = $0 }))
            LabeledContent("Node ID") { Text(NodeService.nodeID).textSelection(.enabled).monospaced() }
            LabeledContent("Status", value: node.status)
            if let task = node.currentTask { LabeledContent("Current task", value: task) }
            LabeledContent("Tasks served", value: "\(node.tasksServed)")
            if let tps = node.lastTokensPerSecond {
                LabeledContent("Last speed", value: String(format: "%.1f tok/s", tps))
            }
            if let err = node.lastError { Text(err).font(.footnote).foregroundStyle(.secondary) }
            Button("Use federated adapter") { Task { await node.useFederatedAdapter() } }
                .disabled(node.isApplyingAdapter)
            if let s = node.adapterStatus { Text(s).font(.footnote) }
        } header: { Text("Compute node") } footer: {
            Text("Pulls generate and LoRA training tasks from the worker above. Only while the app is open: it pauses in the background and resumes when you return.")
        }
    }
}
