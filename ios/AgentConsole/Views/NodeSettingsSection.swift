import SwiftUI

/// Settings → Compute node: serve fleet tasks from this iPad and pick up the
/// federation's averaged adapter.
struct NodeSettingsSection: View {
    let node: NodeService

    var body: some View {
        Section {
            Toggle("Serve tasks", isOn: Binding(get: { node.isServing }, set: { node.setServing($0) }))
            Toggle("Keep screen awake while serving", isOn: Binding(get: { node.keepAwake }, set: { node.keepAwake = $0 }))
            Toggle("Accept training tasks", isOn: Binding(get: { node.acceptTraining }, set: { node.acceptTraining = $0 }))
            Toggle("Allow federated training on my documents",
                   isOn: Binding(get: { node.allowDocumentTraining }, set: { node.allowDocumentTraining = $0 }))
                .disabled(!node.acceptTraining)
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
            Text("Serving pulls generate tasks from the worker above. LoRA training tasks run only if you accept them, and they train on your imported documents only if you allow that; otherwise only the examples the round sends are used. The adapter trained on this iPad is uploaded to the worker. Training never changes the chat model; \"Use federated adapter\" does. Only while the app is open: it pauses when you leave and resumes when you return.")
        }
    }
}
