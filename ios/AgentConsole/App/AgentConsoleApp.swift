import SwiftUI

/// Composition root. One place builds the backend, stores and coordinators.
@MainActor
final class AppModel {
    static let shared = AppModel()

    let backend: any ModelBackend = MLXBackend()
    let documents = DocumentStore()
    let batches = BatchStore()
    lazy var summarizer = DocumentSummarizer(backend: backend, documents: documents)
    lazy var batchCoordinator = BatchCoordinator(
        runner: BatchRunner(store: batches, summarizer: summarizer, documents: documents))
    lazy var agent = LocalAgent(backend: backend, documents: documents) { [unowned self] in
        ToolRegistry(
            summarize: SummarizeImportedFileTool(documents: documents, summarizer: summarizer),
            submitJob: (try? WorkerClient.fromSettings()).map { SubmitWorkerJobTool(client: $0) })
    }
    lazy var node = NodeService(
        backend: backend, documents: documents,
        trainer: (backend as? MLXBackend).map { FederatedTrainer(backend: $0) },
        onModelChanged: { [unowned self] in self.agent.invalidateConversation() })
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        // BGTaskScheduler handlers must be registered before launch completes.
        BatchCoordinator.register { AppModel.shared.batchCoordinator }
        return true
    }
}

@main
struct AgentConsoleApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView(model: AppModel.shared)
        }
        .onChange(of: scenePhase) { _, phase in
            // Foreground-only batches stop at a file boundary instead of being
            // killed mid-generation (Metal work isn't allowed in background).
            if phase == .background, !BatchCoordinator.backgroundGPUAvailable {
                AppModel.shared.batchCoordinator.pauseForeground()
            }
            AppModel.shared.node.scenePhaseChanged(phase)
        }
    }
}

struct RootView: View {
    let model: AppModel
    var body: some View {
        TabView {
            Tab("Agent", systemImage: "sparkles") { ChatView(agent: model.agent, backend: model.backend) }
            Tab("Documents", systemImage: "doc.text") {
                DocumentsView(documents: model.documents, batches: model.batches, coordinator: model.batchCoordinator)
            }
            Tab("Jobs", systemImage: "server.rack") { JobsListView() }
            Tab("Settings", systemImage: "gear") { SettingsView() }
        }
        .task { model.node.resumeIfEnabled() }   // "Serve tasks" survives relaunch
    }
}
