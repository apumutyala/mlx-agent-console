import AppIntents
import Foundation

// The app's own actions, exposed to Siri, Shortcuts and Spotlight.
// They talk to the worker; the phone never becomes the job runner.

struct JobTypeEntity: AppEntity {
    static let typeDisplayRepresentation: TypeDisplayRepresentation = "Job Type"
    static let defaultQuery = JobTypeQuery()
    let id: String
    let summary: String
    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(id)", subtitle: "\(summary)")
    }
}

struct JobTypeQuery: EntityQuery {
    func entities(for identifiers: [String]) async throws -> [JobTypeEntity] {
        try await suggestedEntities().filter { identifiers.contains($0.id) }
    }
    func suggestedEntities() async throws -> [JobTypeEntity] {
        try await WorkerClient.fromSettings().jobTypes()
            // Shortcuts can only start job types whose params all have defaults.
            .filter { $0.params.values.allSatisfy { $0.default != nil } }
            .map { JobTypeEntity(id: $0.name, summary: $0.description) }
    }
}

struct JobEntity: AppEntity {
    static let typeDisplayRepresentation: TypeDisplayRepresentation = "Worker Job"
    static let defaultQuery = JobQuery()
    let id: String
    let type: String
    let status: String
    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(type)", subtitle: "\(status) · \(id)")
    }
    init(_ j: RemoteJob) { id = j.id; type = j.type; status = j.status }
}

struct JobQuery: EntityQuery {
    func entities(for identifiers: [String]) async throws -> [JobEntity] {
        let c = try WorkerClient.fromSettings()
        var out: [JobEntity] = []
        for id in identifiers { out.append(JobEntity(try await c.job(id))) }
        return out
    }
    func suggestedEntities() async throws -> [JobEntity] {
        try await WorkerClient.fromSettings().jobs().prefix(20).map(JobEntity.init)
    }
}

struct StartJobIntent: AppIntent {
    static let title: LocalizedStringResource = "Start Job"
    static let description = IntentDescription("Start a named job on your worker.")

    @Parameter(title: "Job type") var jobType: JobTypeEntity

    func perform() async throws -> some IntentResult & ReturnsValue<JobEntity> & ProvidesDialog {
        let job = try await WorkerClient.fromSettings().submit(type: jobType.id, params: [:])
        let note = job.status == "waiting_approval" ? " It's waiting for your approval in the app." : ""
        return .result(value: JobEntity(job), dialog: "Started \(job.type).\(note)")
    }
}

struct GetJobStatusIntent: AppIntent {
    static let title: LocalizedStringResource = "Get Job Status"
    static let description = IntentDescription("Check a worker job's status.")

    @Parameter(title: "Job") var job: JobEntity?

    func perform() async throws -> some IntentResult & ReturnsValue<String> & ProvidesDialog {
        let c = try WorkerClient.fromSettings()
        let target: RemoteJob
        if let job { target = try await c.job(job.id) }
        else if let latest = try await c.jobs().first { target = latest }
        else { return .result(value: "none", dialog: "No jobs yet.") }
        var text = "\(target.type) is \(target.status.replacingOccurrences(of: "_", with: " "))."
        if let a = target.pendingApprovals.first { text += " Waiting on: \(a.prompt)" }
        return .result(value: target.status, dialog: "\(text)")
    }
}

struct CancelJobIntent: AppIntent {
    static let title: LocalizedStringResource = "Cancel Job"
    static let description = IntentDescription("Cancel a running worker job.")

    @Parameter(title: "Job") var job: JobEntity

    func perform() async throws -> some IntentResult & ProvidesDialog {
        try await requestConfirmation(result: .result(dialog: "Cancel \(job.type)?"))
        let j = try await WorkerClient.fromSettings().cancel(job.id)
        return .result(dialog: "Cancelling \(j.type).")
    }
}

struct AgentConsoleShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(intent: StartJobIntent(), phrases: ["Start a job in \(.applicationName)"],
                    shortTitle: "Start Job", systemImageName: "play.circle")
        AppShortcut(intent: GetJobStatusIntent(), phrases: ["Check my job in \(.applicationName)",
                                                            "Job status in \(.applicationName)"],
                    shortTitle: "Job Status", systemImageName: "list.bullet.rectangle")
        AppShortcut(intent: CancelJobIntent(), phrases: ["Cancel a job in \(.applicationName)"],
                    shortTitle: "Cancel Job", systemImageName: "xmark.octagon")
        AppShortcut(intent: ToggleComputeNodeIntent(), phrases: ["Toggle compute node in \(.applicationName)"],
                    shortTitle: "Compute Node", systemImageName: "cpu")
    }
}
