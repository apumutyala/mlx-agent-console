import SwiftUI

struct JobsListView: View {
    @State private var jobs: [RemoteJob] = []
    @State private var types: [JobTypeInfo] = []
    @State private var error: String?
    @State private var showingNew = false

    var body: some View {
        NavigationStack {
            List {
                if let error { Text(error).foregroundStyle(.red).font(.footnote) }
                ForEach(jobs) { job in
                    NavigationLink(value: job.id) {
                        VStack(alignment: .leading) {
                            Text(job.type).font(.headline)
                            HStack {
                                StatusBadge(status: job.status)
                                if !job.pendingApprovals.isEmpty {
                                    Label("needs approval", systemImage: "hand.raised.fill").foregroundStyle(.orange)
                                }
                                Spacer()
                                Text(Date(timeIntervalSince1970: job.createdAt), style: .relative)
                            }
                            .font(.caption)
                        }
                    }
                }
            }
            .navigationTitle("Worker jobs")
            .navigationDestination(for: String.self) { JobDetailView(feed: JobFeed(jobID: $0)) }
            .toolbar {
                Button("New job", systemImage: "plus") { showingNew = true }.disabled(types.isEmpty)
            }
            .refreshable { await load() }
            .task { await load() }
            .sheet(isPresented: $showingNew) {
                NewJobSheet(types: types) { await load() }
            }
        }
    }

    private func load() async {
        do {
            let c = try WorkerClient.fromSettings()
            async let j = c.jobs()
            async let t = c.jobTypes()
            (jobs, types) = try await (j, t)
            error = nil
        } catch { self.error = error.localizedDescription }
    }
}

struct StatusBadge: View {
    let status: String
    var color: Color {
        switch status {
        case "running": .blue
        case "succeeded": .green
        case "waiting_approval": .orange
        case "failed", "lost", "denied": .red
        default: .gray
        }
    }
    var body: some View {
        Text(status.replacingOccurrences(of: "_", with: " "))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }
}

struct NewJobSheet: View {
    let types: [JobTypeInfo]
    let onSubmit: () async -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var selected: JobTypeInfo?
    @State private var params: [String: String] = [:]
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                Picker("Job type", selection: $selected) {
                    Text("Choose…").tag(JobTypeInfo?.none)
                    ForEach(types, id: \.name) { Text($0.name).tag(Optional($0)) }
                }
                if let t = selected {
                    Text(t.description).font(.footnote)
                    ForEach(t.params.keys.sorted(), id: \.self) { key in
                        TextField(key, text: Binding(get: { params[key] ?? t.params[key]?.default ?? "" },
                                                     set: { params[key] = $0 }))
                            .textInputAutocapitalization(.never).autocorrectionDisabled()
                    }
                    if t.requiresApproval { Label("Starts after you approve it", systemImage: "hand.raised") }
                }
                if let error { Text(error).foregroundStyle(.red) }
            }
            .navigationTitle("New job")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Submit") {
                        guard let t = selected else { return }
                        Task {
                            do {
                                _ = try await WorkerClient.fromSettings().submit(type: t.name, params: params)
                                await onSubmit(); dismiss()
                            } catch { self.error = error.localizedDescription }
                        }
                    }.disabled(selected == nil)
                }
            }
        }
    }
}

struct JobDetailView: View {
    @State var feed: JobFeed
    @State private var actionError: String?
    @State private var confirmCancel = false
    @Environment(\.scenePhase) private var scenePhase
    @State private var followToken = 0

    var body: some View {
        List {
            Section {
                if let job = feed.job {
                    LabeledContent("Status") { StatusBadge(status: job.status) }
                    LabeledContent("Type", value: job.type)
                    if let code = job.exitCode { LabeledContent("Exit code", value: "\(code)") }
                    LabeledContent("Events", value: "\(feed.cursor)")
                }
                if let e = feed.connectionError {
                    Label("Reconnecting: \(e)", systemImage: "wifi.exclamationmark").foregroundStyle(.orange)
                }
                if let e = actionError { Text(e).foregroundStyle(.red) }
            }
            if !feed.pendingApprovals.isEmpty {
                Section("Needs your approval") {
                    ForEach(feed.pendingApprovals, id: \.id) { a in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(a.prompt)
                            HStack {
                                Button("Approve") { act { try await feed.decide(a.id, approve: true) } }
                                    .buttonStyle(.borderedProminent)
                                Button("Deny", role: .destructive) { act { try await feed.decide(a.id, approve: false) } }
                                    .buttonStyle(.bordered)
                            }
                        }
                    }
                }
            }
            Section("Log") {
                ForEach(feed.events.suffix(500)) { ev in
                    Text(ev.displayText)
                        .font(.caption.monospaced())
                        .foregroundStyle(ev.stream == "stderr" ? .red : (ev.kind == "log" ? .primary : .secondary))
                        .textSelection(.enabled)
                }
            }
        }
        .navigationTitle(feed.job?.type ?? feed.jobID)
        .toolbar {
            if let job = feed.job {
                if job.isTerminal {
                    Button("Retry", systemImage: "arrow.clockwise") { act { _ = try await feed.retry() } }
                } else {
                    Button("Cancel job", systemImage: "xmark.octagon", role: .destructive) { confirmCancel = true }
                }
            }
        }
        .confirmationDialog("Cancel this job? The worker stops the whole process group.",
                            isPresented: $confirmCancel, titleVisibility: .visible) {
            Button("Cancel job", role: .destructive) { act { try await feed.cancel() } }
        }
        // Re-follow when returning to the foreground; resumes from the saved cursor.
        .task(id: followToken) { await feed.follow() }
        .onChange(of: scenePhase) { _, p in if p == .active { followToken += 1 } }
    }

    private func act(_ f: @escaping () async throws -> Void) {
        Task {
            do { try await f(); actionError = nil } catch { actionError = error.localizedDescription }
        }
    }
}
