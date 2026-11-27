import SwiftUI
import Xybrid

/// Tap to talk, tap again to stop: the transcript is sent to the agent.
/// Auto-stops (the 2-minute limit) and retries deliver through the same callback.
struct SpeechMicButton: View {
    let onTranscript: (String) -> Void
    private var speech: SpeechService { .shared }

    var body: some View {
        Button {
            Task {
                if speech.isRecording {
                    await speech.finishRecording()
                } else {
                    await speech.startRecording()
                }
            }
        } label: {
            Label(speech.isRecording ? "Stop recording" : "Speak",
                  systemImage: speech.isRecording ? "mic.circle.fill" : "mic.circle")
                .foregroundStyle(speech.isRecording ? Color.red : Color.accentColor)
                .symbolEffect(.pulse, isActive: speech.isRecording)
        }
        .disabled(speech.isBusy && !speech.isRecording)
        .onAppear { speech.onTranscript = onTranscript }
    }
}

/// Toolbar toggle: read assistant replies aloud.
struct SpeakerToggle: View {
    var body: some View {
        @Bindable var speech = SpeechService.shared
        Toggle(isOn: $speech.speakReplies) {
            Label("Read replies aloud",
                  systemImage: speech.speakReplies ? "speaker.wave.2.fill" : "speaker.slash")
        }
        .toggleStyle(.button)
    }
}

/// Status line for downloads (first use needs network), loading, transcription, errors.
struct SpeechStatusBar: View {
    private var speech: SpeechService { .shared }

    var body: some View {
        Group {
            switch speech.phase {
            case .downloading(let model, let p):
                ProgressView("Downloading \(model) (first use, needs network)…", value: p)
            case .loading(let model):
                ProgressView("Loading \(model)…")
            case .recording:
                Label("Listening… tap the mic again to send (\(MicRecorder.maxSeconds / 60) min max)", systemImage: "waveform")
            case .transcribing:
                ProgressView("Transcribing…")
            case .synthesizing, .speaking, .idle:
                VStack(alignment: .leading, spacing: 4) {
                    if let err = speech.lastError {
                        Label(err, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                    }
                    if speech.pendingAudio != nil, !speech.isBusy {
                        HStack {
                            Button("Send recording", systemImage: "arrow.clockwise") {
                                Task { await speech.retryTranscription() }
                            }
                            .disabled(speech.hostIsBusy)
                            Button("Discard", systemImage: "trash", role: .destructive) {
                                speech.discardPendingAudio()
                            }
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
        }
        .font(.footnote)
        .padding(.horizontal)
    }
}

/// Settings: Kokoro voice picker. The list comes from the model itself.
struct SpeechSettingsSection: View {
    var body: some View {
        @Bindable var speech = SpeechService.shared
        Section {
            if speech.voices.isEmpty {
                LabeledContent("Voice", value: speech.voiceId)
                Button("Load voice list", systemImage: "arrow.down.circle") { speech.loadVoices() }
                    .disabled(speech.loadingModel == SpeechService.ttsModelId)
            } else {
                Picker("Voice", selection: $speech.voiceId) {
                    ForEach(speech.voices, id: \.id) { v in
                        Text(Self.label(v)).tag(v.id)
                    }
                }
            }
            Button("Preview voice", systemImage: "play.circle") {
                speech.speak("Hi! This is how replies will sound.")
            }
            .disabled(speech.isRecording || speech.isBusy)
        } header: { Text("Speech") } footer: {
            Text("Read-aloud uses Kokoro on the device. The voice list comes from the model, which downloads (about 180 MB) on first use.")
        }
    }

    private static func label(_ v: XybridVoiceInfo) -> String {
        [Optional(v.name), v.language, v.style].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
    }
}

extension View {
    /// Speaks each finished assistant reply when the speaker toggle is on.
    /// A stopped run ends with a "Stopped." turn, so a truncated reply is never read.
    func speechReadsReplies(_ agent: LocalAgent) -> some View {
        onChange(of: agent.isRunning) { _, running in
            let speech = SpeechService.shared
            speech.hostIsBusy = running
            if running {
                speech.stopSpeaking()
                // A typed message went out while the mic was live: keep the audio, don't send it.
                speech.suspendRecording("Recording paused because a message was sent. Your audio was kept: tap Send or Discard.")
            } else if let last = agent.turns.last, last.role == .assistant {
                speech.speakIfEnabled(last.text)
            }
        }
    }
}
